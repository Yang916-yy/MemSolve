#include "common.cuh"
#include <torch/library.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace memsolve_equilibrium {
namespace {
// Descriptors own host state only; PyTorch owns the stream-cached cuBLASLt workspace.
struct LinearGradientDescriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t algorithm{};
    ~LinearGradientDescriptors() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

struct PlanKey {
    int device;
    int64_t k;
    int64_t m;
    int64_t n;
    uint32_t alignment_a;
    uint32_t alignment_b;
    size_t workspace_limit;

    bool operator==(const PlanKey&) const noexcept = default;
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& key) const noexcept {
        auto mix = [](size_t seed, size_t value) noexcept -> size_t {
            return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
        };
        size_t h = std::hash<int>{}(key.device);
        h = mix(h, std::hash<int64_t>{}(key.k));
        h = mix(h, std::hash<int64_t>{}(key.m));
        h = mix(h, std::hash<int64_t>{}(key.n));
        h = mix(h, std::hash<uint32_t>{}(key.alignment_a));
        h = mix(h, std::hash<uint32_t>{}(key.alignment_b));
        h = mix(h, std::hash<size_t>{}(key.workspace_limit));
        return h;
    }
};

class BoundedPlanCache {
public:
    static constexpr size_t kCapacity = 256;

    LinearGradientDescriptors* find(const PlanKey& key) {
        auto it = index_.find(key);
        if (it == index_.end()) {
            return nullptr;
        }
        entries_.splice(entries_.begin(), entries_, it->second);
        return it->second->second.get();
    }

    LinearGradientDescriptors& insert(const PlanKey& key,
                                      std::unique_ptr<LinearGradientDescriptors> plan) {
        auto existing = index_.find(key);
        if (existing != index_.end()) {
            existing->second->second = std::move(plan);
            entries_.splice(entries_.begin(), entries_, existing->second);
            return *existing->second->second;
        }
        if (entries_.size() >= kCapacity) {
            index_.erase(entries_.back().first);
            entries_.pop_back();
        }
        entries_.emplace_front(key, std::move(plan));
        index_.emplace(key, entries_.begin());
        return *entries_.front().second;
    }

private:
    using Entry = std::pair<PlanKey, std::unique_ptr<LinearGradientDescriptors>>;
    std::list<Entry> entries_;
    std::unordered_map<PlanKey, typename std::list<Entry>::iterator, PlanKeyHash> index_;
};

void validate_compact_matrix_meta(const at::Tensor& a) {
    TORCH_CHECK(a.is_meta() && a.scalar_type() == at::kFloat && a.is_contiguous(),
                "compact matrix must be contiguous Meta FP32");
    TORCH_CHECK(a.dim() >= 2 && a.sym_size(-1) == a.sym_size(-2) && a.sym_numel() > 0,
                "compact matrix must contain nonempty square systems");
    const auto r = a.sym_size(-1);
    TORCH_CHECK(r == 16 || r == 32 || r == 48 || r == 64, "unsupported compact rank");
    TORCH_CHECK(!at::GradMode::is_enabled() || !a.requires_grad(),
                "compact native operations require an explicit autograd owner");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> linear_weight_bias_cuda(
    const at::Tensor& value, const at::Tensor& gradient) {
    for (const auto& t : {value, gradient}) {
        TORCH_CHECK(t.is_cuda() && t.device() == value.device() &&
                    t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.is_contiguous(),
                    "linear gradient inputs must be contiguous 2D CUDA BF16 on one device");
        TORCH_CHECK(!at::GradMode::is_enabled() || !t.requires_grad(),
                    "native linear gradients require an explicit autograd owner");
    }
    TORCH_CHECK(value.size(0) == gradient.size(0) && value.numel() > 0 && gradient.numel() > 0,
                "linear gradient inputs must have matching nonempty token dimensions");
    c10::cuda::CUDAGuard guard(value.device());
    (void)supported_sm();
    const auto k = value.size(0), m = value.size(1), n = gradient.size(1);
    auto options = value.options().dtype(at::kFloat);
    auto weight = at::empty({n, m}, options);
    auto bias = at::empty({n}, options);
    // Thread-local LRU plans let each caller refresh its own bias pointer without eviction cliffs.
    auto alignment = [](const at::Tensor& t) -> uint32_t {
        const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
        return static_cast<uint32_t>(std::min<uintptr_t>(256, address & (~address + 1)));
    };
    const uint32_t alignment_a = alignment(value), alignment_b = alignment(gradient);
    const size_t pool_ws = at::cuda::getCUDABlasLtWorkspaceSize();
    const size_t workspace_limit = std::max<size_t>(pool_ws, size_t{32} << 20);
    thread_local BoundedPlanCache plans;
    const PlanKey key{value.get_device(), k, m, n, alignment_a, alignment_b, workspace_limit};
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    LinearGradientDescriptors* cached = plans.find(key);
    if (cached == nullptr) {
        auto plan = std::make_unique<LinearGradientDescriptors>();
        auto& desc = *plan;

        // Row-major dW = E^T X is column-major dW^T = X^T E.
        // BGRADB reduces E along the same token axis, with FP32 output.
        TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&desc.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans_b = CUBLAS_OP_T;
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADB;
        void* bias_pointer = bias.data_ptr<float>();
        TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.a, CUDA_R_16BF, m, k, m));
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.b, CUDA_R_16BF, n, k, n));
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&desc.d, CUDA_R_32F, m, n, m));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&desc.preference));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignment_a, sizeof(alignment_a)));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignment_b, sizeof(alignment_b)));
        // Require FP32 accumulation, including split-K reduction.
        const uint64_t implementation_mask = ~uint64_t(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F);
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_IMPL_MASK, &implementation_mask, sizeof(implementation_mask)));

        int count = 0;
        TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc.operation, desc.a, desc.b, desc.d, desc.d, desc.preference, 1, &desc.algorithm, &count));
        TORCH_CHECK(count == 1 && desc.algorithm.state == CUBLAS_STATUS_SUCCESS &&
                        desc.algorithm.workspaceSize <= workspace_limit,
                    "cuBLASLt has no FP32-accumulation weight/bias gradient algorithm for this layout");
        cached = &plans.insert(key, std::move(plan));
    }
    auto& desc = *cached;
    auto& algorithm = desc.algorithm;
    void* bias_pointer = bias.data_ptr<float>();
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));
    at::Tensor workspace_owner;
    void* workspace = nullptr;
    if (algorithm.workspaceSize > 0) {
        if (algorithm.workspaceSize <= pool_ws) {
            workspace = at::cuda::getCUDABlasLtWorkspace();
        } else {
            workspace_owner = at::empty(
                {static_cast<int64_t>(algorithm.workspaceSize)},
                value.options().dtype(at::kByte));
            workspace = workspace_owner.data_ptr();
        }
    }
    const float alpha = 1.f, beta = 0.f;
    TORCH_CUDABLAS_CHECK(cublasLtMatmul(handle, desc.operation, &alpha,
        value.data_ptr(), desc.a, gradient.data_ptr(), desc.b, &beta,
        weight.data_ptr(), desc.d, weight.data_ptr(), desc.d, &algorithm.algo,
        workspace, algorithm.workspaceSize, at::cuda::getCurrentCUDAStream()));
    return {weight, bias};
}

std::tuple<at::Tensor, at::Tensor> linear_weight_bias_meta(
    const at::Tensor& value, const at::Tensor& gradient) {
    for (const auto& t : {value, gradient}) {
        TORCH_CHECK(t.is_meta() && t.device() == value.device() &&
                    t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.is_contiguous(),
                    "linear gradient inputs must be contiguous 2D Meta BF16 on one device");
        TORCH_CHECK(!at::GradMode::is_enabled() || !t.requires_grad(),
                    "native linear gradients require an explicit autograd owner");
    }
    TORCH_CHECK(value.sym_size(0) == gradient.sym_size(0) &&
                    value.sym_numel() > 0 && gradient.sym_numel() > 0,
                "linear gradient inputs must have matching nonempty token dimensions");
    auto options = value.options().dtype(at::kFloat);
    auto weight = at::empty_symint({gradient.sym_size(1), value.sym_size(1)}, options);
    auto bias = at::empty_symint({gradient.sym_size(1)}, options);
    return {weight, bias};
}

at::Tensor compact_inverse_meta(const at::Tensor& gram) {
    validate_compact_matrix_meta(gram);
    return at::empty_like(gram, at::MemoryFormat::Contiguous);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compact_lu_meta(const at::Tensor& a) {
    validate_compact_matrix_meta(a);
    auto lu = at::empty_like(a, at::MemoryFormat::Contiguous);
    auto shape = a.sym_sizes().vec();
    shape.pop_back();
    auto pivot = at::empty_symint(shape, a.options().dtype(at::kInt));
    shape.pop_back();
    auto info = at::empty_symint(shape, a.options().dtype(at::kInt));
    return {lu, pivot, info};
}

at::Tensor compact_getrs_meta(
    const at::Tensor& lu, const at::Tensor& pivot, const at::Tensor& rhs, bool /*transpose*/) {
    validate_compact_matrix_meta(lu);
    TORCH_CHECK(rhs.is_meta() && rhs.device() == lu.device() &&
                    rhs.scalar_type() == at::kFloat && rhs.is_contiguous(),
                "compact rhs must be contiguous Meta FP32 on the factor device");
    TORCH_CHECK(!at::GradMode::is_enabled() || !rhs.requires_grad(),
                "compact native operations require an explicit autograd owner");
    TORCH_CHECK(rhs.dim() == lu.dim() && rhs.sym_size(-1) > 0,
                "invalid compact rhs dimensions");
    for (int64_t i = 0; i < lu.dim() - 1; ++i) {
        TORCH_CHECK(rhs.sym_size(i) == lu.sym_size(i), "compact rhs batch/rank mismatch");
    }
    auto shape = lu.sym_sizes().vec();
    shape.pop_back();
    TORCH_CHECK(pivot.is_meta() && pivot.device() == lu.device() &&
                    pivot.scalar_type() == at::kInt && pivot.is_contiguous() &&
                    pivot.sym_sizes().vec() == shape,
                "compact pivots must be contiguous Meta int32 with matching batch/rank");
    return at::empty_like(rhs, at::MemoryFormat::Contiguous);
}
std::tuple<at::Tensor, at::Tensor> linear_weight_bias_autograd(
    const at::Tensor& value, const at::Tensor& gradient) {
    for (const auto& t : {value, gradient}) {
        TORCH_CHECK(!at::GradMode::is_enabled() || !t.requires_grad(),
                    "native linear gradients require an explicit autograd owner");
    }
    c10::impl::ExcludeDispatchKeyGuard no_autograd(c10::autograd_dispatch_keyset);
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("memsolve_equilibrium::linear_weight_bias", "")
                         .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&)>();
    return op.call(value, gradient);
}

at::Tensor compact_inverse_autograd(const at::Tensor& gram) {
    TORCH_CHECK(!at::GradMode::is_enabled() || !gram.requires_grad(),
                "compact native operations require an explicit autograd owner");
    c10::impl::ExcludeDispatchKeyGuard no_autograd(c10::autograd_dispatch_keyset);
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("memsolve_equilibrium::compact_inverse", "")
                         .typed<at::Tensor(const at::Tensor&)>();
    return op.call(gram);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compact_lu_autograd(const at::Tensor& a) {
    TORCH_CHECK(!at::GradMode::is_enabled() || !a.requires_grad(),
                "compact native operations require an explicit autograd owner");
    c10::impl::ExcludeDispatchKeyGuard no_autograd(c10::autograd_dispatch_keyset);
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("memsolve_equilibrium::compact_lu", "")
                         .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(const at::Tensor&)>();
    return op.call(a);
}

at::Tensor compact_getrs_autograd(
    const at::Tensor& lu, const at::Tensor& pivot, const at::Tensor& rhs, bool transpose) {
    for (const auto& t : {lu, rhs}) {
        TORCH_CHECK(!at::GradMode::is_enabled() || !t.requires_grad(),
                    "compact native operations require an explicit autograd owner");
    }
    c10::impl::ExcludeDispatchKeyGuard no_autograd(c10::autograd_dispatch_keyset);
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("memsolve_equilibrium::compact_getrs", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, bool)>();
    return op.call(lu, pivot, rhs, transpose);
}
}  // namespace memsolve_equilibrium

namespace memsolve_equilibrium {
int64_t contract_version() { return 17; }
std::vector<int64_t> supported_architectures() {
    std::vector<int64_t> supported;
    supported.reserve(kRuntimeArchitectures.size());
    for (size_t i = 0; i < kRuntimeArchitectures.size(); ++i) {
        if ((kSupportedSmMask & (1 << static_cast<int>(i))) != 0) {
            supported.push_back(kRuntimeArchitectures[i]);
        }
    }
    return supported;
}
}
TORCH_LIBRARY(memsolve_equilibrium, module) {
    module.def("contract_version() -> int", TORCH_FN(memsolve_equilibrium::contract_version));
    module.def("supported_architectures() -> int[]", TORCH_FN(memsolve_equilibrium::supported_architectures));
    module.def("linear_weight_bias(Tensor value, Tensor gradient) -> (Tensor, Tensor)");
    module.def("compact_inverse(Tensor gram) -> Tensor");
    module.def("compact_lu(Tensor a) -> (Tensor, Tensor, Tensor)");
    module.def("compact_getrs(Tensor lu, Tensor pivots, Tensor rhs, bool transpose=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(memsolve_equilibrium, CUDA, module) {
    module.impl("linear_weight_bias", TORCH_FN(memsolve_equilibrium::linear_weight_bias_cuda));
    module.impl("compact_inverse", TORCH_FN(memsolve_equilibrium::compact_inverse_cuda));
    module.impl("compact_lu", TORCH_FN(memsolve_equilibrium::compact_lu_cuda));
    module.impl("compact_getrs", TORCH_FN(memsolve_equilibrium::compact_getrs_cuda));
}
TORCH_LIBRARY_IMPL(memsolve_equilibrium, CPU, module) {
    module.impl("linear_weight_bias", TORCH_FN(memsolve_equilibrium::linear_weight_bias_cuda));
    module.impl("compact_inverse", TORCH_FN(memsolve_equilibrium::compact_inverse_cuda));
    module.impl("compact_lu", TORCH_FN(memsolve_equilibrium::compact_lu_cuda));
    module.impl("compact_getrs", TORCH_FN(memsolve_equilibrium::compact_getrs_cuda));
}
TORCH_LIBRARY_IMPL(memsolve_equilibrium, Meta, module) {
    module.impl("linear_weight_bias", TORCH_FN(memsolve_equilibrium::linear_weight_bias_meta));
    module.impl("compact_inverse", TORCH_FN(memsolve_equilibrium::compact_inverse_meta));
    module.impl("compact_lu", TORCH_FN(memsolve_equilibrium::compact_lu_meta));
    module.impl("compact_getrs", TORCH_FN(memsolve_equilibrium::compact_getrs_meta));
}
TORCH_LIBRARY_IMPL(memsolve_equilibrium, Autograd, module) {
    module.impl("linear_weight_bias", TORCH_FN(memsolve_equilibrium::linear_weight_bias_autograd));
    module.impl("compact_inverse", TORCH_FN(memsolve_equilibrium::compact_inverse_autograd));
    module.impl("compact_lu", TORCH_FN(memsolve_equilibrium::compact_lu_autograd));
    module.impl("compact_getrs", TORCH_FN(memsolve_equilibrium::compact_getrs_autograd));
}
