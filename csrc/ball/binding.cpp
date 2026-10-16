#include "common.cuh"
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <algorithm>
#include <map>
#include <memory>
#include <tuple>

namespace memsolve_equilibrium {
namespace {
// Descriptors own host state only; the allocator owns the graph-safe workspace.
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
}

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
    // Cache host plans, never tensor storage or device workspaces. Per-thread
    // descriptors avoid races when bias pointers are refreshed on other streams.
    auto alignment = [](const at::Tensor& t) -> uint32_t {
        const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
        return static_cast<uint32_t>(std::min<uintptr_t>(256, address & (~address + 1)));
    };
    const uint32_t alignment_a = alignment(value), alignment_b = alignment(gradient);
    using Key = std::tuple<int,int64_t,int64_t,int64_t,uint32_t,uint32_t>;
    thread_local std::map<Key,std::unique_ptr<LinearGradientDescriptors>> plans;
    const Key key(value.get_device(),k,m,n,alignment_a,alignment_b);
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    auto found = plans.find(key);
    if (found == plans.end()) {
        // Bound host metadata for variable-length workloads. Eviction does not
        // invalidate captured graphs: their launch arguments are already copied.
        if (plans.size() >= 256) plans.clear();
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
        const size_t workspace_limit = 32 * 1024 * 1024;
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignment_a, sizeof(alignment_a)));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignment_b, sizeof(alignment_b)));
        // Do not permit a reduced-precision accumulator even if process-wide
        // PyTorch settings allow one. D and the split-K reduction are FP32.
        const uint64_t implementation_mask = ~uint64_t(CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F);
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(desc.preference, CUBLASLT_MATMUL_PREF_IMPL_MASK, &implementation_mask, sizeof(implementation_mask)));

        int count = 0;
        TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc.operation, desc.a, desc.b, desc.d, desc.d, desc.preference, 1, &desc.algorithm, &count));
        TORCH_CHECK(count == 1 && desc.algorithm.state == CUBLAS_STATUS_SUCCESS,
                    "cuBLASLt has no FP32-accumulation weight/bias gradient algorithm for this layout");
        found = plans.emplace(key,std::move(plan)).first;
    }
    auto& desc = *found->second;
    auto& algorithm = desc.algorithm;
    void* bias_pointer = bias.data_ptr<float>();
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(desc.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));
    auto workspace = at::empty({static_cast<int64_t>(algorithm.workspaceSize)}, value.options().dtype(at::kByte));
    const float alpha = 1.f, beta = 0.f;
    TORCH_CUDABLAS_CHECK(cublasLtMatmul(handle, desc.operation, &alpha,
        value.data_ptr(), desc.a, gradient.data_ptr(), desc.b, &beta,
        weight.data_ptr(), desc.d, weight.data_ptr(), desc.d, &algorithm.algo,
        workspace.data_ptr(), algorithm.workspaceSize, at::cuda::getCurrentCUDAStream()));
    return {weight, bias};
}
}  // namespace memsolve_equilibrium

namespace memsolve_equilibrium {
int64_t contract_version() { return 17; }
}
TORCH_LIBRARY(memsolve_equilibrium, module) {
    module.def("contract_version() -> int", TORCH_FN(memsolve_equilibrium::contract_version));
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
TORCH_LIBRARY_IMPL(memsolve_equilibrium, Autograd, module) {
    module.impl("linear_weight_bias", TORCH_FN(memsolve_equilibrium::linear_weight_bias_cuda));
    module.impl("compact_inverse", TORCH_FN(memsolve_equilibrium::compact_inverse_cuda));
    module.impl("compact_lu", TORCH_FN(memsolve_equilibrium::compact_lu_cuda));
    module.impl("compact_getrs", TORCH_FN(memsolve_equilibrium::compact_getrs_cuda));
}
