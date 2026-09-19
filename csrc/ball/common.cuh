#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Optional.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace lsso_equilibrium {

#ifndef LSSO_CUDA_TARGET_SM
#error "LSSO_CUDA_TARGET_SM must be defined by the strict CUDA build."
#endif

constexpr int kRhsTile = 32;
constexpr int kThreads = 256;
constexpr int kCompiledSm = LSSO_CUDA_TARGET_SM;
constexpr float kComplementInteriorScale = 1.0f - 0x1.0p-23f;

__device__ inline float bounded_complement(float raw) {
    const float exponent = raw >= 0.0f ? expf(-2.0f * raw) : expf(2.0f * raw);
    const float numerator = raw >= 0.0f ? 1.0f - exponent : exponent - 1.0f;
    return kComplementInteriorScale * numerator / (1.0f + exponent);
}

__device__ inline float bounded_complement_derivative(float raw) {
    const float exponent = raw >= 0.0f ? expf(-2.0f * raw) : expf(2.0f * raw);
    const float denominator = 1.0f + exponent;
    return kComplementInteriorScale * 4.0f * exponent / (denominator * denominator);
}

struct SharedCursor {
    explicit __device__ SharedCursor(unsigned char* address) : address(address) {}

    __device__ void* take_bytes(size_t bytes, size_t alignment) {
        auto value = reinterpret_cast<uintptr_t>(address);
        value = (value + alignment - 1) & ~(alignment - 1);
        address = reinterpret_cast<unsigned char*>(value + bytes);
        return reinterpret_cast<void*>(value);
    }

    template <typename value_t>
    __device__ value_t* take(size_t count) {
        return reinterpret_cast<value_t*>(
            take_bytes(sizeof(value_t) * count, alignof(value_t)));
    }

    unsigned char* address;
};

constexpr size_t align_shared_offset(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

struct FastPathShape {
    int64_t batch;
    int64_t length;
    int64_t heads;
    int64_t dim;
    int64_t head_dim;
    int64_t rank;
    int core_mode; // 0 Dynamic, 1 Static, 2 Zero
};

// One contiguous FP32 activation tape per [batch, head] system.  The tensor is
// intentionally private to the native autograd boundary; its layout makes the
// long token sections contiguous for the tiled VJPs.
struct TrainingTapeLayout {
    int64_t b_offset;
    int64_t p_offset;
    int64_t l_offset;
    int64_t z_offset;
    int64_t coordinates_offset;
    int64_t lu_offset;
    int64_t u_offset;
    int64_t scale_offset;
    int64_t stride;
};

inline TrainingTapeLayout training_tape_layout(FastPathShape shape) {
    const int64_t relation_elements = shape.length * shape.rank;
    const int64_t compact_elements = shape.rank * shape.head_dim;
    const int64_t matrix_elements = shape.rank * shape.rank;
    const int64_t b_offset = 0;
    // Materialization overwrites B with P; backward reconstructs B.
    const int64_t p_offset = b_offset;
    const int64_t l_offset = p_offset + relation_elements;
    const int64_t z_offset = l_offset + matrix_elements;
    const int64_t coordinates_offset = z_offset + compact_elements;
    const int64_t lu_offset = coordinates_offset + matrix_elements;
    const int64_t u_offset = lu_offset + matrix_elements;
    const int64_t scale_offset = u_offset + compact_elements;
    return {
        b_offset,
        p_offset,
        l_offset,
        z_offset,
        coordinates_offset,
        lu_offset,
        u_offset,
        scale_offset,
        scale_offset + 1,
    };
}

// Share normalized relation loads between forward and backward reconstruction.
template <typename scalar_t, int rank>
__device__ __forceinline__ float2 generic_relation_pair(
    const scalar_t* projected,
    int64_t batch,
    int64_t head,
    int64_t token,
    int pair,
    int64_t length,
    int64_t projected_width,
    float inverse_length_sqrt) {
    const int even_rank = 2 * pair;
    const int64_t relation_base =
        (batch * length + token) * projected_width + head * rank;
    const float even = static_cast<float>(projected[relation_base + even_rank]);
    const float odd = static_cast<float>(projected[relation_base + even_rank + 1]);
    return make_float2(even * inverse_length_sqrt, odd * inverse_length_sqrt);
}

struct ForwardWorkspaceLayout {
    int64_t b_offset;
    int64_t l_offset;
    int64_t z_offset;
    int64_t coordinates_offset;
    int64_t lu_offset;
    int64_t u_offset;
    int64_t scale_offset;
    int64_t stride;
};

inline ForwardWorkspaceLayout forward_workspace_layout(
    TrainingTapeLayout tape_layout) {
    return {
        tape_layout.b_offset,
        tape_layout.l_offset,
        tape_layout.z_offset,
        tape_layout.coordinates_offset,
        tape_layout.lu_offset,
        tape_layout.u_offset,
        tape_layout.scale_offset,
        tape_layout.stride,
    };
}

inline ForwardWorkspaceLayout inference_workspace_layout(FastPathShape shape) {
    const int64_t relation_elements = shape.length * shape.rank;
    const int64_t compact_elements = shape.rank * shape.head_dim;
    const int64_t matrix_elements = shape.rank * shape.rank;
    const int64_t b_offset = 0;
    const int64_t l_offset = b_offset + relation_elements;
    const int64_t z_offset = l_offset + matrix_elements;
    // Coordinates are only retained on the training tape for the core VJP.
    // The fused inference core builds its generator directly from the
    // Tensor-Core accumulator and therefore needs no global coordinate slot.
    const int64_t coordinates_offset = 0;
    const int64_t lu_offset = z_offset + compact_elements;
    const int64_t u_offset = lu_offset + matrix_elements;
    const int64_t scale_offset = u_offset + compact_elements;
    return {
        b_offset,
        l_offset,
        z_offset,
        coordinates_offset,
        lu_offset,
        u_offset,
        scale_offset,
        scale_offset + 1,
    };
}

inline void check_same_cuda_device(const at::Tensor& value, const at::Tensor& expected,
                                   const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.device() == expected.device(), name,
                " must share projected.device() (got ", value.device(), " vs ",
                expected.device(), ")");
}

inline FastPathShape validate_fast_inputs(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts) {
    TORCH_CHECK(projected.is_cuda(), "projected must be a CUDA tensor");
    TORCH_CHECK(projected.is_contiguous(), "projected must be contiguous [B, N, H*R + D]");
    TORCH_CHECK(projected.dim() == 3,
                "projected must have shape [B, N, H*R + D], got ", projected.sizes());
    TORCH_CHECK(
        projected.scalar_type() == at::kBFloat16,
        "projected must use bfloat16 under the mixed precision CUDA contract, got ",
        projected.scalar_type());

    TORCH_CHECK(core_base_raw.is_contiguous(), "core_base_raw must be contiguous");
    TORCH_CHECK(core_drive_weight.is_contiguous(), "core_drive_weight must be contiguous");
    TORCH_CHECK(eta_raw.is_contiguous(), "eta_raw must be contiguous");
    check_same_cuda_device(core_base_raw, projected, "core_base_raw");
    check_same_cuda_device(core_drive_weight, projected, "core_drive_weight");
    check_same_cuda_device(eta_raw, projected, "eta_raw");
    TORCH_CHECK(core_base_raw.scalar_type() == at::kFloat,
                "core_base_raw must use float32");
    TORCH_CHECK(core_drive_weight.scalar_type() == at::kFloat,
                "core_drive_weight must use float32");
    TORCH_CHECK(eta_raw.scalar_type() == at::kFloat, "eta_raw must use float32");
    TORCH_CHECK(core_base_raw.dim() == 3 && (core_base_raw.size(2) == 0 || core_base_raw.size(1) == core_base_raw.size(2)),
                "core_base_raw must have shape [H, R, R] or [H, R, 0] for Zero, got ", core_base_raw.sizes());

    const auto batch = projected.size(0);
    const auto length = projected.size(1);
    const auto width = projected.size(2);
    const auto heads = core_base_raw.size(0);
    const auto rank = core_base_raw.size(1);
    TORCH_CHECK(batch > 0 && length > 0 && heads > 0,
                "B, N, and H must be positive");
    TORCH_CHECK(rank == 16 || rank == 32 || rank == 48 || rank == 64,
                "the CUDA fast path supports rank in {16, 32, 48, 64}, got ", rank);
    TORCH_CHECK(width > heads * rank,
                "projected width must exceed H*R for a nonempty content channel");
    const auto dim = width - heads * rank;
    TORCH_CHECK(dim % heads == 0,
                "projected content width D must be divisible by H");
    const auto head_dim = dim / heads;
    const int core_mode = core_base_raw.numel() == 0 ? 2 : (core_drive_weight.numel() == 0 ? 1 : 0);
    TORCH_CHECK(core_base_raw.numel() != 0 || core_drive_weight.numel() == 0, "Zero cannot have a dynamic drive");
    TORCH_CHECK(core_drive_weight.sizes() == at::IntArrayRef({heads, head_dim, core_mode == 0 ? rank : 0}),
                "core_drive_weight must have shape [H, D/H, R] for Dynamic or [H, D/H, 0] otherwise, got ",
                core_drive_weight.sizes());
    TORCH_CHECK(eta_raw.sizes() == at::IntArrayRef({heads}),
                "eta_raw must have shape [H], got ", eta_raw.sizes());

    if (valid_counts.has_value()) {
        const auto& counts = *valid_counts;
        check_same_cuda_device(counts, projected, "valid_counts");
        TORCH_CHECK(counts.is_contiguous(), "valid_counts must be contiguous");
        TORCH_CHECK(counts.scalar_type() == at::kFloat,
                    "valid_counts must use float32");
        TORCH_CHECK(counts.dim() == 1 && counts.size(0) == batch,
                    "valid_counts must have shape [B], got ", counts.sizes());
    }

    return {batch, length, heads, dim, head_dim, rank, core_mode};
}

inline int supported_sm() {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const int capability = properties.major * 100 + properties.minor * 10;
    const int normalized_capability = capability == 1210 ? 1200 : capability;
    TORCH_CHECK(
        normalized_capability == kCompiledSm,
        "the loaded LSSO CUDA extension targets SM", kCompiledSm,
        " but received SM", properties.major, ".", properties.minor,
        "; load the matching strict artifact");

    switch (capability) {
        case 800:
        case 860:
        case 870:
        case 890:
        case 900:
        case 1000:
        case 1200:
            return capability;
        case 1210:
            return 1200;
        default:
            TORCH_CHECK(false,
                        "the LSSO CUDA fast path supports known Ampere-and-newer "
                        "architectures (SM80, SM86, SM87, SM89, SM90, SM100, SM120); "
                        "got SM", properties.major, ".", properties.minor);
    }
}

at::Tensor forward_inference_cuda(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts);

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_train_cuda(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const at::Tensor& tape,
    const at::Tensor& pivots,
    const c10::optional<at::Tensor>& valid_counts);

}  // namespace lsso_equilibrium
