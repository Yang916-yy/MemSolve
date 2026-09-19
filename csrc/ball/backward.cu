#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cublasdx.hpp>
#include <cusolverdx.hpp>
#include <cusolverdx/detail/shared_memory.hpp>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_set>

namespace lsso_equilibrium {
namespace {

constexpr float kSoftplusOneOffset = 0.5413248546129181f;

template <typename scalar_t>
__device__ __forceinline__ float load_scalar(const scalar_t* values, int64_t index) {
    return static_cast<float>(values[index]);
}

template <typename scalar_t>
__device__ __forceinline__ void store_scalar(
    scalar_t* values,
    int64_t index,
    float value) {
    values[index] = static_cast<scalar_t>(value);
}

__device__ __forceinline__ float softplus_one(float raw) {
    const float shifted = raw + kSoftplusOneOffset;
    return shifted > 20.0f ? shifted : log1pf(expf(shifted));
}

__device__ __forceinline__ float softplus_one_derivative(float raw) {
    const float shifted = raw + kSoftplusOneOffset;
    if (shifted >= 0.0f) {
        return 1.0f / (1.0f + expf(-shifted));
    }
    const float exponential = expf(shifted);
    return exponential / (1.0f + exponential);
}

__device__ __forceinline__ void reduce_sum(float* values) {
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            values[threadIdx.x] += values[threadIdx.x + stride];
        }
        __syncthreads();
    }
}

constexpr int kGenericVjpTokenTile = 64;

template <int rank, int token_rows = kRhsTile>
struct GenericBackwardMathDx {
    using Token = decltype(
        cublasdx::Size<token_rows, rank, kRhsTile>() +
        cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::row_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());

    using Compact = decltype(
        cublasdx::Size<rank, rank, kRhsTile>() +
        cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::row_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());

    using Content = decltype(
        cublasdx::Size<kRhsTile, kRhsTile, rank>() +
        cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::row_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());

    using CoreContent = decltype(
        cublasdx::Size<token_rows, rank, rank>() +
        cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::row_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());

    using FrameVjpLeftTrsm = decltype(
        cusolverdx::Size<rank, rank>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::left>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::transposed>() +
        cusolverdx::Diag<cusolverdx::diag::non_unit>() +
        cusolverdx::Arrangement<
            cusolverdx::arrangement::row_major,
            cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kThreads>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());

    using FrameVjpRightTrsm = decltype(
        cusolverdx::Size<rank, rank>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::right>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::non_transposed>() +
        cusolverdx::Diag<cusolverdx::diag::non_unit>() +
        cusolverdx::Arrangement<
            cusolverdx::arrangement::row_major,
            cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kThreads>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());
};

template <int rank>
struct GenericEquilibriumGetrsMathDx {
    using Transposed = decltype(
        cusolverdx::Size<rank, rank, kRhsTile>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::getrs_partial_pivot>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::transposed>() +
        cusolverdx::Arrangement<
            cusolverdx::arrangement::row_major,
            cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kRhsTile>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());
};

template <int rank>
constexpr size_t generic_frame_compact_shared_bytes() {
    using Compact = typename GenericBackwardMathDx<rank>::Compact;
    constexpr size_t a_elements = cublasdx::cosize(Compact::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Compact::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Compact::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * b_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    return offset;
}

template <int rank>
constexpr size_t generic_content_vjp_shared_bytes() {
    using Content = typename GenericBackwardMathDx<rank>::Content;
    constexpr size_t a_elements = cublasdx::cosize(Content::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Content::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Content::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * b_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    return offset;
}

template <int rank>
constexpr size_t generic_core_content_vjp_shared_bytes() {
    using CoreContent = typename GenericBackwardMathDx<rank>::CoreContent;
    constexpr size_t a_elements = cublasdx::cosize(CoreContent::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(CoreContent::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(CoreContent::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * b_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * rank * rank;
    return offset;
}

template <int rank>
constexpr size_t generic_frame_vjp_shared_bytes() {
    using LeftTrsm = typename GenericBackwardMathDx<rank>::FrameVjpLeftTrsm;
    using RightTrsm = typename GenericBackwardMathDx<rank>::FrameVjpRightTrsm;
    static_assert(LeftTrsm::lda == RightTrsm::lda);
    static_assert(LeftTrsm::ldb == RightTrsm::ldb);
    static_assert(LeftTrsm::shared_memory_size == RightTrsm::shared_memory_size);
    return LeftTrsm::shared_memory_size;
}

// High-rank relation VJPs are register-bound in their per-token form.  The
// panel form is algebraically identical: D_B = D_P L^{-1}, then D_B += 2 B C.
template <int rank>
struct PanelRelationVjpMathDx {
    // Wider local panels amortize GEMM/TRSM setup. At rank 64 the wider
    // panels lose throughput in measurements, so keep 32-token panels.
    static constexpr int kTokenPanel = rank == 64 ? kRhsTile : 64;
    using Trsm = decltype(
        cusolverdx::Size<kTokenPanel, rank>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::right>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::non_transposed>() +
        cusolverdx::Diag<cusolverdx::diag::non_unit>() +
        cusolverdx::Arrangement<
            cusolverdx::arrangement::row_major,
            cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kThreads>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());

    using Token = typename GenericBackwardMathDx<rank, kTokenPanel>::Token;
    using Correction = typename GenericBackwardMathDx<rank, kTokenPanel>::CoreContent;
};

template <int rank>
constexpr size_t panel_relation_vjp_shared_bytes() {
    using Trsm = typename PanelRelationVjpMathDx<rank>::Trsm;
    using Correction = typename PanelRelationVjpMathDx<rank>::Correction;
    constexpr size_t a_elements = cublasdx::cosize(Correction::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Correction::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Correction::get_layout_smem_c());
    size_t offset = Trsm::shared_memory_size;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * b_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    return offset;
}

// The frame product and relation VJP run back-to-back for each 32-token
// subtile. Their storage therefore has disjoint lifetimes.
template <int rank>
constexpr size_t fused_frame_relation_vjp_shared_bytes() {
    return panel_relation_vjp_shared_bytes<rank>();
}

void validate_grad_output(
    const at::Tensor& grad_output,
    const at::Tensor& projected,
    FastPathShape shape) {
    check_same_cuda_device(grad_output, projected, "grad_output");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.scalar_type() == at::kBFloat16,
                "grad_output must use bfloat16");
    TORCH_CHECK(
        grad_output.sizes() == at::IntArrayRef({shape.batch, shape.length, shape.dim}),
        "grad_output must have shape [B, N, D], got ", grad_output.sizes());
}

void validate_training_tape(
    const at::Tensor& tape,
    const at::Tensor& pivots,
    const at::Tensor& projected,
    FastPathShape shape) {
    const auto layout = training_tape_layout(shape);
    check_same_cuda_device(tape, projected, "tape");
    check_same_cuda_device(pivots, projected, "pivots");
    TORCH_CHECK(tape.is_contiguous(), "tape must be contiguous");
    TORCH_CHECK(tape.scalar_type() == at::kFloat, "tape must use float32");
    TORCH_CHECK(
        tape.sizes() == at::IntArrayRef({shape.batch * shape.heads, layout.stride}),
        "tape has an incompatible shape ", tape.sizes());
    TORCH_CHECK(pivots.is_contiguous(), "pivots must be contiguous");
    TORCH_CHECK(pivots.scalar_type() == at::kInt, "pivots must use int32");
    TORCH_CHECK(
        pivots.sizes() == at::IntArrayRef({shape.batch * shape.heads, shape.rank}),
        "pivots has an incompatible shape ", pivots.sizes());
}

// Both statistics are the same FP32 contraction P_tile^T V_tile.  Reuse
// cuBLASDx's shared-memory GEMM (as in the forward frame Gram), loading each
// frame tile once and each content/gradient panel once.  No BF16 rounding is
// introduced: the reconstructed content statistic protects the eta VJP.
template <typename scalar_t, int rank, bool gradient_statistic>
__global__ __launch_bounds__(kThreads) void generic_statistic_partials_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ projected,
    const float* __restrict__ tape,
    float* __restrict__ partial,
    float* __restrict__ partial_eta,
    TrainingTapeLayout tape_layout,
    int64_t batch_count,
    int64_t length,
    int64_t heads,
    int64_t dim,
    int64_t head_dim,
    int64_t tile_count) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y);
    const int64_t token_start = tile_index * kGenericVjpTokenTile;
    if (system_index >= batch_count * heads || token_start >= length) {
        return;
    }
    const int64_t batch = system_index / heads;
    const int64_t head = system_index - batch * heads;
    const int64_t projected_width = heads * rank + dim;
    const float* frame =
        tape + system_index * tape_layout.stride + tape_layout.p_offset;
    using Gemm = decltype(
        cublasdx::Size<rank, kRhsTile, kGenericVjpTokenTile>() +
        cublasdx::Precision<float, float, float>() +
        cublasdx::Alignment<16, 16, 16>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<
            cublasdx::col_major,
            cublasdx::row_major,
            cublasdx::row_major>() +
        cublasdx::Block() +
        cublasdx::BlockDim<kThreads>() +
        cublasdx::StaticBlockDim() +
        cublasdx::SM<kCompiledSm>());
    __shared__ __align__(16) float a_storage[cublasdx::cosize(Gemm::get_layout_smem_a())];
    __shared__ __align__(16) float b_storage[cublasdx::cosize(Gemm::get_layout_smem_b())];
    __shared__ __align__(16) float c_storage[cublasdx::cosize(Gemm::get_layout_smem_c())];
    __shared__ float reduction[kThreads];
    auto a = cublasdx::make_tensor(a_storage, Gemm::get_layout_smem_a());
    auto b = cublasdx::make_tensor(b_storage, Gemm::get_layout_smem_b());
    auto c = cublasdx::make_tensor(c_storage, Gemm::get_layout_smem_c());
    for (int i = threadIdx.x; i < rank * kGenericVjpTokenTile; i += blockDim.x) {
        const int t = i / rank;
        const int r = i % rank;
        a(r, t) = token_start + t < length
            ? frame[(token_start + t) * rank + r]
            : 0.0f;
    }
    float eta_local = 0.0f;
    for (int64_t start = 0; start < head_dim; start += kRhsTile) {
        for (int i = threadIdx.x; i < kGenericVjpTokenTile * kRhsTile; i += blockDim.x) {
            const int t = i / kRhsTile;
            const int f = i % kRhsTile;
            const int64_t token = token_start + t;
            const int64_t feature = start + f;
            float value = 0.0f;
            if (token < length && feature < head_dim) {
                const int64_t content_index =
                    (batch * length + token) * projected_width +
                    heads * rank + head * head_dim + feature;
                if constexpr (gradient_statistic) {
                    value = load_scalar(grad_output,
                        (batch * length + token) * dim + head * head_dim + feature);
                    eta_local += value * load_scalar(projected, content_index);
                } else {
                    value = load_scalar(projected, content_index);
                }
            }
            b(t, f) = value;
        }
        for (int i = threadIdx.x; i < cublasdx::cosize(Gemm::get_layout_smem_c());
             i += blockDim.x) {
            c_storage[i] = 0.0f;
        }
        __syncthreads();
        Gemm().execute(1.0f, a, b, 0.0f, c);
        __syncthreads();
        for (int i = threadIdx.x; i < rank * kRhsTile; i += blockDim.x) {
            const int r = i / kRhsTile;
            const int f = i % kRhsTile;
            if (start + f < head_dim) {
                partial[((system_index * tile_count + tile_index) * rank + r) *
                        head_dim + start + f] = c(r, f);
            }
        }
        __syncthreads();
    }
    if constexpr (gradient_statistic) {
        reduction[threadIdx.x] = eta_local;
        __syncthreads();
        reduce_sum(reduction);
        if (threadIdx.x == 0) {
            partial_eta[system_index * tile_count + tile_index] = reduction[0];
        }
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_reduce_partials_kernel(
    const float* __restrict__ partials,
    float* __restrict__ output,
    int64_t system_count,
    int64_t tile_count,
    int64_t head_dim) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }
    const int64_t output_slice = static_cast<int64_t>(blockIdx.y);
    const int64_t output_stride = static_cast<int64_t>(gridDim.y) * blockDim.x;
    for (int64_t linear = output_slice * blockDim.x + threadIdx.x;
         linear < rank * head_dim;
         linear += output_stride) {
        float value = 0.0f;
        for (int64_t tile = 0; tile < tile_count; ++tile) {
            value += partials[
                (system_index * tile_count + tile) * rank * head_dim + linear];
        }
        output[system_index * rank * head_dim + linear] = value;
    }
}

struct GenericPartialReductionLaunch {
    int64_t output_slices;
    int threads;
};

// Output slices are disjoint: each element retains the original ascending
// FP32 token-tile sum while long, low-system-count workloads gain CTAs.
GenericPartialReductionLaunch generic_partial_reduction_launch(
    int64_t system_count,
    int64_t token_tiles,
    int64_t output_elements) {
    constexpr int kShardedThreads = 128;
    constexpr int64_t kTargetReductionCtas = 128;
    if (token_tiles < 16 || system_count >= kTargetReductionCtas ||
        output_elements <= kShardedThreads) {
        return {1, kThreads};
    }
    const int64_t maximum_slices =
        (output_elements + kShardedThreads - 1) / kShardedThreads;
    const int64_t requested_slices =
        (kTargetReductionCtas + system_count - 1) / system_count;
    return {
        std::min(maximum_slices, requested_slices),
        kShardedThreads,
    };
}

template <int rank>
__global__ __launch_bounds__(kRhsTile) void generic_equilibrium_vjp_kernel(
    const float* __restrict__ d_t,
    const float* __restrict__ tape,
    const int* __restrict__ pivots,
    const float* __restrict__ eta_raw,
    float* __restrict__ equilibrium_adjoint,
    float* __restrict__ state_adjoint,
    TrainingTapeLayout tape_layout,
    int64_t system_count,
    int64_t heads,
    int64_t head_dim,
    int core_mode) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t rhs_tile = static_cast<int64_t>(blockIdx.y);
    if (system_index >= system_count) {
        return;
    }
    const int64_t head = system_index - (system_index / heads) * heads;
    const int64_t rhs_start = rhs_tile * kRhsTile;
    const float* system_tape = tape + system_index * tape_layout.stride;
    const int64_t core_index = core_mode == 1 ? head : system_index;
    const float* lu = tape + core_index * tape_layout.stride + tape_layout.lu_offset;
    const int* system_pivots = pivots + core_index * rank;
    const float* system_d_t = d_t + system_index * rank * head_dim;
    float* system_equilibrium_adjoint = equilibrium_adjoint + system_index * rank * head_dim;
    float* system_state_adjoint = state_adjoint + system_index * rank * head_dim;
    const float eta = bounded_complement(eta_raw[head]);

    if (core_mode == 2) {
        for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear % kRhsTile;
            if (rhs_start + column < head_dim) {
                const int offset = row * head_dim + rhs_start + column;
                system_equilibrium_adjoint[offset] = system_d_t[offset];
                system_state_adjoint[offset] = -eta * system_d_t[offset];
            }
        }
        return;
    }

    using Getrs = typename GenericEquilibriumGetrsMathDx<rank>::Transposed;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto [lu_tile, rhs, pivot_tile] = cusolverdx::shared_memory::slice<float, float, int>(
        shared_raw,
        16u, rank * Getrs::lda,
        16u, rank * Getrs::ldb,
        16u, rank);
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        lu_tile[linear] = lu[linear];
    }
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        rhs[linear] = rhs_start + column < head_dim
            ? 2.0f * system_d_t[row * head_dim + rhs_start + column]
            : 0.0f;
    }
    for (int linear = threadIdx.x; linear < rank; linear += blockDim.x) {
        pivot_tile[linear] = system_pivots[linear];
    }
    __syncthreads();
    Getrs().execute(lu_tile, Getrs::lda, pivot_tile, rhs, Getrs::ldb);
    __syncthreads();
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        if (rhs_start + column < head_dim) {
            const int64_t offset = row * head_dim + rhs_start + column;
            system_equilibrium_adjoint[offset] = rhs[linear];
            system_state_adjoint[offset] =
                -(1.0f + eta) * system_d_t[offset] + rhs[linear];
        }
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_d_k_kernel(
    const float* __restrict__ equilibrium_adjoint,
    const float* __restrict__ tape,
    float* __restrict__ d_k,
    TrainingTapeLayout tape_layout,
    int64_t system_count,
    int64_t head_dim) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }
    const float* system_tape = tape + system_index * tape_layout.stride;
    const float* equilibrium = system_tape + tape_layout.u_offset;
    const float* adjoint = equilibrium_adjoint + system_index * rank * head_dim;
    float* output = d_k + system_index * rank * rank;

    constexpr int output_count = (rank * rank + kThreads - 1) / kThreads;
    __shared__ float tiles[2 * rank * kRhsTile];
    float* adjoint_tile = tiles;
    float* equilibrium_tile = tiles + rank * kRhsTile;
    float values[output_count] = {};
    for (int64_t feature_start = 0; feature_start < head_dim;
         feature_start += kRhsTile) {
        for (int linear = threadIdx.x; linear < rank * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear - row * kRhsTile;
            const int64_t feature = feature_start + column;
            adjoint_tile[linear] = feature < head_dim
                ? adjoint[row * head_dim + feature]
                : 0.0f;
        }
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int row = linear / rank;
            const int column = linear - row * rank;
            const int64_t feature = feature_start + row;
            equilibrium_tile[column * kRhsTile + row] = feature < head_dim
                ? equilibrium[column * head_dim + feature]
                : 0.0f;
        }
        __syncthreads();
        for (int index = 0; index < output_count; ++index) {
            const int linear = threadIdx.x + index * blockDim.x;
            if (linear < rank * rank) {
                const int row = linear / rank;
                const int column = linear - row * rank;
                float value = values[index];
#pragma unroll
                for (int feature = 0; feature < kRhsTile; ++feature) {
                    value -= adjoint_tile[row * kRhsTile + feature] *
                        equilibrium_tile[column * kRhsTile + feature];
                }
                values[index] = value;
            }
        }
        __syncthreads();
    }
    for (int index = 0; index < output_count; ++index) {
        const int linear = threadIdx.x + index * blockDim.x;
        if (linear < rank * rank) {
            output[linear] = values[index];
        }
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_core_relation_vjp_kernel(
    const float* __restrict__ tape,
    float* __restrict__ d_k,
    float* __restrict__ grad_core_base_raw,
    TrainingTapeLayout tape_layout,
    int64_t system_count,
    int64_t heads,
    int core_mode) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }
    const int64_t head = system_index - (system_index / heads) * heads;
    const float* system_tape = tape + (core_mode == 1 ? head : system_index) * tape_layout.stride;
    const float* saved_coordinates = system_tape + tape_layout.coordinates_offset;
    float* system_d_k = d_k + system_index * rank * rank;

    __shared__ float coordinates[rank * rank];
    __shared__ float raw_adjoint[rank * rank];
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        coordinates[linear] = saved_coordinates[linear];
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        if (row < column) {
            raw_adjoint[linear] =
                system_d_k[linear] - system_d_k[column * rank + row];
            continue;
        }

        float d_factor = 0.0f;
        for (int inner = column; inner < rank; ++inner) {
            const float factor = inner == column
                ? softplus_one(coordinates[column * rank + column])
                : coordinates[inner * rank + column];
            d_factor +=
                (system_d_k[row * rank + inner] + system_d_k[inner * rank + row]) *
                factor;
        }
        raw_adjoint[linear] = row > column
            ? d_factor
            : d_factor * softplus_one_derivative(coordinates[linear]);
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        atomicAdd(
            grad_core_base_raw + head * rank * rank + linear,
            raw_adjoint[linear]);
    }
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        system_d_k[linear] = raw_adjoint[linear];
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_core_content_vjp_kernel(
    const float* __restrict__ tape,
    const float* __restrict__ core_drive_weight,
    const float* __restrict__ raw_adjoint,
    const float* __restrict__ valid_counts,
    float* __restrict__ state_adjoint,
    float* __restrict__ grad_core_drive_weight,
    TrainingTapeLayout tape_layout,
    int64_t system_count,
    int64_t heads,
    int64_t head_dim,
    float inverse_length_sqrt) {
    using CoreContent = typename GenericBackwardMathDx<rank>::CoreContent;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }
    const int64_t head = system_index - (system_index / heads) * heads;
    const int64_t batch = system_index / heads;
    if (valid_counts != nullptr) {
        inverse_length_sqrt = rsqrtf(valid_counts[batch]);
    }
    const float* system_tape = tape + system_index * tape_layout.stride;
    const float* compact_state = system_tape + tape_layout.z_offset;
    const float* system_raw_adjoint = raw_adjoint + system_index * rank * rank;
    float* system_state_adjoint = state_adjoint + system_index * rank * head_dim;

    constexpr size_t a_elements = cublasdx::cosize(CoreContent::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(CoreContent::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(CoreContent::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    __nv_bfloat16* a_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * a_elements, 16));
    __nv_bfloat16* b_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * b_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    __nv_bfloat16* scaled_adjoint = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * rank * rank, 16));
    auto core_a = cublasdx::make_tensor(a_tile, CoreContent::get_layout_smem_a());
    auto core_b = cublasdx::make_tensor(b_tile, CoreContent::get_layout_smem_b());
    auto core_c = cublasdx::make_tensor(gemm_output, CoreContent::get_layout_smem_c());

    // The upstream scale belongs before FP16 conversion to match the canonical
    // BF16 VJP rounding boundary used by tensor_core_matmul.
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        scaled_adjoint[linear] = __float2bfloat16_rn(
            system_raw_adjoint[linear] * inverse_length_sqrt);
    }
    __syncthreads();
    for (int64_t feature_start = 0; feature_start < head_dim;
         feature_start += kRhsTile) {
        for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
            gemm_output[linear] = 0.0f;
        }
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int feature = linear / rank;
            const int row = linear - feature * rank;
            const int64_t global_feature = feature_start + feature;
            core_a(feature, row) = __float2bfloat16_rn(
                global_feature < head_dim
                    ? compact_state[row * head_dim + global_feature]
                    : 0.0f);
        }
        for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
            const int row = linear / rank;
            const int column = linear - row * rank;
            core_b(row, column) = scaled_adjoint[row * rank + column];
        }
        __syncthreads();
        CoreContent().execute(1.0f, core_a, core_b, 0.0f, core_c);
        __syncthreads();
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int feature = linear / rank;
            const int column = linear - feature * rank;
            const int64_t global_feature = feature_start + feature;
            if (global_feature < head_dim) {
                atomicAdd(
                    grad_core_drive_weight +
                        (head * head_dim + global_feature) * rank + column,
                    core_c(feature, column));
            }
        }
        __syncthreads();

        for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
            gemm_output[linear] = 0.0f;
        }
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int feature = linear / rank;
            const int row = linear - feature * rank;
            const int64_t global_feature = feature_start + feature;
            core_a(feature, row) = __float2bfloat16_rn(
                global_feature < head_dim
                    ? core_drive_weight[
                          (head * head_dim + global_feature) * rank + row]
                    : 0.0f);
        }
        for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
            const int inner = linear / rank;
            const int row = linear - inner * rank;
            core_b(inner, row) = scaled_adjoint[row * rank + inner];
        }
        __syncthreads();
        CoreContent().execute(1.0f, core_a, core_b, 0.0f, core_c);
        __syncthreads();
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int feature = linear / rank;
            const int row = linear - feature * rank;
            const int64_t global_feature = feature_start + feature;
            if (global_feature < head_dim) {
                system_state_adjoint[row * head_dim + global_feature] +=
                    core_c(feature, row);
            }
        }
        __syncthreads();
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_compact_adjoint_kernel(
    const float* __restrict__ d_t,
    float* __restrict__ compact_state_reconstruction,
    const float* __restrict__ partial_eta,
    const float* __restrict__ tape,
    const float* __restrict__ state_adjoint,
    const float* __restrict__ eta_raw,
    float* __restrict__ frame_adjoint,
    float* __restrict__ grad_eta_raw,
    TrainingTapeLayout tape_layout,
    int64_t system_count,
    int64_t heads,
    int64_t head_dim,
    int64_t tile_count) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }
    const int64_t head = system_index - (system_index / heads) * heads;
    const float* system_tape = tape + system_index * tape_layout.stride;
    const float* compact_state = system_tape + tape_layout.z_offset;
    const float* equilibrium = system_tape + tape_layout.u_offset;
    const float* system_d_t = d_t + system_index * rank * head_dim;
    const float* system_state_adjoint = state_adjoint + system_index * rank * head_dim;
    float* system_compact_state_reconstruction =
        compact_state_reconstruction + system_index * rank * head_dim;
    const float eta = bounded_complement(eta_raw[head]);

    using Compact = typename GenericBackwardMathDx<rank>::Compact;
    constexpr size_t a_elements = cublasdx::cosize(Compact::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Compact::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Compact::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    __nv_bfloat16* a_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * a_elements, 16));
    __nv_bfloat16* b_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * b_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto compact_a = cublasdx::make_tensor(a_tile, Compact::get_layout_smem_a());
    auto compact_b = cublasdx::make_tensor(b_tile, Compact::get_layout_smem_b());
    auto compact_c = cublasdx::make_tensor(gemm_output, Compact::get_layout_smem_c());

    // The complement VJP needs the same strict-FP32 P^T C reconstruction as
    // the second compact product below.  Keep the scalar reduction here so
    // that reconstruction does not make a second global-memory pass.
    float eta_state_dot = 0.0f;

    for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    __syncthreads();
    for (int64_t feature_start = 0; feature_start < head_dim;
         feature_start += kRhsTile) {
        for (int linear = threadIdx.x; linear < rank * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int feature = linear - row * kRhsTile;
            const int64_t global_feature = feature_start + feature;
            compact_a(row, feature) = __float2bfloat16_rn(
                global_feature < head_dim
                    ? 2.0f * equilibrium[row * head_dim + global_feature] -
                        (1.0f + eta) * compact_state[row * head_dim + global_feature]
                    : 0.0f);
        }
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int feature = linear / rank;
            const int column = linear - feature * rank;
            const int64_t global_feature = feature_start + feature;
            compact_b(feature, column) = __float2bfloat16_rn(
                global_feature < head_dim
                    ? system_d_t[column * head_dim + global_feature]
                    : 0.0f);
        }
        __syncthreads();
        Compact().execute(-1.0f, compact_a, compact_b, 1.0f, compact_c);
        __syncthreads();
        for (int linear = threadIdx.x; linear < rank * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int feature = linear - row * kRhsTile;
            const int64_t global_feature = feature_start + feature;
            compact_a(row, feature) = __float2bfloat16_rn(
                global_feature < head_dim
                    ? system_state_adjoint[row * head_dim + global_feature]
                    : 0.0f);
        }
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int feature = linear / rank;
            const int column = linear - feature * rank;
            const int64_t global_feature = feature_start + feature;
            const float compact_value = global_feature < head_dim
                ? system_compact_state_reconstruction[
                      column * head_dim + global_feature]
                : 0.0f;
            if (global_feature < head_dim) {
                eta_state_dot +=
                    system_d_t[column * head_dim + global_feature] * compact_value;
            }
            compact_b(feature, column) = __float2bfloat16_rn(compact_value);
        }
        __syncthreads();
        Compact().execute(-1.0f, compact_a, compact_b, 1.0f, compact_c);
        __syncthreads();
        // This feature panel has consumed Z-hat. Reuse its dead storage for
        // the frame state read by each later token-level VJP CTA.
        for (int linear = threadIdx.x; linear < rank * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int feature = linear - row * kRhsTile;
            const int64_t global_feature = feature_start + feature;
            if (global_feature < head_dim) {
                system_compact_state_reconstruction[row * head_dim + global_feature] =
                    2.0f * equilibrium[row * head_dim + global_feature] -
                    (1.0f + eta) * compact_state[row * head_dim + global_feature];
            }
        }
        __syncthreads();
    }
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        frame_adjoint[system_index * rank * rank + linear] = compact_c(row, column);
    }

    static_assert(c_elements >= kThreads);
    float* const eta_reduction = gemm_output;
    float eta_content = 0.0f;
    for (int64_t tile = threadIdx.x; tile < tile_count; tile += blockDim.x) {
        eta_content += partial_eta[system_index * tile_count + tile];
    }
    eta_reduction[threadIdx.x] = eta_content;
    __syncthreads();
    reduce_sum(eta_reduction);
    const float total_eta_content = eta_reduction[0];
    eta_reduction[threadIdx.x] = eta_state_dot;
    __syncthreads();
    reduce_sum(eta_reduction);
    if (threadIdx.x == 0) {
        atomicAdd(
            grad_eta_raw + head,
            (total_eta_content - eta_reduction[0]) *
                bounded_complement_derivative(eta_raw[head]));
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_vjp_kernel(
    const float* __restrict__ frame_adjoint,
    const float* __restrict__ tape,
    float* __restrict__ frame_c,
    TrainingTapeLayout tape_layout,
    int64_t system_count) {
    using LeftTrsm = typename GenericBackwardMathDx<rank>::FrameVjpLeftTrsm;
    using RightTrsm = typename GenericBackwardMathDx<rank>::FrameVjpRightTrsm;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }
    const float* system_tape = tape + system_index * tape_layout.stride;
    const float* lower = system_tape + tape_layout.l_offset;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto [lower_tile, coordinates] = cusolverdx::shared_memory::slice<float, float>(
        shared_raw,
        alignof(float),
        rank * LeftTrsm::lda,
        alignof(float));
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lower_tile[row * LeftTrsm::lda + column] = lower[linear];
        const float* system_frame_adjoint = frame_adjoint + system_index * rank * rank;
        const float lower_value = row > column
            ? system_frame_adjoint[linear]
            : (row < column
                   ? system_frame_adjoint[column * rank + row]
                   : system_frame_adjoint[linear]);
        coordinates[row * LeftTrsm::ldb + column] = 0.5f * lower_value;
    }
    __syncthreads();
    LeftTrsm().execute(lower_tile, LeftTrsm::lda, coordinates, LeftTrsm::ldb);
    __syncthreads();
    RightTrsm().execute(lower_tile, RightTrsm::lda, coordinates, RightTrsm::ldb);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        frame_c[system_index * rank * rank + linear] =
            coordinates[row * LeftTrsm::ldb + column];
    }
}

// Algebraically this is the existing pair
//
//   D_P = G F^T + X D_U^T,
//   D_B = D_P L^{-1} + 2 B C,
//
// followed by length normalization. D_P has one consumer, so it stays in
// shared memory rather than round-tripping [B*H, N, R] FP32 storage.
template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void fused_frame_relation_vjp_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ projected,
    const float* __restrict__ state_adjoint,
    const float* __restrict__ frame_state,
    const float* __restrict__ tape,
    const float* __restrict__ frame_c,
    const float* __restrict__ valid_counts,
    scalar_t* __restrict__ grad_projected,
    TrainingTapeLayout tape_layout,
    int64_t batch_count,
    int64_t length,
    int64_t heads,
    int64_t dim,
    int64_t head_dim,
    int64_t tile_count) {
    constexpr int kRelationPanel = PanelRelationVjpMathDx<rank>::kTokenPanel;
    using Token = typename PanelRelationVjpMathDx<rank>::Token;
    using Trsm = typename PanelRelationVjpMathDx<rank>::Trsm;
    using Correction = typename PanelRelationVjpMathDx<rank>::Correction;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count || tile_index >= tile_count) {
        return;
    }
    const int64_t batch = system_index / heads;
    const int64_t head = system_index - batch * heads;
    const int64_t token_start = tile_index * kGenericVjpTokenTile;
    const int token_count = static_cast<int>(
        length - token_start < kGenericVjpTokenTile ? length - token_start : kGenericVjpTokenTile);
    const int64_t projected_width = heads * rank + dim;
    const float* system_state_adjoint = state_adjoint + system_index * rank * head_dim;
    const float* system_frame_state = frame_state + system_index * rank * head_dim;
    const float* system_tape = tape + system_index * tape_layout.stride;
    const float* lower = system_tape + tape_layout.l_offset;
    const float* c = frame_c + system_index * rank * rank;
    const float inverse_scale = 1.0f / system_tape[tape_layout.scale_offset];
    const float inverse_length_sqrt = valid_counts == nullptr
        ? rsqrtf(static_cast<float>(length))
        : rsqrtf(valid_counts[batch]);

    constexpr size_t token_a_elements = cublasdx::cosize(Token::get_layout_smem_a());
    constexpr size_t token_b_elements = cublasdx::cosize(Token::get_layout_smem_b());
    constexpr size_t token_c_elements = cublasdx::cosize(Token::get_layout_smem_c());
    constexpr size_t correction_a_elements =
        cublasdx::cosize(Correction::get_layout_smem_a());
    constexpr size_t correction_b_elements =
        cublasdx::cosize(Correction::get_layout_smem_b());
    constexpr size_t correction_c_elements =
        cublasdx::cosize(Correction::get_layout_smem_c());

    static_assert(
        token_a_elements * sizeof(__nv_bfloat16) <=
            kRelationPanel * Trsm::ldb * sizeof(float));
    static_assert((rank * Trsm::lda * sizeof(float)) % 16 == 0);
    static_assert(token_b_elements <= correction_a_elements);
    static_assert(token_c_elements <= correction_c_elements);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto [lower_tile, relation_adjoint] = cusolverdx::shared_memory::slice<float, float>(
        shared_raw,
        alignof(float),
        rank * Trsm::lda,
        alignof(float));
    SharedCursor shared(shared_raw + Trsm::shared_memory_size);
    __nv_bfloat16* correction_a = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * correction_a_elements, 16));
    __nv_bfloat16* correction_b = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * correction_b_elements, 16));
    float* correction_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * correction_c_elements, 16));

    auto token_a = cublasdx::make_tensor(
        reinterpret_cast<__nv_bfloat16*>(relation_adjoint), Token::get_layout_smem_a());
    auto token_b = cublasdx::make_tensor(correction_a, Token::get_layout_smem_b());
    auto token_c = cublasdx::make_tensor(correction_output, Token::get_layout_smem_c());
    auto correction_a_tensor = cublasdx::make_tensor(
        correction_a, Correction::get_layout_smem_a());
    auto correction_b_tensor = cublasdx::make_tensor(
        correction_b, Correction::get_layout_smem_b());
    auto correction_c_tensor = cublasdx::make_tensor(
        correction_output, Correction::get_layout_smem_c());

    for (int token_offset = 0; token_offset < token_count; token_offset += kRelationPanel) {
        const int subtile_count = token_count - token_offset < kRelationPanel
            ? token_count - token_offset
            : kRelationPanel;
        for (int linear = threadIdx.x; linear < token_c_elements; linear += blockDim.x) {
            correction_output[linear] = 0.0f;
        }
        __syncthreads();
        for (int64_t feature_start = 0; feature_start < head_dim;
             feature_start += kRhsTile) {
            for (int linear = threadIdx.x; linear < kRelationPanel * kRhsTile;
                 linear += blockDim.x) {
                const int row = linear / kRhsTile;
                const int column = linear - row * kRhsTile;
                const int64_t token = token_start + token_offset + row;
                const int64_t feature = feature_start + column;
                token_a(row, column) = __float2bfloat16_rn(
                    row < subtile_count && feature < head_dim
                        ? load_scalar(grad_output,
                              (batch * length + token) * dim + head * head_dim + feature)
                        : 0.0f);
            }
            for (int linear = threadIdx.x; linear < kRhsTile * rank;
                 linear += blockDim.x) {
                const int feature = linear / rank;
                const int row = linear - feature * rank;
                const int64_t global_feature = feature_start + feature;
                token_b(feature, row) = __float2bfloat16_rn(
                    global_feature < head_dim
                        ? system_frame_state[row * head_dim + global_feature]
                        : 0.0f);
            }
            __syncthreads();
            Token().execute(1.0f, token_a, token_b, 1.0f, token_c);
            __syncthreads();
            for (int linear = threadIdx.x; linear < kRelationPanel * kRhsTile;
                 linear += blockDim.x) {
                const int row = linear / kRhsTile;
                const int column = linear - row * kRhsTile;
                const int64_t token = token_start + token_offset + row;
                const int64_t feature = feature_start + column;
                token_a(row, column) = __float2bfloat16_rn(
                    row < subtile_count && feature < head_dim
                        ? load_scalar(
                              projected,
                              (batch * length + token) * projected_width + heads * rank +
                                  head * head_dim + feature)
                        : 0.0f);
            }
            for (int linear = threadIdx.x; linear < kRhsTile * rank;
                 linear += blockDim.x) {
                const int feature = linear / rank;
                const int row = linear - feature * rank;
                const int64_t global_feature = feature_start + feature;
                token_b(feature, row) = __float2bfloat16_rn(
                    global_feature < head_dim
                        ? system_state_adjoint[row * head_dim + global_feature]
                        : 0.0f);
            }
            __syncthreads();
            Token().execute(1.0f, token_a, token_b, 1.0f, token_c);
            __syncthreads();
        }

        for (int linear = threadIdx.x; linear < kRelationPanel * rank;
             linear += blockDim.x) {
            const int row = linear / rank;
            const int column = linear - row * rank;
            relation_adjoint[row * Trsm::ldb + column] = row < subtile_count
                ? token_c(row, column)
                : 0.0f;
        }
        __syncthreads();

        for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
            const int row = linear / rank;
            const int column = linear - row * rank;
            lower_tile[row * Trsm::lda + column] = lower[linear];
            correction_b_tensor(row, column) = __float2bfloat16_rn(c[linear]);
        }
        for (int linear = threadIdx.x; linear < kRelationPanel * (rank / 2);
             linear += blockDim.x) {
            const int row = linear / (rank / 2);
            const int pair = linear - row * (rank / 2);
            float2 value = make_float2(0.0f, 0.0f);
            if (row < subtile_count) {
                value = generic_relation_pair<scalar_t, rank>(
                    projected, batch, head, token_start + token_offset + row,
                    pair, length, projected_width, inverse_length_sqrt);
                value.x *= inverse_scale;
                value.y *= inverse_scale;
            }
            correction_a_tensor(row, 2 * pair) = __float2bfloat16_rn(value.x);
            correction_a_tensor(row, 2 * pair + 1) = __float2bfloat16_rn(value.y);
        }
        for (int linear = threadIdx.x; linear < correction_c_elements;
             linear += blockDim.x) {
            correction_output[linear] = 0.0f;
        }
        __syncthreads();
        Trsm().execute(lower_tile, Trsm::lda, relation_adjoint, Trsm::ldb);
        __syncthreads();
        Correction().execute(
            2.0f,
            correction_a_tensor,
            correction_b_tensor,
            0.0f,
            correction_c_tensor);
        __syncthreads();

        for (int linear = threadIdx.x; linear < subtile_count * (rank / 2);
             linear += blockDim.x) {
            const int local_token = linear / (rank / 2);
            const int pair = linear - local_token * (rank / 2);
            const int even_rank = 2 * pair;
            const int64_t token = token_start + token_offset + local_token;
            const float d_even = (
                relation_adjoint[local_token * Trsm::ldb + even_rank] +
                correction_c_tensor(local_token, even_rank)) * inverse_scale;
            const float d_odd = (
                relation_adjoint[local_token * Trsm::ldb + even_rank + 1] +
                correction_c_tensor(local_token, even_rank + 1)) * inverse_scale;
            const int64_t relation_base =
                (batch * length + token) * projected_width + head * rank;
            store_scalar(
                grad_projected,
                relation_base + even_rank,
                d_even * inverse_length_sqrt);
            store_scalar(
                grad_projected,
                relation_base + even_rank + 1,
                d_odd * inverse_length_sqrt);
        }
        __syncthreads();
    }
}

template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void generic_content_vjp_kernel(
    const scalar_t* __restrict__ grad_output,
    const float* __restrict__ tape,
    const float* __restrict__ state_adjoint,
    const float* __restrict__ eta_raw,
    scalar_t* __restrict__ grad_projected,
    TrainingTapeLayout tape_layout,
    int64_t batch_count,
    int64_t length,
    int64_t heads,
    int64_t dim,
    int64_t head_dim) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count) {
        return;
    }
    const int64_t batch = system_index / heads;
    const int64_t head = system_index - batch * heads;
    const int64_t token_start = tile_index * kGenericVjpTokenTile;
    const int token_count = static_cast<int>(
        length - token_start < kGenericVjpTokenTile ? length - token_start : kGenericVjpTokenTile);
    const int64_t projected_width = heads * rank + dim;
    const float* system_tape = tape + system_index * tape_layout.stride;
    const float* frame = system_tape + tape_layout.p_offset;
    const float* system_state_adjoint = state_adjoint + system_index * rank * head_dim;
    const float eta = bounded_complement(eta_raw[head]);

    using Content = typename GenericBackwardMathDx<rank>::Content;
    constexpr size_t a_elements = cublasdx::cosize(Content::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Content::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Content::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    __nv_bfloat16* a_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * a_elements, 16));
    __nv_bfloat16* b_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * b_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto content_a = cublasdx::make_tensor(a_tile, Content::get_layout_smem_a());
    auto content_b = cublasdx::make_tensor(b_tile, Content::get_layout_smem_b());
    auto content_c = cublasdx::make_tensor(gemm_output, Content::get_layout_smem_c());

    for (int token_offset = 0; token_offset < token_count; token_offset += kRhsTile) {
        const int subtile_count = token_count - token_offset < kRhsTile
            ? token_count - token_offset
            : kRhsTile;
        for (int64_t feature_start = 0; feature_start < head_dim;
             feature_start += kRhsTile) {
            for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
                gemm_output[linear] = 0.0f;
            }
            for (int linear = threadIdx.x; linear < kRhsTile * rank;
                 linear += blockDim.x) {
                const int row = linear / rank;
                const int column = linear - row * rank;
                const int64_t token = token_start + token_offset + row;
                content_a(row, column) = __float2bfloat16_rn(
                    row < subtile_count ? frame[token * rank + column] : 0.0f);
            }
            for (int linear = threadIdx.x; linear < rank * kRhsTile;
                 linear += blockDim.x) {
                const int row = linear / kRhsTile;
                const int column = linear - row * kRhsTile;
                const int64_t feature = feature_start + column;
                content_b(row, column) = __float2bfloat16_rn(
                    feature < head_dim
                        ? system_state_adjoint[row * head_dim + feature]
                        : 0.0f);
            }
            __syncthreads();
            Content().execute(1.0f, content_a, content_b, 0.0f, content_c);
            __syncthreads();
            for (int linear = threadIdx.x; linear < kRhsTile * kRhsTile;
                 linear += blockDim.x) {
                const int row = linear / kRhsTile;
                const int column = linear - row * kRhsTile;
                const int64_t token = token_start + token_offset + row;
                const int64_t feature = feature_start + column;
                if (row < subtile_count && feature < head_dim) {
                    const float value = eta * load_scalar(grad_output,
                        (batch * length + token) * dim + head * head_dim + feature) +
                        content_c(row, column);
                    store_scalar(
                        grad_projected,
                        (batch * length + token) * projected_width + heads * rank +
                            head * head_dim + feature,
                        value);
                }
            }
            __syncthreads();
        }
    }
}

template <typename scalar_t, int rank>
void configure_backward_kernel_attributes(int device) {
    static std::mutex mutex;
    static std::unordered_set<int> configured_devices;
    std::lock_guard<std::mutex> lock(mutex);
    if (configured_devices.find(device) != configured_devices.end()) {
        return;
    }

    constexpr size_t core_content_vjp_shared_bytes = generic_core_content_vjp_shared_bytes<rank>();
    constexpr size_t frame_compact_shared_bytes = generic_frame_compact_shared_bytes<rank>();
    constexpr size_t frame_relation_shared_bytes =
        fused_frame_relation_vjp_shared_bytes<rank>();
    constexpr size_t frame_vjp_shared_bytes = generic_frame_vjp_shared_bytes<rank>();
    constexpr size_t content_vjp_shared_bytes = generic_content_vjp_shared_bytes<rank>();
    using Getrs = typename GenericEquilibriumGetrsMathDx<rank>::Transposed;
    constexpr size_t equilibrium_vjp_shared_bytes = Getrs::shared_memory_size;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_core_content_vjp_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(core_content_vjp_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_compact_adjoint_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(frame_compact_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fused_frame_relation_vjp_kernel<scalar_t, rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(frame_relation_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_vjp_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(frame_vjp_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_content_vjp_kernel<scalar_t, rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(content_vjp_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_equilibrium_vjp_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(equilibrium_vjp_shared_bytes)));
    configured_devices.insert(device);
}

template <typename scalar_t, int rank>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> launch_generic_backward(
    const at::Tensor& grad_output,
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const at::Tensor& tape,
    const at::Tensor& pivots,
    const c10::optional<at::Tensor>& valid_counts,
    FastPathShape shape) {
    c10::cuda::CUDAGuard guard(projected.device());
    configure_backward_kernel_attributes<scalar_t, rank>(projected.get_device());
    const int64_t system_count = shape.batch * shape.heads;
    const int64_t token_tiles =
        (shape.length + kGenericVjpTokenTile - 1) / kGenericVjpTokenTile;
    const int64_t rhs_tiles = (shape.head_dim + kRhsTile - 1) / kRhsTile;
    const auto tape_layout = training_tape_layout(shape);
    const auto options = projected.options().dtype(at::kFloat);
    auto grad_projected = at::empty_like(projected);
    auto grad_core_base_raw = at::empty_like(core_base_raw);
    auto grad_core_drive_weight = at::empty_like(core_drive_weight);
    auto grad_eta_raw = at::empty_like(eta_raw);
    auto partial_d_t = at::empty(
        {system_count, token_tiles, rank, shape.head_dim}, options);
    auto partial_eta = at::empty({system_count, token_tiles}, options);
    auto d_t = at::empty({system_count, rank, shape.head_dim}, options);
    auto equilibrium_adjoint = at::empty({system_count, rank, shape.head_dim}, options);
    auto compact_state_adjoint = at::empty({system_count, rank, shape.head_dim}, options);
    auto d_k = at::empty({system_count, rank, rank}, options);
    // The core VJP is d_k's only consumer; its compact matrix storage then
    // holds the direct QR-frame adjoint.
    auto frame_adjoint = d_k;
    auto frame_c = at::empty({system_count, rank, rank}, options);

    const auto stream = at::cuda::getCurrentCUDAStream(projected.get_device()).stream();
    if (grad_core_base_raw.numel() > 0) C10_CUDA_CHECK(cudaMemsetAsync(
        grad_core_base_raw.data_ptr<float>(),
        0,
        grad_core_base_raw.numel() * sizeof(float),
        stream));
    if (grad_core_drive_weight.numel() > 0) C10_CUDA_CHECK(cudaMemsetAsync(
        grad_core_drive_weight.data_ptr<float>(),
        0,
        grad_core_drive_weight.numel() * sizeof(float),
        stream));
    C10_CUDA_CHECK(cudaMemsetAsync(
        grad_eta_raw.data_ptr<float>(),
        0,
        grad_eta_raw.numel() * sizeof(float),
        stream));

    const dim3 token_grid(
        static_cast<unsigned int>(system_count), static_cast<unsigned int>(token_tiles));
    const dim3 rhs_grid(
        static_cast<unsigned int>(system_count), static_cast<unsigned int>(rhs_tiles));
    const auto reduction_launch = generic_partial_reduction_launch(
        system_count, token_tiles, rank * shape.head_dim);
    const dim3 reduction_grid(
        static_cast<unsigned int>(system_count),
        static_cast<unsigned int>(reduction_launch.output_slices));
    generic_statistic_partials_kernel<scalar_t, rank, true><<<token_grid, kThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        projected.data_ptr<scalar_t>(),
        tape.data_ptr<float>(),
        partial_d_t.data_ptr<float>(),
        partial_eta.data_ptr<float>(),
        tape_layout,
        shape.batch,
        shape.length,
        shape.heads,
        shape.dim,
        shape.head_dim,
        token_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    generic_reduce_partials_kernel<rank><<<
        reduction_grid, reduction_launch.threads, 0, stream>>>(
        partial_d_t.data_ptr<float>(),
        d_t.data_ptr<float>(),
        system_count,
        token_tiles,
        shape.head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    generic_statistic_partials_kernel<scalar_t, rank, false><<<token_grid, kThreads, 0, stream>>>(
        nullptr,
        projected.data_ptr<scalar_t>(),
        tape.data_ptr<float>(),
        partial_d_t.data_ptr<float>(),
        nullptr,
        tape_layout,
        shape.batch,
        shape.length,
        shape.heads,
        shape.dim,
        shape.head_dim,
        token_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    using Getrs = typename GenericEquilibriumGetrsMathDx<rank>::Transposed;
    constexpr size_t equilibrium_vjp_shared_bytes = Getrs::shared_memory_size;
    generic_equilibrium_vjp_kernel<rank><<<
        rhs_grid, kRhsTile, equilibrium_vjp_shared_bytes, stream>>>(
        d_t.data_ptr<float>(),
        tape.data_ptr<float>(),
        pivots.data_ptr<int>(),
        eta_raw.data_ptr<float>(),
        equilibrium_adjoint.data_ptr<float>(),
        compact_state_adjoint.data_ptr<float>(),
        tape_layout,
        system_count,
        shape.heads,
        shape.head_dim, shape.core_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (shape.core_mode != 2) {
    generic_d_k_kernel<rank><<<system_count, kThreads, 0, stream>>>(
        equilibrium_adjoint.data_ptr<float>(),
        tape.data_ptr<float>(),
        d_k.data_ptr<float>(),
        tape_layout,
        system_count,
        shape.head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    // d_k is the equilibrium adjoint's final consumer.  Reduce the exact
    // FP32 compact-state partials once into that dead buffer: this is the
    // same ascending-tile sum previously reconstructed per frame CTA.
    generic_reduce_partials_kernel<rank><<<
        reduction_grid, reduction_launch.threads, 0, stream>>>(
        partial_d_t.data_ptr<float>(),
        equilibrium_adjoint.data_ptr<float>(),
        system_count,
        token_tiles,
        shape.head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (shape.core_mode != 2) {
    generic_core_relation_vjp_kernel<rank><<<system_count, kThreads, 0, stream>>>(
        tape.data_ptr<float>(),
        d_k.data_ptr<float>(),
        grad_core_base_raw.data_ptr<float>(),
        tape_layout,
        system_count,
        shape.heads, shape.core_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (shape.core_mode == 0) {
    constexpr size_t core_content_vjp_shared_bytes = generic_core_content_vjp_shared_bytes<rank>();
    generic_core_content_vjp_kernel<rank><<<
        system_count, kThreads, core_content_vjp_shared_bytes, stream>>>(
        tape.data_ptr<float>(),
        core_drive_weight.data_ptr<float>(),
        d_k.data_ptr<float>(),
        valid_counts.has_value() ? valid_counts->data_ptr<float>() : nullptr,
        compact_state_adjoint.data_ptr<float>(),
        grad_core_drive_weight.data_ptr<float>(),
        tape_layout,
        system_count,
        shape.heads,
        shape.head_dim,
        1.0f / std::sqrt(static_cast<float>(shape.length)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    constexpr size_t frame_compact_shared_bytes = generic_frame_compact_shared_bytes<rank>();
    generic_frame_compact_adjoint_kernel<rank><<<
        system_count, kThreads, frame_compact_shared_bytes, stream>>>(
        d_t.data_ptr<float>(),
        equilibrium_adjoint.data_ptr<float>(),
        partial_eta.data_ptr<float>(),
        tape.data_ptr<float>(),
        compact_state_adjoint.data_ptr<float>(),
        eta_raw.data_ptr<float>(),
        frame_adjoint.data_ptr<float>(),
        grad_eta_raw.data_ptr<float>(),
        tape_layout,
        system_count,
        shape.heads,
        shape.head_dim,
        token_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    constexpr size_t frame_vjp_shared_bytes = generic_frame_vjp_shared_bytes<rank>();
    generic_frame_vjp_kernel<rank><<<
        system_count, kThreads, frame_vjp_shared_bytes, stream>>>(
        frame_adjoint.data_ptr<float>(),
        tape.data_ptr<float>(),
        frame_c.data_ptr<float>(),
        tape_layout,
        system_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    constexpr size_t content_vjp_shared_bytes = generic_content_vjp_shared_bytes<rank>();
    generic_content_vjp_kernel<scalar_t, rank><<<
        token_grid, kThreads, content_vjp_shared_bytes, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        tape.data_ptr<float>(),
        compact_state_adjoint.data_ptr<float>(),
        eta_raw.data_ptr<float>(),
        grad_projected.data_ptr<scalar_t>(),
        tape_layout,
        shape.batch,
        shape.length,
        shape.heads,
        shape.dim,
        shape.head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    constexpr size_t frame_relation_shared_bytes =
        fused_frame_relation_vjp_shared_bytes<rank>();
    fused_frame_relation_vjp_kernel<scalar_t, rank><<<
        token_grid, kThreads, frame_relation_shared_bytes, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        projected.data_ptr<scalar_t>(),
        compact_state_adjoint.data_ptr<float>(),
        equilibrium_adjoint.data_ptr<float>(),
        tape.data_ptr<float>(),
        frame_c.data_ptr<float>(),
        valid_counts.has_value() ? valid_counts->data_ptr<float>() : nullptr,
        grad_projected.data_ptr<scalar_t>(),
        tape_layout,
        shape.batch,
        shape.length,
        shape.heads,
        shape.dim,
        shape.head_dim,
        token_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_projected,
        grad_core_base_raw,
        grad_core_drive_weight,
        grad_eta_raw,
    };
}

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch_expanded_backward(
    const at::Tensor& grad_output,
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const at::Tensor& tape,
    const at::Tensor& pivots,
    const c10::optional<at::Tensor>& valid_counts,
    FastPathShape shape) {
    switch (shape.rank) {
        case 16:
            return launch_generic_backward<scalar_t, 16>(
                grad_output, projected, core_base_raw, core_drive_weight, eta_raw,
                tape, pivots, valid_counts, shape);
        case 32:
            return launch_generic_backward<scalar_t, 32>(
                grad_output, projected, core_base_raw, core_drive_weight, eta_raw,
                tape, pivots, valid_counts, shape);
        case 48:
            return launch_generic_backward<scalar_t, 48>(
                grad_output, projected, core_base_raw, core_drive_weight, eta_raw,
                tape, pivots, valid_counts, shape);
        case 64:
            return launch_generic_backward<scalar_t, 64>(
                grad_output, projected, core_base_raw, core_drive_weight, eta_raw,
                tape, pivots, valid_counts, shape);
        default:
            TORCH_CHECK(false, "unreachable supported rank");
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const at::Tensor& tape,
    const at::Tensor& pivots,
    const c10::optional<at::Tensor>& valid_counts) {
    const auto shape = validate_fast_inputs(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        valid_counts);
    validate_grad_output(grad_output, projected, shape);
    validate_training_tape(tape, pivots, projected, shape);
    c10::cuda::CUDAGuard guard(projected.device());
    (void)supported_sm();

    return dispatch_expanded_backward<at::BFloat16>(
        grad_output,
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        tape,
        pivots,
        valid_counts,
        shape);
}

}  // namespace lsso_equilibrium
