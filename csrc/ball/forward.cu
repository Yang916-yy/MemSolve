#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cublasdx.hpp>
#include <cusolverdx.hpp>
#include <cusolverdx/detail/shared_memory.hpp>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
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

struct ForwardResult {
    at::Tensor output;
    at::Tensor tape;
    at::Tensor pivots;
};

constexpr int kGenericTokenTile = 32;
// Selective fusion: accumulate four adjacent GEMMs per CTA, then reduce
// independent groups. Keep token-group parallelism instead of a full-sequence CTA.
constexpr int kCrossTilesPerBlock = 4;
constexpr int kCrossTokenChunk = 64;
// Below this point, an extra partial-Gram launch and global workspace cost more
// than keeping the complete reduction in one system block.
constexpr int kParallelGramMinimumTokenTiles = 32;

template <int rank>
constexpr int generic_frame_materialize_token_tile() {
    // One 128-token right-hand-side panel amortizes the Cholesky-factor load
    // and triangular-solve setup across four adjacent token tiles.
    return 128;
}

template <int rank>
struct GenericFrameMathDx {
    using Gram = decltype(
        cublasdx::Size<rank, rank, kGenericTokenTile>() +
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

    using Cross = decltype(
        cublasdx::Size<rank, kRhsTile, kGenericTokenTile>() +
        cublasdx::Precision<__nv_bfloat16, __nv_bfloat16, float>() +
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

    using Core = decltype(
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

    using Readout = decltype(
        cublasdx::Size<kGenericTokenTile, kRhsTile, rank>() +
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

    using FrameBase = decltype(
        cusolverdx::Size<rank, rank, kRhsTile>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kThreads>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());

    using Potrf = decltype(
        FrameBase() + cusolverdx::Function<cusolverdx::function::potrf>());

    using FrameTrsm = decltype(
        cusolverdx::Size<generic_frame_materialize_token_tile<rank>(), rank>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::right>() +
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
};

template <int rank>
struct GenericCoreFactorMathDx;

template <int rank>
constexpr size_t generic_frame_shared_bytes() {
    using Traits = GenericFrameMathDx<rank>;
    using Gram = typename Traits::Gram;
    using Potrf = typename Traits::Potrf;
    constexpr size_t a_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr size_t c_elements = cublasdx::cosize(Gram::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    offset = align_shared_offset(offset, alignof(float));
    offset += Potrf::shared_memory_size;
    return offset;
}

template <int rank>
constexpr size_t generic_frame_gram_partial_shared_bytes() {
    using Gram = typename GenericFrameMathDx<rank>::Gram;
    constexpr size_t a_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr size_t c_elements = cublasdx::cosize(Gram::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    return offset;
}

template <int rank>
constexpr size_t generic_frame_factor_partials_shared_bytes() {
    using Potrf = typename GenericFrameMathDx<rank>::Potrf;
    return Potrf::shared_memory_size;
}

template <int rank>
constexpr size_t generic_cross_shared_bytes() {
    using Cross = typename GenericFrameMathDx<rank>::Cross;
    constexpr size_t a_elements = cublasdx::cosize(Cross::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Cross::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Cross::get_layout_smem_c());
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
constexpr size_t generic_frame_materialize_shared_bytes() {
    using FrameTrsm = typename GenericFrameMathDx<rank>::FrameTrsm;
    return FrameTrsm::shared_memory_size;
}

template <int rank>
constexpr size_t generic_core_factor_shared_bytes() {
    using Core = typename GenericFrameMathDx<rank>::Core;
    using Getrf = typename GenericCoreFactorMathDx<rank>::Getrf;
    constexpr size_t ab_bytes = sizeof(__nv_bfloat16) * (
        cublasdx::cosize(Core::get_layout_smem_a()) +
        cublasdx::cosize(Core::get_layout_smem_b()));
    constexpr size_t c_bytes = sizeof(float) * cublasdx::cosize(Core::get_layout_smem_c());
    // A/B die after the final GEMM. The accumulator must remain separate
    // while the solver scratch first holds F and then its assembled matrix.
    return align_shared_offset(c_bytes, 16) +
        (ab_bytes > Getrf::shared_memory_size ? ab_bytes : Getrf::shared_memory_size);
}

template <int rank>
constexpr size_t generic_output_shared_bytes() {
    using Readout = typename GenericFrameMathDx<rank>::Readout;
    constexpr size_t a_elements = cublasdx::cosize(Readout::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Readout::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Readout::get_layout_smem_c());
    size_t offset = 0;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * a_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(__nv_bfloat16) * b_elements;
    offset = align_shared_offset(offset, 16);
    offset += sizeof(float) * c_elements;
    return offset;
}

template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_kernel(
    const scalar_t* __restrict__ projected,
    const float* __restrict__ valid_counts,
    float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    int64_t batch_count,
    int64_t length,
    int64_t heads,
    int64_t dim) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count) {
        return;
    }
    const int64_t batch = system_index / heads;
    const int64_t head = system_index - batch * heads;
    const int64_t projected_width = heads * rank + dim;
    const float inverse_length_sqrt = valid_counts == nullptr
        ? rsqrtf(static_cast<float>(length))
        : rsqrtf(valid_counts[batch]);
    float* system_tape = tape + system_index * workspace_layout.stride;
    float* b = system_tape + workspace_layout.b_offset;

    __shared__ float reduction[kThreads];

    float local_scale = 1.0f;
    for (int64_t linear = threadIdx.x; linear < length * (rank / 2);
         linear += blockDim.x) {
        const int64_t token = linear / (rank / 2);
        const int pair = static_cast<int>(linear - token * (rank / 2));
        const float2 relation = generic_relation_pair<scalar_t, rank>(
            projected,
            batch,
            head,
            token,
            pair,
            length,
            projected_width,
            inverse_length_sqrt);
        b[token * rank + 2 * pair] = relation.x;
        b[token * rank + 2 * pair + 1] = relation.y;
        local_scale = fmaxf(local_scale, fmaxf(fabsf(relation.x), fabsf(relation.y)));
    }
    reduction[threadIdx.x] = local_scale;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = reduction[0];
    const float inverse_scale = 1.0f / scale;
    if (scale != 1.0f) {
        for (int64_t linear = threadIdx.x; linear < length * rank;
             linear += blockDim.x) {
            b[linear] *= inverse_scale;
        }
    }
    if (threadIdx.x == 0) {
        system_tape[workspace_layout.scale_offset] = scale;
    }
}


// Split only long, low-system-count scans. All paths preserve the same detached
// maximum scale and FP32 normalization; no sequence-wide CTA is needed here.
constexpr int kScaleTokenTile = 256;

template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void relation_scale_partials_kernel(
    const scalar_t* projected, const float* valid_counts, float* partials,
    int64_t length, int64_t heads, int64_t dim, int64_t tiles) {
    const int64_t system = blockIdx.x;
    const int64_t batch = system / heads;
    const int64_t head = system % heads;
    const float normalizer = rsqrtf(valid_counts ? valid_counts[batch] : float(length));
    float value = 1.0f;
    const int64_t start = int64_t(blockIdx.y) * kScaleTokenTile;
    for (int i = threadIdx.x; i < kScaleTokenTile * rank; i += blockDim.x) {
        const int64_t token = start + i / rank;
        if (token < length) {
            value = fmaxf(value, fabsf(load_scalar(projected,
                (batch * length + token) * (heads * rank + dim) + head * rank + i % rank) * normalizer));
        }
    }
    __shared__ float values[kThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[system * tiles + blockIdx.y] = values[0];
}

__global__ __launch_bounds__(kThreads) void relation_scale_reduce_kernel(
    const float* partials, float* tape, ForwardWorkspaceLayout layout, int64_t tiles) {
    float value = 1.0f;
    for (int64_t i = threadIdx.x; i < tiles; i += blockDim.x)
        value = fmaxf(value, partials[int64_t(blockIdx.x) * tiles + i]);
    __shared__ float values[kThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) tape[int64_t(blockIdx.x) * layout.stride + layout.scale_offset] = values[0];
}

template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void relation_prepare_tiled_kernel(
    const scalar_t* projected, const float* valid_counts, float* tape,
    ForwardWorkspaceLayout layout, int64_t length, int64_t heads, int64_t dim) {
    const int64_t system = blockIdx.x, batch = system / heads, head = system % heads;
    const float normalizer = rsqrtf(valid_counts ? valid_counts[batch] : float(length));
    float* state = tape + system * layout.stride;
    const float inverse_scale = 1.0f / state[layout.scale_offset];
    const int64_t start = int64_t(blockIdx.y) * kScaleTokenTile;
    for (int i = threadIdx.x; i < kScaleTokenTile * rank; i += blockDim.x) {
        const int64_t token = start + i / rank;
        if (token < length) {
            const float a = load_scalar(projected,
                (batch * length + token) * (heads * rank + dim) + head * rank + i % rank) * normalizer;
            state[layout.b_offset + token * rank + i % rank] = a * inverse_scale;
        }
    }
}

// Bound each serial partial sum, exposing additional independent CTAs before
// the small Cholesky factorization. FP32 reduction order remains deterministic.
template <int rank>
__global__ __launch_bounds__(kThreads) void gram_group_reduce_kernel(
    const float* input, float* output, int64_t tiles, int64_t groups) {
    constexpr int elements = rank * (rank + 1) / 2;
    constexpr int group_size = 32;
    const int64_t system = blockIdx.x, group = blockIdx.y;
    for (int i = threadIdx.x; i < elements; i += blockDim.x) {
        float value = 0.0f;
        const int64_t end = min(tiles, (group + 1) * group_size);
        for (int64_t t = group * group_size; t < end; ++t)
            value += input[(system * tiles + t) * elements + i];
        output[(system * groups + group) * elements + i] = value;
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_factor_kernel(
    const float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    int64_t system_count,
    int64_t length) {
    using Traits = GenericFrameMathDx<rank>;
    using Gram = typename Traits::Gram;
    using Potrf = typename Traits::Potrf;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* b = system_tape + workspace_layout.b_offset;
    float* lower_tape = const_cast<float*>(system_tape) + workspace_layout.l_offset;

    constexpr size_t a_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr size_t c_elements = cublasdx::cosize(Gram::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    float* a_tile = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * a_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto* factor_block = static_cast<unsigned char*>(
        shared.take_bytes(Potrf::shared_memory_size, alignof(float)));
    float* factor = reinterpret_cast<float*>(factor_block);
    auto gram_a = cublasdx::make_tensor(a_tile, Gram::get_layout_smem_a());
    auto gram_b = cublasdx::make_tensor(a_tile, Gram::get_layout_smem_b());
    auto gram_c = cublasdx::make_tensor(gemm_output, Gram::get_layout_smem_c());
    __shared__ typename Potrf::status_type info;

    for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    __syncthreads();
    const float scale = system_tape[workspace_layout.scale_offset];
    const float regularizer = 1.0f / (scale * scale);
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        gram_c(row, column) = row == column ? regularizer : 0.0f;
    }
    __syncthreads();
    for (int64_t token_start = 0; token_start < length; token_start += kGenericTokenTile) {
        for (int linear = threadIdx.x; linear < rank * kGenericTokenTile;
             linear += blockDim.x) {
            const int column = linear / rank;
            const int row = linear - column * rank;
            const int64_t token = token_start + column;
            gram_a(row, column) = token < length ? b[token * rank + row] : 0.0f;
        }
        __syncthreads();
        Gram().execute(1.0f, gram_a, gram_b, 1.0f, gram_c);
        __syncthreads();
    }

    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        factor[row * Potrf::lda + column] = gram_c(row, column);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        info = 0;
    }
    __syncthreads();
    Potrf().execute(factor, Potrf::lda, &info);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lower_tape[linear] = row >= column
            ? factor[row * Potrf::lda + column]
            : 0.0f;
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_gram_partials_kernel(
    const float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    float* __restrict__ gram_partials,
    int64_t gram_partial_system_stride,
    int64_t system_count,
    int64_t length,
    int64_t token_tiles) {
    using Traits = GenericFrameMathDx<rank>;
    using Gram = typename Traits::Gram;

    const int64_t work_index = static_cast<int64_t>(blockIdx.x);
    const int64_t work_count = system_count * token_tiles;
    if (work_index >= work_count) {
        return;
    }
    const int64_t system_index = work_index / token_tiles;
    const int64_t tile_index = work_index - system_index * token_tiles;

    constexpr int kPackedLowerElements = rank * (rank + 1) / 2;
    constexpr size_t a_elements = cublasdx::cosize(Gram::get_layout_smem_a());
    constexpr size_t c_elements = cublasdx::cosize(Gram::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    float* a_tile = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * a_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto gram_a = cublasdx::make_tensor(a_tile, Gram::get_layout_smem_a());
    auto gram_b = cublasdx::make_tensor(a_tile, Gram::get_layout_smem_b());
    auto gram_c = cublasdx::make_tensor(gemm_output, Gram::get_layout_smem_c());

    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* b = system_tape + workspace_layout.b_offset;
    const int64_t token_start = tile_index * kGenericTokenTile;
    for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    for (int linear = threadIdx.x; linear < rank * kGenericTokenTile;
         linear += blockDim.x) {
        const int column = linear / rank;
        const int row = linear - column * rank;
        const int64_t token = token_start + column;
        gram_a(row, column) = token < length ? b[token * rank + row] : 0.0f;
    }
    __syncthreads();
    Gram().execute(1.0f, gram_a, gram_b, 0.0f, gram_c);
    __syncthreads();

    float* partial = gram_partials +
        system_index * gram_partial_system_stride +
        tile_index * kPackedLowerElements;
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        if (row >= column) {
            partial[row * (row + 1) / 2 + column] = gram_c(row, column);
        }
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_factor_partials_kernel(
    const float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    const float* __restrict__ gram_partials,
    int64_t gram_partial_system_stride,
    int64_t system_count,
    int64_t token_tiles) {
    using Traits = GenericFrameMathDx<rank>;
    using Potrf = typename Traits::Potrf;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    if (system_index >= system_count) {
        return;
    }

    constexpr int kPackedLowerElements = rank * (rank + 1) / 2;
    const float* system_tape = tape + system_index * workspace_layout.stride;
    float* lower_tape = const_cast<float*>(system_tape) + workspace_layout.l_offset;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    auto* factor_block = static_cast<unsigned char*>(
        shared.take_bytes(Potrf::shared_memory_size, alignof(float)));
    float* factor = reinterpret_cast<float*>(factor_block);
    __shared__ typename Potrf::status_type info;

    const float scale = system_tape[workspace_layout.scale_offset];
    const float regularizer = 1.0f / (scale * scale);
    const float* system_partials = gram_partials +
        system_index * gram_partial_system_stride;
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        float value = row == column ? regularizer : 0.0f;
        if (row >= column) {
            const int packed_index = row * (row + 1) / 2 + column;
            for (int64_t tile = 0; tile < token_tiles; ++tile) {
                value += system_partials[tile * kPackedLowerElements + packed_index];
            }
        }
        factor[row * Potrf::lda + column] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        info = 0;
    }
    __syncthreads();
    Potrf().execute(factor, Potrf::lda, &info);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lower_tape[linear] = row >= column
            ? factor[row * Potrf::lda + column]
            : 0.0f;
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_frame_materialize_kernel(
    const float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    int64_t frame_offset,
    int64_t system_count,
    int64_t length) {
    using FrameTrsm = typename GenericFrameMathDx<rank>::FrameTrsm;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t tile_index = static_cast<int64_t>(blockIdx.y);
    if (system_index >= system_count) {
        return;
    }
    constexpr int materialize_token_tile = generic_frame_materialize_token_tile<rank>();
    const int64_t token_start = tile_index * materialize_token_tile;
    const int token_count = static_cast<int>(
        length - token_start < materialize_token_tile
            ? length - token_start
            : materialize_token_tile);
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* b = system_tape + workspace_layout.b_offset;
    const float* lower = system_tape + workspace_layout.l_offset;
    float* frame = const_cast<float*>(system_tape) + frame_offset;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto [lower_tile, rhs_tile] = cusolverdx::shared_memory::slice<float, float>(
        shared_raw,
        alignof(float),
        rank * FrameTrsm::lda,
        alignof(float));
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lower_tile[row * FrameTrsm::lda + column] = lower[linear];
    }
    for (int linear = threadIdx.x; linear < materialize_token_tile * rank;
         linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        const int64_t token = token_start + row;
        rhs_tile[row * FrameTrsm::ldb + column] = 0.0f;
        if (row < token_count) {
            rhs_tile[row * FrameTrsm::ldb + column] = b[token * rank + column];
        }
    }
    __syncthreads();
    FrameTrsm().execute(lower_tile, FrameTrsm::lda, rhs_tile, FrameTrsm::ldb);
    __syncthreads();
    for (int linear = threadIdx.x; linear < token_count * rank;
         linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        frame[(token_start + row) * rank + column] =
            rhs_tile[row * FrameTrsm::ldb + column];
    }
}

template <typename scalar_t, int rank, bool direct = false>
__global__ __launch_bounds__(kThreads) void generic_cross_state_partials_kernel(
    const scalar_t* __restrict__ projected,
    float* __restrict__ tape,
    float* __restrict__ partial_cross,
    ForwardWorkspaceLayout workspace_layout,
    int64_t frame_offset,
    int64_t batch_count,
    int64_t length,
    int64_t heads,
    int64_t dim,
    int64_t head_dim,
    int64_t rhs_tiles,
    int64_t partial_tile_stride,
    int64_t token_group_start,
    int64_t token_group_count) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t rhs_tile = static_cast<int64_t>(blockIdx.y);
    const int64_t local_token_group = static_cast<int64_t>(blockIdx.z);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count || rhs_tile >= rhs_tiles ||
        local_token_group >= token_group_count) {
        return;
    }
    const int64_t batch = system_index / heads;
    const int64_t head = system_index - batch * heads;
    const int64_t rhs_start = rhs_tile * kRhsTile;
    const int64_t group_start =
        (token_group_start + local_token_group) *
        kCrossTilesPerBlock * kGenericTokenTile;
    const int64_t projected_width = heads * rank + dim;
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* frame = system_tape + frame_offset;

    using Cross = typename GenericFrameMathDx<rank>::Cross;
    constexpr size_t a_elements = cublasdx::cosize(Cross::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Cross::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Cross::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    __nv_bfloat16* a_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * a_elements, 16));
    __nv_bfloat16* b_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * b_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto cross_a = cublasdx::make_tensor(a_tile, Cross::get_layout_smem_a());
    auto cross_b = cublasdx::make_tensor(b_tile, Cross::get_layout_smem_b());
    auto cross_c = cublasdx::make_tensor(gemm_output, Cross::get_layout_smem_c());

    for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    __syncthreads();
    const int tiles = direct ? static_cast<int>((length + kGenericTokenTile - 1) / kGenericTokenTile) : kCrossTilesPerBlock;
    for (int tile = 0; tile < tiles; ++tile) {
        const int64_t token_start = group_start + tile * kGenericTokenTile;
        if (token_start >= length) {
            break;
        }
        for (int linear = threadIdx.x; linear < rank * kGenericTokenTile;
             linear += blockDim.x) {
            const int column = linear / rank;
            const int row = linear - column * rank;
            const int64_t token = token_start + column;
            cross_a(row, column) = __float2bfloat16_rn(
                token < length ? frame[token * rank + row] : 0.0f);
        }
        for (int linear = threadIdx.x; linear < kGenericTokenTile * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear - row * kRhsTile;
            const int64_t token = token_start + row;
            const int64_t feature = rhs_start + column;
            cross_b(row, column) = __float2bfloat16_rn(
                token < length && feature < head_dim
                    ? load_scalar(
                          projected,
                          (batch * length + token) * projected_width + heads * rank +
                              head * head_dim + feature)
                    : 0.0f);
        }
        __syncthreads();
        Cross().execute(1.0f, cross_a, cross_b, 1.0f, cross_c);
        __syncthreads();
    }
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        if constexpr (direct) {
            if (rhs_start + column < head_dim) {
                tape[system_index * workspace_layout.stride + workspace_layout.z_offset +
                     row * head_dim + rhs_start + column] = cross_c(row, column);
            }
        } else {
            partial_cross[((system_index * rhs_tiles + rhs_tile) * partial_tile_stride +
                           local_token_group) * rank * kRhsTile + linear] = cross_c(row, column);
        }
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void generic_cross_state_reduce_kernel(
    const float* __restrict__ partial_cross,
    float* __restrict__ tape,
    ForwardWorkspaceLayout workspace_layout,
    int64_t system_count,
    int64_t head_dim,
    int64_t rhs_tiles,
    int64_t partial_tile_stride,
    int64_t token_group_count,
    bool accumulate) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t rhs_tile = static_cast<int64_t>(blockIdx.y);
    if (system_index >= system_count || rhs_tile >= rhs_tiles) {
        return;
    }
    const int64_t rhs_start = rhs_tile * kRhsTile;
    float* system_tape = tape + system_index * workspace_layout.stride;
    float* compact_state = system_tape + workspace_layout.z_offset;
    const float* partial = partial_cross +
        (system_index * rhs_tiles + rhs_tile) * partial_tile_stride * rank * kRhsTile;
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        if (rhs_start + column < head_dim) {
            float value = accumulate
                ? compact_state[row * head_dim + rhs_start + column]
                : 0.0f;
            for (int64_t tile = 0; tile < token_group_count; ++tile) {
                value += partial[tile * rank * kRhsTile + linear];
            }
            compact_state[row * head_dim + rhs_start + column] = value;
        }
    }
}

template <int rank>
__device__ __forceinline__ float generic_factor_value(
    float coordinate,
    int row,
    int column) {
    if (row > column) {
        return coordinate;
    }
    if (row == column) {
        return softplus_one(coordinate);
    }
    return 0.0f;
}

template <int rank>
struct GenericCoreFactorMathDx {
    using Getrf = decltype(
        cusolverdx::Size<rank, rank>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::getrf_partial_pivot>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kThreads>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());
};

template <int rank>
struct GenericCoreSolveMathDx {
    using Getrs = decltype(
        cusolverdx::Size<rank, rank, kRhsTile>() +
        cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::getrs_partial_pivot>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::non_transposed>() +
        cusolverdx::Arrangement<
            cusolverdx::arrangement::row_major,
            cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() +
        cusolverdx::BlockDim<kRhsTile>() +
        cusolverdx::BatchesPerBlock<1>() +
        cusolverdx::SM<kCompiledSm>());
};

template <int rank, bool record_tape>
__global__ __launch_bounds__(kThreads) void generic_core_factor_kernel(
    const float* __restrict__ tape,
    const float* __restrict__ core_base_raw,
    const float* __restrict__ core_drive_weight,
    const float* __restrict__ valid_counts,
    int* __restrict__ pivots,
    ForwardWorkspaceLayout workspace_layout,
    int64_t batch_count,
    int64_t heads,
    int64_t head_dim,
    float inverse_length_sqrt) {
    using Core = typename GenericFrameMathDx<rank>::Core;
    using Getrf = typename GenericCoreFactorMathDx<rank>::Getrf;
    using MatrixScalar = typename Getrf::a_data_type;

    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count) {
        return;
    }
    const int64_t head = system_index - (system_index / heads) * heads;
    const int64_t batch = system_index / heads;
    if (valid_counts != nullptr) {
        inverse_length_sqrt = rsqrtf(valid_counts[batch]);
    }
    float* system_tape = const_cast<float*>(tape) + system_index * workspace_layout.stride;
    const float* compact_state = system_tape + workspace_layout.z_offset;
    float* lu = system_tape + workspace_layout.lu_offset;
    int* system_pivots = pivots + system_index * rank;
    float* coordinates = nullptr;
    if constexpr (record_tape) {
        coordinates = system_tape + workspace_layout.coordinates_offset;
    }

    __shared__ typename Getrf::status_type info;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    constexpr size_t core_a_elements = cublasdx::cosize(Core::get_layout_smem_a());
    constexpr size_t core_b_elements = cublasdx::cosize(Core::get_layout_smem_b());
    constexpr size_t core_c_elements = cublasdx::cosize(Core::get_layout_smem_c());
    static_assert(core_c_elements >= rank * rank);
    float* gemm_output = reinterpret_cast<float*>(shared_raw);
    auto* stage_raw = shared_raw + align_shared_offset(
        sizeof(float) * core_c_elements, 16);
    __nv_bfloat16* a_tile = reinterpret_cast<__nv_bfloat16*>(stage_raw);
    __nv_bfloat16* b_tile = a_tile + core_a_elements;
    auto* solver_raw = stage_raw;
    auto [matrix, factor_pivots] = cusolverdx::shared_memory::slice<MatrixScalar, int>(
        solver_raw,
        alignof(MatrixScalar), rank * Getrf::lda,
        alignof(int));
    auto core_a = cublasdx::make_tensor(a_tile, Core::get_layout_smem_a());
    auto core_b = cublasdx::make_tensor(b_tile, Core::get_layout_smem_b());
    auto core_c = cublasdx::make_tensor(gemm_output, Core::get_layout_smem_c());

    for (int linear = threadIdx.x; linear < core_c_elements; linear += blockDim.x) {
        gemm_output[linear] = 0.0f;
    }
    __syncthreads();
    for (int64_t feature_start = 0; core_drive_weight != nullptr && feature_start < head_dim;
         feature_start += kRhsTile) {
        for (int linear = threadIdx.x; linear < rank * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear - row * kRhsTile;
            const int64_t feature = feature_start + column;
            core_a(row, column) = __float2bfloat16_rn(
                feature < head_dim
                    ? compact_state[row * head_dim + feature]
                    : 0.0f);
        }
        for (int linear = threadIdx.x; linear < kRhsTile * rank;
             linear += blockDim.x) {
            const int row = linear / rank;
            const int column = linear - row * rank;
            const int64_t feature = feature_start + row;
            core_b(row, column) = __float2bfloat16_rn(
                feature < head_dim
                    ? core_drive_weight[(head * head_dim + feature) * rank + column]
                    : 0.0f);
        }
        __syncthreads();
        Core().execute(1.0f, core_a, core_b, 1.0f, core_c);
        __syncthreads();
    }
    // Once the BF16 core product is consumed, reuse solver storage for F and
    // the upper Omega coordinates; gemm_output becomes the assembled K.
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        const float coordinate =
            core_base_raw[(head * rank + row) * rank + column] +
            core_c(row, column) * inverse_length_sqrt;
        if constexpr (record_tape) {
            coordinates[linear] = coordinate;
        }
        matrix[row * Getrf::lda + column] = row >= column
            ? generic_factor_value<rank>(coordinate, row, column)
            : coordinate;
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * (rank + 1) / 2;
         linear += blockDim.x) {
        int remaining = linear;
        int row = 0;
        while (remaining > row) {
            remaining -= ++row;
        }
        const int column = remaining;
        float gram = 0.0f;
#pragma unroll
        for (int inner = 0; inner < rank; ++inner) {
            if (inner <= column) {
                gram = fmaf(
                    matrix[row * Getrf::lda + inner],
                    matrix[column * Getrf::lda + inner],
                    gram);
            }
        }
        const float upper = row == column
            ? 0.0f
            : matrix[column * Getrf::lda + row];
        gemm_output[row * rank + column] = gram - upper;
        gemm_output[column * rank + row] = gram + upper;
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        matrix[row * Getrf::lda + column] =
            gemm_output[linear] + (row == column ? 1.0f : 0.0f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        info = 0;
    }
    __syncthreads();
    Getrf().execute(matrix, factor_pivots, &info);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * rank; linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        lu[linear] = matrix[row * Getrf::lda + column];
    }
    for (int linear = threadIdx.x; linear < rank; linear += blockDim.x) {
        // GETRF and GETRS use the same one-based pivot contract.
        system_pivots[linear] = factor_pivots[linear];
    }
}


// Solve against identity once per head. This is the same compact correction
// map as M=2S-I, storing S to reuse the existing fused equilibrium tape/VJP.
template <int rank>
__global__ __launch_bounds__(kRhsTile) void static_core_map_kernel(
    float* tape, const int* pivots, ForwardWorkspaceLayout layout) {
    using Getrs = typename GenericCoreSolveMathDx<rank>::Getrs;
    const int64_t head = blockIdx.x;
    float* destination = tape + head * layout.stride + layout.lu_offset;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto [lu, rhs, pivot] = cusolverdx::shared_memory::slice<float, float, int>(
        shared_raw, 16u, rank * Getrs::lda, 16u, rank * Getrs::ldb, 16u, rank);
    for (int i = threadIdx.x; i < rank * rank; i += blockDim.x) lu[i] = destination[i];
    for (int i = threadIdx.x; i < rank; i += blockDim.x) pivot[i] = pivots[head * rank + i];
    __syncthreads();
    for (int start = 0; start < rank; start += kRhsTile) {
        for (int i = threadIdx.x; i < rank * kRhsTile; i += blockDim.x)
            rhs[i] = i / kRhsTile == start + i % kRhsTile ? 1.0f : 0.0f;
        __syncthreads();
        Getrs().execute(lu, Getrs::lda, pivot, rhs, Getrs::ldb);
        __syncthreads();
        for (int i = threadIdx.x; i < rank * kRhsTile; i += blockDim.x)
            if (start + i % kRhsTile < rank)
                destination[(i / kRhsTile) * rank + start + i % kRhsTile] = rhs[i];
        __syncthreads();
    }
}

template <int rank>
__global__ __launch_bounds__(kThreads) void static_core_apply_kernel(
    float* tape, ForwardWorkspaceLayout layout, int64_t heads, int64_t head_dim) {
    const int64_t system = blockIdx.x, start = int64_t(blockIdx.y) * kRhsTile;
    const float* map = tape + (system % heads) * layout.stride + layout.lu_offset;
    float* entry = tape + system * layout.stride;
    for (int i = threadIdx.x; i < rank * kRhsTile; i += blockDim.x) {
        const int row = i / kRhsTile, col = start + i % kRhsTile;
        if (col < head_dim)
            entry[layout.u_offset + row * head_dim + col] =
                apply_core_map_element<rank, false>(map, entry + layout.z_offset, row, col, head_dim);
    }
}

template <int rank>
__global__ __launch_bounds__(kRhsTile) void generic_solve_kernel(
    const float* __restrict__ tape,
    const int* __restrict__ pivots,
    ForwardWorkspaceLayout workspace_layout,
    int64_t batch_count,
    int64_t heads,
    int64_t head_dim,
    int core_mode) {
    const int64_t system_index = static_cast<int64_t>(blockIdx.x);
    const int64_t rhs_tile = static_cast<int64_t>(blockIdx.y);
    const int64_t system_count = batch_count * heads;
    if (system_index >= system_count) {
        return;
    }
    const int64_t rhs_start = rhs_tile * kRhsTile;
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* compact_state = system_tape + workspace_layout.z_offset;
    const int64_t core_index = core_mode == 1 ? system_index % heads : system_index;
    const float* lu = tape + core_index * workspace_layout.stride + workspace_layout.lu_offset;
    float* equilibrium = const_cast<float*>(system_tape) + workspace_layout.u_offset;
    const int* system_pivots = pivots + core_index * rank;

    using Getrs = typename GenericCoreSolveMathDx<rank>::Getrs;

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
            ? compact_state[row * head_dim + rhs_start + column]
            : 0.0f;
    }
    for (int linear = threadIdx.x; linear < rank; linear += blockDim.x) {
        pivot_tile[linear] = system_pivots[linear];
    }
    __syncthreads();
    Getrs().execute(lu_tile, Getrs::lda, pivot_tile, rhs, Getrs::ldb);
    __syncthreads();
    for (int linear = threadIdx.x; linear < rank * kRhsTile; linear += blockDim.x) {
        const int row = linear / kRhsTile;
        const int column = linear - row * kRhsTile;
        if (rhs_start + column < head_dim) {
            equilibrium[row * head_dim + rhs_start + column] = rhs[linear];
        }
    }
}

template <typename scalar_t, int rank>
__global__ __launch_bounds__(kThreads) void generic_output_kernel(
    const scalar_t* __restrict__ projected,
    const float* __restrict__ tape,
    const float* __restrict__ eta_raw,
    scalar_t* __restrict__ output,
    ForwardWorkspaceLayout workspace_layout,
    int64_t frame_offset,
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
    const int64_t token_start = tile_index * kGenericTokenTile;
    const int token_count = static_cast<int>(
        length - token_start < kGenericTokenTile ? length - token_start : kGenericTokenTile);
    const int64_t projected_width = heads * rank + dim;
    const float* system_tape = tape + system_index * workspace_layout.stride;
    const float* frame = system_tape + frame_offset;
    const float* compact_state = system_tape + workspace_layout.z_offset;
    const float* equilibrium = system_tape + workspace_layout.u_offset;
    const float eta = bounded_complement(eta_raw[head]);

    using Readout = typename GenericFrameMathDx<rank>::Readout;
    constexpr size_t a_elements = cublasdx::cosize(Readout::get_layout_smem_a());
    constexpr size_t b_elements = cublasdx::cosize(Readout::get_layout_smem_b());
    constexpr size_t c_elements = cublasdx::cosize(Readout::get_layout_smem_c());
    extern __shared__ __align__(16) unsigned char shared_raw[];
    SharedCursor shared(shared_raw);
    __nv_bfloat16* a_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * a_elements, 16));
    __nv_bfloat16* b_tile = reinterpret_cast<__nv_bfloat16*>(
        shared.take_bytes(sizeof(__nv_bfloat16) * b_elements, 16));
    float* gemm_output = reinterpret_cast<float*>(
        shared.take_bytes(sizeof(float) * c_elements, 16));
    auto readout_a = cublasdx::make_tensor(a_tile, Readout::get_layout_smem_a());
    auto readout_b = cublasdx::make_tensor(b_tile, Readout::get_layout_smem_b());
    auto readout_c = cublasdx::make_tensor(gemm_output, Readout::get_layout_smem_c());

    for (int linear = threadIdx.x; linear < kGenericTokenTile * rank;
         linear += blockDim.x) {
        const int row = linear / rank;
        const int column = linear - row * rank;
        const int64_t token = token_start + row;
        readout_a(row, column) = __float2bfloat16_rn(
            row < token_count ? frame[token * rank + column] : 0.0f);
    }
    __syncthreads();
    for (int64_t rhs_start = 0; rhs_start < head_dim; rhs_start += kRhsTile) {
        for (int linear = threadIdx.x; linear < c_elements; linear += blockDim.x) {
            gemm_output[linear] = 0.0f;
        }
        for (int linear = threadIdx.x; linear < rank * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear - row * kRhsTile;
            const int64_t feature = rhs_start + column;
            const float compact_mix = feature < head_dim
                ? compact_readout_coefficient(compact_state[row * head_dim + feature],
                    equilibrium[row * head_dim + feature], eta,
                    workspace_layout.u_offset == workspace_layout.z_offset)
                : 0.0f;
            readout_b(row, column) = __float2bfloat16_rn(compact_mix);
        }
        __syncthreads();
        Readout().execute(1.0f, readout_a, readout_b, 0.0f, readout_c);
        __syncthreads();
        for (int linear = threadIdx.x; linear < token_count * kRhsTile;
             linear += blockDim.x) {
            const int row = linear / kRhsTile;
            const int column = linear - row * kRhsTile;
            const int64_t feature = rhs_start + column;
            if (feature < head_dim) {
                const int64_t token = token_start + row;
                const float content = load_scalar(
                    projected,
                    (batch * length + token) * projected_width + heads * rank +
                        head * head_dim + feature);
                store_scalar(
                    output,
                    (batch * length + token) * dim + head * head_dim + feature,
                    eta * content + readout_c(row, column));
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t, int rank, bool record_tape>
void configure_forward_kernel_attributes(int device) {
    static std::mutex mutex;
    static std::unordered_set<int> configured_devices;
    std::lock_guard<std::mutex> lock(mutex);
    if (configured_devices.find(device) != configured_devices.end()) {
        return;
    }

    constexpr size_t frame_shared_bytes = generic_frame_shared_bytes<rank>();
    constexpr size_t gram_partial_shared_bytes =
        generic_frame_gram_partial_shared_bytes<rank>();
    constexpr size_t factor_partials_shared_bytes =
        generic_frame_factor_partials_shared_bytes<rank>();
    constexpr size_t frame_materialize_shared_bytes =
        generic_frame_materialize_shared_bytes<rank>();
    constexpr size_t cross_shared_bytes = generic_cross_shared_bytes<rank>();
    constexpr size_t core_factor_shared_bytes = generic_core_factor_shared_bytes<rank>();
    using Getrs = typename GenericCoreSolveMathDx<rank>::Getrs;
    constexpr size_t solve_shared_bytes = Getrs::shared_memory_size;
    constexpr size_t output_shared_bytes = generic_output_shared_bytes<rank>();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_factor_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(frame_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_gram_partials_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(gram_partial_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_factor_partials_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(factor_partials_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_frame_materialize_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(frame_materialize_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_cross_state_partials_kernel<scalar_t, rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(cross_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_core_factor_kernel<rank, record_tape>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(core_factor_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_solve_kernel<rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(solve_shared_bytes)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        generic_output_kernel<scalar_t, rank>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(output_shared_bytes)));
    configured_devices.insert(device);
}

template <typename scalar_t, int rank, bool record_tape>
ForwardResult launch_generic_forward(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts,
    FastPathShape shape) {
    auto output = at::empty({shape.batch, shape.length, shape.dim}, projected.options());
    const int64_t system_count = shape.batch * shape.heads;
    const auto tape_layout = training_tape_layout(shape);
    const auto workspace_layout = record_tape
        ? forward_workspace_layout(tape_layout)
        : inference_workspace_layout(shape);
    const auto workspace_options = projected.options().dtype(at::kFloat);
    auto workspace = at::empty({system_count, workspace_layout.stride}, workspace_options);
    auto pivot_workspace = at::empty({system_count, shape.core_mode == 2 ? 0 : rank}, projected.options().dtype(at::kInt));
    at::Tensor tape;
    at::Tensor pivots;
    int64_t frame_offset = workspace_layout.b_offset;
    if constexpr (record_tape) {
        tape = workspace;
        pivots = pivot_workspace;
        frame_offset = tape_layout.p_offset;
    }

    c10::cuda::CUDAGuard guard(projected.device());
    configure_forward_kernel_attributes<scalar_t, rank, record_tape>(
        projected.get_device());
    const auto stream = at::cuda::getCurrentCUDAStream(projected.get_device()).stream();
    if (shape.length >= 4096 && system_count < 128) {
        const int64_t tiles = (shape.length + kScaleTokenTile - 1) / kScaleTokenTile;
        auto partials = at::empty({system_count, tiles}, workspace_options);
        const dim3 grid(system_count, tiles);
        relation_scale_partials_kernel<scalar_t, rank><<<grid, kThreads, 0, stream>>>(
            projected.data_ptr<scalar_t>(), valid_counts.has_value() ? valid_counts->data_ptr<float>() : nullptr,
            partials.data_ptr<float>(), shape.length, shape.heads, shape.dim, tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        relation_scale_reduce_kernel<<<system_count, kThreads, 0, stream>>>(
            partials.data_ptr<float>(), workspace.data_ptr<float>(), workspace_layout, tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        relation_prepare_tiled_kernel<scalar_t, rank><<<grid, kThreads, 0, stream>>>(
            projected.data_ptr<scalar_t>(), valid_counts.has_value() ? valid_counts->data_ptr<float>() : nullptr,
            workspace.data_ptr<float>(), workspace_layout, shape.length, shape.heads, shape.dim);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
    generic_frame_kernel<scalar_t, rank><<<system_count, kThreads, 0, stream>>>(
        projected.data_ptr<scalar_t>(),
        valid_counts.has_value() ? valid_counts->data_ptr<float>() : nullptr,
        workspace.data_ptr<float>(),
        workspace_layout,
        shape.batch,
        shape.length,
        shape.heads,
        shape.dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    }

    const int64_t token_tiles =
        (shape.length + kGenericTokenTile - 1) / kGenericTokenTile;
    if (token_tiles >= kParallelGramMinimumTokenTiles) {
        constexpr int kPackedLowerElements = rank * (rank + 1) / 2;
        at::Tensor gram_partials;
        float* gram_partial_data = nullptr;
        int64_t gram_partial_system_stride =
            token_tiles * kPackedLowerElements;
        // B is still live here; the future P region now aliases B.
        gram_partials = at::empty(
            {system_count, token_tiles, kPackedLowerElements}, workspace_options);
        gram_partial_data = gram_partials.data_ptr<float>();
        const int64_t gram_blocks = system_count * token_tiles;
        constexpr size_t gram_partial_shared_bytes =
            generic_frame_gram_partial_shared_bytes<rank>();
        generic_frame_gram_partials_kernel<rank><<<
            static_cast<unsigned int>(gram_blocks),
            kThreads,
            gram_partial_shared_bytes,
            stream>>>(
            workspace.data_ptr<float>(),
            workspace_layout,
            gram_partial_data,
            gram_partial_system_stride,
            system_count,
            shape.length,
            token_tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        at::Tensor grouped_partials;
        int64_t factor_tiles = token_tiles;
        if (token_tiles >= 128 && system_count < 128) {
            factor_tiles = (token_tiles + 31) / 32;
            grouped_partials = at::empty({system_count, factor_tiles, kPackedLowerElements}, workspace_options);
            gram_group_reduce_kernel<rank><<<dim3(system_count, factor_tiles), kThreads, 0, stream>>>(
                gram_partial_data, grouped_partials.data_ptr<float>(), token_tiles, factor_tiles);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            gram_partial_data = grouped_partials.data_ptr<float>();
            gram_partial_system_stride = factor_tiles * kPackedLowerElements;
        }
        constexpr size_t factor_partials_shared_bytes =
            generic_frame_factor_partials_shared_bytes<rank>();
        generic_frame_factor_partials_kernel<rank><<<
            system_count, kThreads, factor_partials_shared_bytes, stream>>>(
            workspace.data_ptr<float>(),
            workspace_layout,
            gram_partial_data,
            gram_partial_system_stride,
            system_count,
            factor_tiles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        constexpr size_t frame_shared_bytes = generic_frame_shared_bytes<rank>();
        generic_frame_factor_kernel<rank><<<
            system_count, kThreads, frame_shared_bytes, stream>>>(
            workspace.data_ptr<float>(),
            workspace_layout,
            system_count,
            shape.length);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    constexpr int materialize_token_tile = generic_frame_materialize_token_tile<rank>();
    const int64_t frame_token_tiles =
        (shape.length + materialize_token_tile - 1) / materialize_token_tile;
    const dim3 frame_grid(
        static_cast<unsigned int>(system_count), static_cast<unsigned int>(frame_token_tiles));
    constexpr size_t frame_materialize_shared_bytes = generic_frame_materialize_shared_bytes<rank>();
    generic_frame_materialize_kernel<rank><<<
        frame_grid, kThreads, frame_materialize_shared_bytes, stream>>>(
        workspace.data_ptr<float>(),
        workspace_layout,
        frame_offset,
        system_count,
        shape.length);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t rhs_tiles = (shape.head_dim + kRhsTile - 1) / kRhsTile;
    const dim3 rhs_grid(
        static_cast<unsigned int>(system_count), static_cast<unsigned int>(rhs_tiles));
    constexpr size_t cross_shared_bytes = generic_cross_shared_bytes<rank>();
    const int64_t cross_groups =
        (token_tiles + kCrossTilesPerBlock - 1) / kCrossTilesPerBlock;
    const int64_t cross_tile_capacity = cross_groups < kCrossTokenChunk
        ? cross_groups
        : kCrossTokenChunk;
    if (shape.length <= 256) {
        // At most eight tiles per RHS CTA: no global partial or reduction launch.
        generic_cross_state_partials_kernel<scalar_t, rank, true><<<
            rhs_grid, kThreads, cross_shared_bytes, stream>>>(
            projected.data_ptr<scalar_t>(), workspace.data_ptr<float>(), nullptr,
            workspace_layout, frame_offset, shape.batch, shape.length, shape.heads,
            shape.dim, shape.head_dim, rhs_tiles, 1, 0, 1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        auto partial_cross = at::empty(
            {system_count, rhs_tiles, cross_tile_capacity, rank, kRhsTile},
            workspace_options);
        for (int64_t token_group_start = 0; token_group_start < cross_groups;
             token_group_start += kCrossTokenChunk) {
            const int64_t token_group_count =
                cross_groups - token_group_start < kCrossTokenChunk
                ? cross_groups - token_group_start
                : kCrossTokenChunk;
            const dim3 cross_grid(
                static_cast<unsigned int>(system_count),
                static_cast<unsigned int>(rhs_tiles),
                static_cast<unsigned int>(token_group_count));
            generic_cross_state_partials_kernel<scalar_t, rank><<<
                cross_grid, kThreads, cross_shared_bytes, stream>>>(
                projected.data_ptr<scalar_t>(),
                workspace.data_ptr<float>(),
                partial_cross.data_ptr<float>(),
                workspace_layout,
                frame_offset,
                shape.batch,
                shape.length,
                shape.heads,
                shape.dim,
                shape.head_dim,
                rhs_tiles,
                cross_tile_capacity,
                token_group_start,
                token_group_count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            generic_cross_state_reduce_kernel<rank><<<rhs_grid, kThreads, 0, stream>>>(
                partial_cross.data_ptr<float>(),
                workspace.data_ptr<float>(),
                workspace_layout,
                system_count,
                shape.head_dim,
                rhs_tiles,
                cross_tile_capacity,
                token_group_count,
                token_group_start != 0);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    if (shape.core_mode != 2) {
    const int64_t core_systems = shape.core_mode == 1 ? shape.heads : system_count;
    constexpr size_t core_factor_shared_bytes = generic_core_factor_shared_bytes<rank>();
    generic_core_factor_kernel<rank, record_tape><<<
        core_systems, kThreads, core_factor_shared_bytes, stream>>>(
        workspace.data_ptr<float>(),
        core_base_raw.data_ptr<float>(),
        shape.core_mode == 0 ? core_drive_weight.data_ptr<float>() : nullptr,
        valid_counts.has_value() ? valid_counts->data_ptr<float>() : nullptr,
        pivot_workspace.data_ptr<int>(),
        workspace_layout,
        shape.core_mode == 1 ? 1 : shape.batch,
        shape.heads,
        shape.head_dim,
        1.0f / std::sqrt(static_cast<float>(shape.length)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    using Getrs = typename GenericCoreSolveMathDx<rank>::Getrs;
    constexpr size_t solve_shared_bytes = Getrs::shared_memory_size;
    if (use_shared_core_map(shape)) {
        static_core_map_kernel<rank><<<shape.heads, kRhsTile, solve_shared_bytes, stream>>>(
            workspace.data_ptr<float>(), pivot_workspace.data_ptr<int>(), workspace_layout);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        static_core_apply_kernel<rank><<<rhs_grid, kThreads, 0, stream>>>(
            workspace.data_ptr<float>(), workspace_layout, shape.heads, shape.head_dim);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
    generic_solve_kernel<rank><<<rhs_grid, kRhsTile, solve_shared_bytes, stream>>>(
        workspace.data_ptr<float>(),
        pivot_workspace.data_ptr<int>(),
        workspace_layout,
        shape.batch,
        shape.heads,
        shape.head_dim, shape.core_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    }

    const dim3 output_grid(
        static_cast<unsigned int>(system_count), static_cast<unsigned int>(token_tiles));
    constexpr size_t output_shared_bytes = generic_output_shared_bytes<rank>();
    generic_output_kernel<scalar_t, rank><<<
        output_grid, kThreads, output_shared_bytes, stream>>>(
        projected.data_ptr<scalar_t>(),
        workspace.data_ptr<float>(),
        eta_raw.data_ptr<float>(),
        output.data_ptr<scalar_t>(),
        workspace_layout,
        frame_offset,
        shape.batch,
        shape.length,
        shape.heads,
        shape.dim,
        shape.head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, tape, pivots};
}

template <typename scalar_t, bool record_tape>
ForwardResult dispatch_expanded_forward(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts,
    FastPathShape shape) {
    switch (shape.rank) {
        case 16:
            return launch_generic_forward<scalar_t, 16, record_tape>(
                projected, core_base_raw, core_drive_weight, eta_raw,
                valid_counts, shape);
        case 32:
            return launch_generic_forward<scalar_t, 32, record_tape>(
                projected, core_base_raw, core_drive_weight, eta_raw,
                valid_counts, shape);
        case 48:
            return launch_generic_forward<scalar_t, 48, record_tape>(
                projected, core_base_raw, core_drive_weight, eta_raw,
                valid_counts, shape);
        case 64:
            return launch_generic_forward<scalar_t, 64, record_tape>(
                projected, core_base_raw, core_drive_weight, eta_raw,
                valid_counts, shape);
        default:
            TORCH_CHECK(false, "unreachable supported rank");
    }
}

}  // namespace

at::Tensor forward_inference_cuda(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts) {
    const auto shape = validate_fast_inputs(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        valid_counts);
    c10::cuda::CUDAGuard guard(projected.device());
    (void)supported_sm();
    return dispatch_expanded_forward<at::BFloat16, false>(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        valid_counts,
        shape).output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_train_cuda(
    const at::Tensor& projected,
    const at::Tensor& core_base_raw,
    const at::Tensor& core_drive_weight,
    const at::Tensor& eta_raw,
    const c10::optional<at::Tensor>& valid_counts) {
    const auto shape = validate_fast_inputs(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        valid_counts);
    c10::cuda::CUDAGuard guard(projected.device());
    (void)supported_sm();
    ForwardResult result = dispatch_expanded_forward<at::BFloat16, true>(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        valid_counts,
        shape);
    return {result.output, result.tape, result.pivots};
}

}  // namespace lsso_equilibrium
