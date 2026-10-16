#include "common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <cusolverdx.hpp>

namespace memsolve_equilibrium {
namespace {

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

// Compact-only boundaries reuse MathDx LU without a token/frame tape.
template<int rank>
__global__ __launch_bounds__(kThreads) void compact_lu_kernel(
    const float* input, float* output, int* pivots, int* status) {
    using Solver = typename GenericCoreFactorMathDx<rank>::Getrf;
    extern __shared__ __align__(16) unsigned char storage[];
    float* matrix = reinterpret_cast<float*>(storage);
    __shared__ int pivot[rank];
    __shared__ int info;
    const int64_t s = blockIdx.x;
    for(int i=threadIdx.x; i<rank*rank; i+=blockDim.x)
        matrix[(i/rank)*Solver::lda+i%rank] = input[s*rank*rank+i];
    __syncthreads();
    Solver().execute(matrix, pivot, &info);
    __syncthreads();
    for(int i=threadIdx.x; i<rank*rank; i+=blockDim.x)
        output[s*rank*rank+i] = matrix[(i/rank)*Solver::lda+i%rank];
    for(int i=threadIdx.x; i<rank; i+=blockDim.x) pivots[s*rank+i]=pivot[i];
    if(threadIdx.x==0) status[s]=info;
}

template<int rank, bool transpose>
__global__ __launch_bounds__(kRhsTile) void compact_getrs_kernel(
    const float* factors, const int* pivots, const float* input,
    float* output, int64_t columns) {
    using Solver = decltype(
        cusolverdx::Size<rank,rank,kRhsTile>() +
        cusolverdx::Precision<float>() + cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::getrs_partial_pivot>() +
        cusolverdx::TransposeMode<transpose ? cusolverdx::transpose::transposed : cusolverdx::transpose::non_transposed>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major,cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() + cusolverdx::BlockDim<kRhsTile>() +
        cusolverdx::BatchesPerBlock<1>() + cusolverdx::SM<kCompiledSm>());
    __shared__ float lu[rank*Solver::lda];
    __shared__ float rhs[rank*Solver::ldb];
    __shared__ int pivot[rank];
    const int64_t s=blockIdx.x, start=blockIdx.y*kRhsTile;
    for(int i=threadIdx.x; i<rank*rank; i+=blockDim.x)
        lu[(i/rank)*Solver::lda+i%rank]=factors[s*rank*rank+i];
    for(int i=threadIdx.x; i<rank*kRhsTile; i+=blockDim.x) {
        const int row=i/kRhsTile, col=i%kRhsTile;
        rhs[row*Solver::ldb+col]= start+col<columns ? input[(s*rank+row)*columns+start+col] : 0.0f;
    }
    for(int i=threadIdx.x; i<rank; i+=blockDim.x) pivot[i]=pivots[s*rank+i];
    __syncthreads();
    Solver().execute(lu,Solver::lda,pivot,rhs,Solver::ldb);
    __syncthreads();
    for(int i=threadIdx.x; i<rank*kRhsTile; i+=blockDim.x) {
        const int row=i/kRhsTile, col=i%kRhsTile;
        if(start+col<columns) output[(s*rank+row)*columns+start+col]=rhs[row*Solver::ldb+col];
    }
}

template<int rank>
void launch_compact_lu(const at::Tensor& a, at::Tensor& lu, at::Tensor& pivot, at::Tensor& info) {
    using Solver=typename GenericCoreFactorMathDx<rank>::Getrf;
    constexpr size_t shared=Solver::shared_memory_size;
    compact_lu_kernel<rank><<<a.numel()/(rank*rank),kThreads,shared,at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(),lu.data_ptr<float>(),pivot.data_ptr<int>(),info.data_ptr<int>());
}

template<int rank>
void launch_compact_getrs(const at::Tensor& lu,const at::Tensor& pivot,const at::Tensor& rhs,at::Tensor& out,bool transpose) {
    const int64_t cols=rhs.size(-1);
    dim3 grid(lu.numel()/(rank*rank),(cols+kRhsTile-1)/kRhsTile);
    if(transpose)
        compact_getrs_kernel<rank,true><<<grid,kRhsTile,0,at::cuda::getCurrentCUDAStream()>>>(lu.data_ptr<float>(),pivot.data_ptr<int>(),rhs.data_ptr<float>(),out.data_ptr<float>(),cols);
    else
        compact_getrs_kernel<rank,false><<<grid,kRhsTile,0,at::cuda::getCurrentCUDAStream()>>>(lu.data_ptr<float>(),pivot.data_ptr<int>(),rhs.data_ptr<float>(),out.data_ptr<float>(),cols);
}


// POTRF and identity-panel TRSM share one factorization per system.
// The factor and implicit identity stay inside this CTA throughout.
template<int rank>
__global__ __launch_bounds__(128) void compact_inverse_kernel(
    const float* gram, float* output) {
    constexpr int columns = rank;
    using Factor = decltype(cusolverdx::Size<rank,rank>() + cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() + cusolverdx::Function<cusolverdx::function::potrf>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major>() +
        cusolverdx::LeadingDimension<rank>() + cusolverdx::Block() + cusolverdx::BlockDim<128>() +
        cusolverdx::BatchesPerBlock<1>() + cusolverdx::SM<kCompiledSm>());
    using First = decltype(cusolverdx::Size<rank,kRhsTile>() + cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() + cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::left>() + cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::Diag<cusolverdx::diag::non_unit>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major,cusolverdx::arrangement::row_major>() +
        cusolverdx::LeadingDimension<rank,kRhsTile>() + cusolverdx::Block() + cusolverdx::BlockDim<128>() +
        cusolverdx::BatchesPerBlock<1>() + cusolverdx::SM<kCompiledSm>());
    __shared__ float a[rank*rank];
    __shared__ float b[rank*kRhsTile];
    __shared__ int info;
    const int64_t s=blockIdx.x;
    for(int i=threadIdx.x;i<rank*rank;i+=blockDim.x)
        a[i]=gram[s*rank*rank+i]+(i/rank==i%rank?1.f:0.f);
    __syncthreads();
    Factor().execute(a,&info);
    __syncthreads();
    for(int64_t start=0;start<columns;start+=kRhsTile) {
        for(int i=threadIdx.x;i<rank*kRhsTile;i+=blockDim.x) {
            const int row=i/kRhsTile;const int64_t col=start+i%kRhsTile;
            b[i]=(row==col && col<columns)?1.f:0.f;
        }
        __syncthreads();
        First().execute(a,b);
        __syncthreads();
        for(int i=threadIdx.x;i<rank*kRhsTile;i+=blockDim.x) {
            const int row=i/kRhsTile;const int64_t col=start+i%kRhsTile;
            if(col<columns)output[(s*rank+row)*columns+col]=b[i];
        }
        __syncthreads();
    }
}

}  // namespace

static void validate_compact_matrix(const at::Tensor& a) {
    TORCH_CHECK(a.is_cuda() && a.scalar_type()==at::kFloat && a.is_contiguous(),
        "compact matrix must be contiguous CUDA FP32");
    TORCH_CHECK(a.dim()>=2 && a.size(-1)==a.size(-2) && a.numel()>0,
        "compact matrix must contain nonempty square systems");
    const auto r=a.size(-1);
    TORCH_CHECK(r==16 || r==32 || r==48 || r==64,"unsupported compact rank");
    TORCH_CHECK(!at::GradMode::is_enabled() || !a.requires_grad(),
        "compact native operations require an explicit autograd owner");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> compact_lu_cuda(const at::Tensor& a) {
    validate_compact_matrix(a);
    c10::cuda::CUDAGuard guard(a.device());
    (void)supported_sm();
    auto lu=at::empty_like(a);
    auto shape=a.sizes().vec(); shape.pop_back();
    auto pivot=at::empty(shape,a.options().dtype(at::kInt));
    shape.pop_back();
    auto info=at::empty(shape,a.options().dtype(at::kInt));
    switch(a.size(-1)) {
#define MEMSOLVE_LU_CASE(R) case R: launch_compact_lu<R>(a,lu,pivot,info); break;
        MEMSOLVE_LU_CASE(16) MEMSOLVE_LU_CASE(32) MEMSOLVE_LU_CASE(48) MEMSOLVE_LU_CASE(64)
#undef MEMSOLVE_LU_CASE
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {lu,pivot,info};
}

at::Tensor compact_getrs_cuda(const at::Tensor& lu,const at::Tensor& pivot,const at::Tensor& rhs,bool transpose) {
    validate_compact_matrix(lu);
    TORCH_CHECK(rhs.is_cuda() && rhs.device()==lu.device() && rhs.scalar_type()==at::kFloat && rhs.is_contiguous(),
        "compact rhs must be contiguous CUDA FP32 on the factor device");
    TORCH_CHECK(!at::GradMode::is_enabled() || !rhs.requires_grad(),
        "compact native operations require an explicit autograd owner");
    TORCH_CHECK(rhs.dim()==lu.dim() && rhs.size(-1)>0,"invalid compact rhs dimensions");
    for(int i=0;i<lu.dim()-1;++i) TORCH_CHECK(rhs.size(i)==lu.size(i),"compact rhs batch/rank mismatch");
    auto shape=lu.sizes().vec();shape.pop_back();
    TORCH_CHECK(pivot.is_cuda() && pivot.device()==lu.device() && pivot.scalar_type()==at::kInt && pivot.is_contiguous() && pivot.sizes().vec()==shape,
        "compact pivots must be contiguous CUDA int32 with matching batch/rank");
    c10::cuda::CUDAGuard guard(lu.device());
    (void)supported_sm();
    auto output=at::empty_like(rhs);
    switch(lu.size(-1)) {
#define MEMSOLVE_GETRS_CASE(R) case R: launch_compact_getrs<R>(lu,pivot,rhs,output,transpose); break;
        MEMSOLVE_GETRS_CASE(16) MEMSOLVE_GETRS_CASE(32) MEMSOLVE_GETRS_CASE(48) MEMSOLVE_GETRS_CASE(64)
#undef MEMSOLVE_GETRS_CASE
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


// Factor and invert in one CTA. Identity RHS panels and the Cholesky
// factor never leave shared memory; only the reusable inverse is stored.
at::Tensor compact_inverse_cuda(const at::Tensor& gram) {
    validate_compact_matrix(gram);
    c10::cuda::CUDAGuard guard(gram.device()); (void)supported_sm();
    auto inverse=at::empty_like(gram);
    const auto rank=gram.size(-1), systems=gram.numel()/(rank*rank);
    switch(rank) {
#define MEMSOLVE_INVERSE_CASE(R) case R: compact_inverse_kernel<R><<<systems,128,0,at::cuda::getCurrentCUDAStream()>>>(gram.data_ptr<float>(),inverse.data_ptr<float>()); break;
        MEMSOLVE_INVERSE_CASE(16) MEMSOLVE_INVERSE_CASE(32) MEMSOLVE_INVERSE_CASE(48) MEMSOLVE_INVERSE_CASE(64)
#undef MEMSOLVE_INVERSE_CASE
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return inverse;
}

}  // namespace memsolve_equilibrium
