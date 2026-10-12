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


// RHS panels follow NVIDIA's cuSOLVERDx trsm_batched_block example.
// The factor may have either dense matrix layout, as returned by POTRF.
template<int rank, bool transpose>
__global__ __launch_bounds__(128) void compact_trsm_kernel(
    const float* factor, const float* rhs, float* output, int64_t columns,
    int64_t ar, int64_t ac, int64_t br, int64_t bc) {
    using Solver = decltype(
        cusolverdx::Size<rank, kRhsTile>() + cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::left>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::Diag<cusolverdx::diag::non_unit>() +
        cusolverdx::TransposeMode<transpose ? cusolverdx::transpose::transposed : cusolverdx::transpose::non_transposed>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major,cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() + cusolverdx::BlockDim<128>() +
        cusolverdx::BatchesPerBlock<1>() + cusolverdx::SM<kCompiledSm>());
    __shared__ float a[rank * Solver::lda];
    __shared__ float b[rank * Solver::ldb];
    int64_t s=blockIdx.x, panel=blockIdx.y*kRhsTile;
    for(int i=threadIdx.x;i<rank*rank;i+=blockDim.x)
        a[(i/rank)*Solver::lda+i%rank]=factor[s*rank*rank+(i/rank)*ar+(i%rank)*ac];
    for(int i=threadIdx.x;i<rank*kRhsTile;i+=blockDim.x) {
        int64_t row=i/kRhsTile,col=panel+i%kRhsTile;
        b[row*Solver::ldb+i%kRhsTile]=col<columns ? rhs[s*rank*columns+row*br+col*bc] : 0.f;
    }
    __syncthreads();
    Solver().execute(a,b);
    __syncthreads();
    for(int i=threadIdx.x;i<rank*kRhsTile;i+=blockDim.x) {
        int64_t row=i/kRhsTile,col=panel+i%kRhsTile;
        if(col<columns)output[(s*rank+row)*columns+col]=b[row*Solver::ldb+i%kRhsTile];
    }
}

// Fuse L^-1 and L^-T while the factor and RHS remain in shared memory.
template<int rank>
__global__ __launch_bounds__(128) void compact_potrs_kernel(
    const float* factor, const float* rhs, float* output, int64_t columns,
    int64_t ar, int64_t ac, int64_t br, int64_t bc) {
    using Solver = decltype(
        cusolverdx::Size<rank, rank, kRhsTile>() + cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::potrs>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major,cusolverdx::arrangement::row_major>() +
        cusolverdx::Block() + cusolverdx::BlockDim<128>() +
        cusolverdx::BatchesPerBlock<1>() + cusolverdx::SM<kCompiledSm>());
    __shared__ float a[rank * Solver::lda];
    __shared__ float b[rank * Solver::ldb];
    int64_t s=blockIdx.x, panel=blockIdx.y*kRhsTile;
    for(int i=threadIdx.x;i<rank*rank;i+=blockDim.x)
        a[(i/rank)*Solver::lda+i%rank]=factor[s*rank*rank+(i/rank)*ar+(i%rank)*ac];
    for(int i=threadIdx.x;i<rank*kRhsTile;i+=blockDim.x) {
        int64_t row=i/kRhsTile,col=panel+i%kRhsTile;
        b[row*Solver::ldb+i%kRhsTile]=col<columns ? rhs[s*rank*columns+row*br+col*bc] : 0.f;
    }
    __syncthreads();
    Solver().execute(a,b);
    __syncthreads();
    for(int i=threadIdx.x;i<rank*kRhsTile;i+=blockDim.x) {
        int64_t row=i/kRhsTile,col=panel+i%kRhsTile;
        if(col<columns)output[(s*rank+row)*columns+col]=b[row*Solver::ldb+i%kRhsTile];
    }
}

template<int rank>
void launch_compact_trsm(const at::Tensor& a,const at::Tensor& rhs,at::Tensor& out,bool transpose,bool paired) {
    const int64_t cols=rhs.size(-1);
    dim3 grid(a.numel()/(rank*rank),(cols+kRhsTile-1)/kRhsTile);
#define MEMSOLVE_TRSM_LAUNCH(T) compact_trsm_kernel<rank,T><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(),rhs.data_ptr<float>(),out.data_ptr<float>(),cols,a.stride(-2),a.stride(-1),rhs.stride(-2),rhs.stride(-1))
    if(paired) {
        compact_potrs_kernel<rank><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(),rhs.data_ptr<float>(),out.data_ptr<float>(),cols,a.stride(-2),a.stride(-1),rhs.stride(-2),rhs.stride(-1));
    } else if(transpose){MEMSOLVE_TRSM_LAUNCH(true);}else{MEMSOLVE_TRSM_LAUNCH(false);}
#undef MEMSOLVE_TRSM_LAUNCH
}

// POTRF and the first RHS solve share one factorization per system.
// RHS panels loop inside the CTA: never refactor once per panel.
template<int rank, bool complete>
__global__ __launch_bounds__(128) void compact_factor_solve_kernel(
    const float* gram,const float* rhs,float* factor,float* output,int64_t columns) {
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
    using Full = decltype(cusolverdx::Size<rank,rank,kRhsTile>() + cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() + cusolverdx::Function<cusolverdx::function::potrs>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
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
    for(int i=threadIdx.x;i<rank*rank;i+=blockDim.x)
        factor[s*rank*rank+i]=i/rank>=i%rank?a[i]:0.f;
    for(int64_t start=0;start<columns;start+=kRhsTile) {
        for(int i=threadIdx.x;i<rank*kRhsTile;i+=blockDim.x) {
            const int row=i/kRhsTile;const int64_t col=start+i%kRhsTile;
            b[i]=col<columns?rhs[(s*rank+row)*columns+col]:0.f;
        }
        __syncthreads();
        if constexpr(complete) Full().execute(a,b); else First().execute(a,b);
        __syncthreads();
        for(int i=threadIdx.x;i<rank*kRhsTile;i+=blockDim.x) {
            const int row=i/kRhsTile;const int64_t col=start+i%kRhsTile;
            if(col<columns)output[(s*rank+row)*columns+col]=b[i];
        }
        __syncthreads();
    }
}

template<int rank>
void launch_compact_factor_solve(const at::Tensor& gram,const at::Tensor& rhs,at::Tensor& factor,at::Tensor& state,bool complete) {
#define MEMSOLVE_FACTOR_LAUNCH(C) compact_factor_solve_kernel<rank,C><<<gram.numel()/(rank*rank),128,0,at::cuda::getCurrentCUDAStream()>>>(gram.data_ptr<float>(),rhs.data_ptr<float>(),factor.data_ptr<float>(),state.data_ptr<float>(),rhs.size(-1))
    if(complete){MEMSOLVE_FACTOR_LAUNCH(true);}else{MEMSOLVE_FACTOR_LAUNCH(false);}
#undef MEMSOLVE_FACTOR_LAUNCH
}

// Cholesky VJP: L^-T sym_lower(L^T dL)/2 L^-1. Both sides reuse
// the same factor and RHS in shared memory; no global transpose is needed.
template<int rank>
__global__ __launch_bounds__(128) void compact_cholesky_backward_kernel(
    const float* factor, const float* product, float* output) {
    using Left = decltype(
        cusolverdx::Size<rank, rank>() + cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::left>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::Diag<cusolverdx::diag::non_unit>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::transposed>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major,cusolverdx::arrangement::row_major>() +
        cusolverdx::LeadingDimension<rank,rank>() +
        cusolverdx::Block() + cusolverdx::BlockDim<128>() +
        cusolverdx::BatchesPerBlock<1>() + cusolverdx::SM<kCompiledSm>());
    using Right = decltype(
        cusolverdx::Size<rank, rank>() + cusolverdx::Precision<float>() +
        cusolverdx::Type<cusolverdx::type::real>() +
        cusolverdx::Function<cusolverdx::function::trsm>() +
        cusolverdx::Side<cusolverdx::side::right>() +
        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
        cusolverdx::Diag<cusolverdx::diag::non_unit>() +
        cusolverdx::TransposeMode<cusolverdx::transpose::non_transposed>() +
        cusolverdx::Arrangement<cusolverdx::arrangement::row_major,cusolverdx::arrangement::row_major>() +
        cusolverdx::LeadingDimension<rank,rank>() +
        cusolverdx::Block() + cusolverdx::BlockDim<128>() +
        cusolverdx::BatchesPerBlock<1>() + cusolverdx::SM<kCompiledSm>());
    __shared__ float a[rank * rank];
    __shared__ float b[rank * rank];
    const int64_t offset = int64_t(blockIdx.x) * rank * rank;
    for (int i = threadIdx.x; i < rank * rank; i += blockDim.x) {
        const int row = i / rank, col = i % rank;
        a[i] = factor[offset + i];
        b[i] = 0.5f * product[offset + (row >= col ? i : col * rank + row)];
    }
    __syncthreads();
    Left().execute(a, b);
    __syncthreads();
    Right().execute(a, b);
    __syncthreads();
    for (int i = threadIdx.x; i < rank * rank; i += blockDim.x)
        output[offset + i] = b[i];
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


static at::Tensor compact_triangular_solve(const at::Tensor& factor,const at::Tensor& rhs,bool transpose,bool paired) {
    TORCH_CHECK(factor.dim()>=2 && rhs.dim()==factor.dim(),"invalid triangular solve dimensions");
    auto dense_matrix=[](const at::Tensor& t){return t.is_contiguous() || t.transpose(-2,-1).is_contiguous();};
    for(const auto& t:{factor,rhs}) {
        TORCH_CHECK(t.is_cuda() && t.device()==factor.device() && t.scalar_type()==at::kFloat && dense_matrix(t),
            "triangular inputs must be dense CUDA FP32 on the factor device");
        TORCH_CHECK(!at::GradMode::is_enabled() || !t.requires_grad(),
            "compact native operations require an explicit autograd owner");
    }
    const auto rank=factor.size(-1);
    TORCH_CHECK(factor.size(-2)==rank && factor.numel()>0 && rhs.size(-1)>0,"empty or nonsquare triangular system");
    TORCH_CHECK(rank==16 || rank==32 || rank==48 || rank==64,"unsupported compact rank");
    for(int i=0;i<factor.dim()-1;++i)TORCH_CHECK(factor.size(i)==rhs.size(i),"triangular rhs batch/rank mismatch");
    c10::cuda::CUDAGuard guard(factor.device());
    (void)supported_sm();
    auto output=at::empty(rhs.sizes(),rhs.options());
    switch(rank) {
#define MEMSOLVE_TRSM_CASE(R) case R: launch_compact_trsm<R>(factor,rhs,output,transpose,paired); break;
        MEMSOLVE_TRSM_CASE(16) MEMSOLVE_TRSM_CASE(32) MEMSOLVE_TRSM_CASE(48) MEMSOLVE_TRSM_CASE(64)
#undef MEMSOLVE_TRSM_CASE
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor compact_trsm_cuda(const at::Tensor& factor,const at::Tensor& rhs,bool transpose) {
    return compact_triangular_solve(factor,rhs,transpose,false);
}

at::Tensor compact_potrs_cuda(const at::Tensor& factor,const at::Tensor& rhs) {
    return compact_triangular_solve(factor,rhs,false,true);
}

std::tuple<at::Tensor,at::Tensor> compact_factor_solve_cuda(const at::Tensor& gram,const at::Tensor& rhs,bool complete) {
    validate_compact_matrix(gram);
    TORCH_CHECK(rhs.is_cuda() && rhs.device()==gram.device() && rhs.scalar_type()==at::kFloat && rhs.is_contiguous(),
                "factor-solve rhs must be contiguous CUDA FP32 on the Gram device");
    TORCH_CHECK(!at::GradMode::is_enabled() || !rhs.requires_grad(), "compact native operations require an explicit autograd owner");
    TORCH_CHECK(rhs.dim()==gram.dim() && rhs.size(-1)>0,"invalid factor-solve rhs dimensions");
    for(int i=0;i<gram.dim()-1;++i) TORCH_CHECK(gram.size(i)==rhs.size(i),"factor-solve rhs batch/rank mismatch");
    c10::cuda::CUDAGuard guard(gram.device()); (void)supported_sm();
    auto factor=at::empty_like(gram),state=at::empty_like(rhs);
    switch(gram.size(-1)) {
#define MEMSOLVE_FACTOR_CASE(R) case R: launch_compact_factor_solve<R>(gram,rhs,factor,state,complete); break;
        MEMSOLVE_FACTOR_CASE(16) MEMSOLVE_FACTOR_CASE(32) MEMSOLVE_FACTOR_CASE(48) MEMSOLVE_FACTOR_CASE(64)
#undef MEMSOLVE_FACTOR_CASE
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {factor,state};
}

at::Tensor compact_cholesky_backward_cuda(const at::Tensor& factor, const at::Tensor& product) {
    validate_compact_matrix(factor);
    validate_compact_matrix(product);
    TORCH_CHECK(product.device() == factor.device() && product.sizes() == factor.sizes(),
                "Cholesky adjoint must match the factor device and shape");
    c10::cuda::CUDAGuard guard(factor.device());
    (void)supported_sm();
    auto output = at::empty_like(product);
    const auto systems = factor.numel() / (factor.size(-1) * factor.size(-1));
    switch (factor.size(-1)) {
#define MEMSOLVE_CHOLESKY_BWD_CASE(R) case R: compact_cholesky_backward_kernel<R><<<systems,128,0,at::cuda::getCurrentCUDAStream()>>>(factor.data_ptr<float>(),product.data_ptr<float>(),output.data_ptr<float>()); break;
        MEMSOLVE_CHOLESKY_BWD_CASE(16) MEMSOLVE_CHOLESKY_BWD_CASE(32)
        MEMSOLVE_CHOLESKY_BWD_CASE(48) MEMSOLVE_CHOLESKY_BWD_CASE(64)
#undef MEMSOLVE_CHOLESKY_BWD_CASE
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace memsolve_equilibrium
