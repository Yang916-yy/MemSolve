#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <tuple>

namespace memsolve_equilibrium {
#ifndef MEMSOLVE_CUDA_TARGET_SM
#error "MEMSOLVE_CUDA_TARGET_SM must be defined by the strict CUDA build."
#endif
constexpr int kRhsTile = 32;
constexpr int kThreads = 256;
constexpr int kCompiledSm = MEMSOLVE_CUDA_TARGET_SM;

inline int supported_sm() {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const int capability = properties.major * 100 + properties.minor * 10;
    const int normalized_capability = capability == 1210 ? 1200 : capability;
    TORCH_CHECK(
        normalized_capability == kCompiledSm,
        "the loaded MemSolve CUDA extension targets SM", kCompiledSm,
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
                        "the MemSolve CUDA fast path supports known Ampere-and-newer "
                        "architectures (SM80, SM86, SM87, SM89, SM90, SM100, SM120); "
                        "got SM", properties.major, ".", properties.minor);
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compact_lu_cuda(const at::Tensor& a);
at::Tensor compact_getrs_cuda(const at::Tensor& lu, const at::Tensor& pivots,
                             const at::Tensor& rhs, bool transpose);
at::Tensor compact_potrs_cuda(const at::Tensor& factor, const at::Tensor& rhs);
at::Tensor compact_trsm_cuda(const at::Tensor& factor, const at::Tensor& rhs, bool transpose);
std::tuple<at::Tensor,at::Tensor> compact_factor_solve_cuda(const at::Tensor& gram,const at::Tensor& rhs,bool complete);
at::Tensor compact_cholesky_backward_cuda(const at::Tensor& factor, const at::Tensor& product);
}  // namespace memsolve_equilibrium
