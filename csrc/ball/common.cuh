#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <tuple>

namespace memsolve_equilibrium {
#ifndef MEMSOLVE_CUDA_TARGET_SM
#error "MEMSOLVE_CUDA_TARGET_SM must be defined by the strict CUDA build."
#endif
constexpr int kRhsTile = 32;
constexpr int kThreads = 256;
constexpr int kCompiledSm = MEMSOLVE_CUDA_TARGET_SM;
constexpr std::array<int, 7> kRuntimeArchitectures = {80, 86, 87, 89, 90, 100, 120};
#ifdef MEMSOLVE_SUPPORTED_SM_MASK
constexpr int kSupportedSmMask = MEMSOLVE_SUPPORTED_SM_MASK;
#else
constexpr int kSupportedSmMask = 0x7f;
#endif

constexpr bool is_sm_compiled(int normalized_sm) noexcept {
    for (size_t i = 0; i < kRuntimeArchitectures.size(); ++i) {
        if (kRuntimeArchitectures[i] == normalized_sm) {
            return (kSupportedSmMask & (1 << static_cast<int>(i))) != 0;
        }
    }
    return false;
}

inline int supported_sm() {
    struct CachedSmTable {
        std::array<std::atomic<int>, 16> values;
        CachedSmTable() noexcept {
            for (auto& entry : values) {
                entry.store(-1, std::memory_order_relaxed);
            }
        }
    };
    static CachedSmTable cache;

    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    if (device >= 0 && static_cast<size_t>(device) < cache.values.size()) {
        const int cached = cache.values[static_cast<size_t>(device)].load(std::memory_order_relaxed);
        if (C10_LIKELY(cached != -1)) {
            return cached;
        }
    }

    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const int capability = properties.major * 100 + properties.minor * 10;
    const int normalized_capability =
        (capability == 1030 || capability == 1100) ? 1000
        : (capability == 1210 ? 1200 : capability);
    if constexpr (kCompiledSm == 0) {
        TORCH_CHECK(
            capability >= 800,
            "the MemSolve CUDA fast path supports Ampere-and-newer architectures (SM80+); "
            "got SM", properties.major, ".", properties.minor);
        TORCH_CHECK(
            is_sm_compiled(normalized_capability / 10),
            "the loaded MemSolve CUDA extension does not include SM",
            properties.major, ".", properties.minor,
            "; rebuild with MEMSOLVE_CUDA_ARCHITECTURES covering this GPU");
        if (device >= 0 && static_cast<size_t>(device) < cache.values.size()) {
            cache.values[static_cast<size_t>(device)].store(
                normalized_capability, std::memory_order_relaxed);
        }
        return normalized_capability;
    } else {
        TORCH_CHECK(
            normalized_capability == kCompiledSm,
            "the loaded MemSolve CUDA extension targets SM", kCompiledSm,
            " but received SM", properties.major, ".", properties.minor,
            "; load the matching strict artifact");

        switch (normalized_capability) {
            case 800:
            case 860:
            case 870:
            case 890:
            case 900:
            case 1000:
            case 1200:
                if (device >= 0 && static_cast<size_t>(device) < cache.values.size()) {
                    cache.values[static_cast<size_t>(device)].store(
                        normalized_capability, std::memory_order_relaxed);
                }
                return normalized_capability;
            default:
                TORCH_CHECK(false,
                            "the MemSolve CUDA fast path supports known Ampere-and-newer "
                            "architectures (SM80, SM86, SM87, SM89, SM90, SM100, SM120); "
                            "got SM", properties.major, ".", properties.minor);
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compact_lu_cuda(const at::Tensor& a);
at::Tensor compact_getrs_cuda(const at::Tensor& lu, const at::Tensor& pivots,
                             const at::Tensor& rhs, bool transpose);
at::Tensor compact_inverse_cuda(const at::Tensor& gram);
}  // namespace memsolve_equilibrium
