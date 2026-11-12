#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace memsolve_equilibrium {
namespace native_solvers {

// Datacenter SMs (SM80 A100/A800, SM90 H100/H200/H800, SM100 B100/B200/GB200)
// provide 1:2-rate hardware FP64 units (32 DFMA/clk/SM), enabling full 53-bit
// IEEE FP64 LU & GETRS solves at high speed. Consumer/edge SMs use FP32 working
// storage with IEEE FP64 pivot reciprocals and __fmaf_rn.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
using LuWorkScalar = double;
#else
using LuWorkScalar = float;
#endif

constexpr int kNativeRhsTile = 16;

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

__device__ __forceinline__ double fma_rn(double a, double b, double c) {
    return __fma_rn(a, b, c);
}

// Single-barrier-per-step block-unrolled Gauss-Jordan Cholesky sweep for rank in {16, 32, 48, 64}:
// 128 threads = 16 row-lanes x 8 col-lanes. Only active lower-triangular blocks (<= 20 floats/thread)
// are materialized in registers; full compile-time unrolling eliminates all dynamic register/SMEM
// muxes and guarantees STACK:0, LOCAL:0 across SM80 and SM90a.
template <int rank>
__global__ __launch_bounds__(128, (rank <= 32 ? 8 : 4)) void compact_inverse_cta_kernel(
    const float* __restrict__ gram,
    float* __restrict__ output) {
    static_assert(rank == 16 || rank == 32 || rank == 48 || rank == 64, "CTA inverse supports rank 16, 32, 48, or 64");
    constexpr int kThreadsPerBlock = 128;
    constexpr int kLd = rank + 1;
    constexpr int kRows = rank / 16;  // 3 for r=48, 4 for r=64
    constexpr int kCols = rank / 8;   // 6 for r=48, 8 for r=64

    __shared__ float s_io[rank * kLd];
    __shared__ float s_col[2][rank];
    __shared__ float s_row[2][rank];
    __shared__ double s_diag[rank];
    __shared__ double s_inv[2];

    const int tid = threadIdx.x;
    const int r_tid = tid >> 3;  // 0..15
    const int c_tid = tid & 7;   // 0..7
    const int64_t sys = blockIdx.x;
    const float* __restrict__ g_sys = gram + sys * rank * rank;
    float* __restrict__ out_sys = output + sys * rank * rank;

    for (int idx = tid; idx < rank * rank; idx += kThreadsPerBlock) {
        s_io[(idx / rank) * kLd + (idx % rank)] = g_sys[idx];
    }
    __syncthreads();

    float reg_w[kRows][kCols];
    #pragma unroll
    for (int ri = 0; ri < kRows; ++ri) {
        const int r = ri * 16 + r_tid;
        #pragma unroll
        for (int ci = 0; ci <= 2 * ri + 1; ++ci) {
            const int c = ci * 8 + c_tid;
            if (r > c) {
                const double sym = 0.5 * (static_cast<double>(s_io[r * kLd + c]) +
                                          static_cast<double>(s_io[c * kLd + r]));
                reg_w[ri][ci] = static_cast<float>(sym);
            } else if (r == c) {
                const double d = 1.0 + static_cast<double>(s_io[r * kLd + r]);
                reg_w[ri][ci] = static_cast<float>(d);
                s_diag[r] = d;
            } else {
                reg_w[ri][ci] = 0.0f;
            }
        }
    }

    #pragma unroll
    for (int kb = 0; kb < kRows; ++kb) {
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const int k_ri = kb;
            const int k_ci = 2 * kb + kh;
            #pragma unroll
            for (int kc = 0; kc < 8; ++kc) {
                const int k_rtid = kh * 8 + kc;
                const int k = k_ci * 8 + kc;
                const int buf = kc & 1;

                if (c_tid == kc) {
                    if (r_tid > k_rtid) {
                        s_col[buf][k_ri * 16 + r_tid] = reg_w[k_ri][k_ci];
                    }
                    #pragma unroll
                    for (int ri = k_ri + 1; ri < kRows; ++ri) {
                        s_col[buf][ri * 16 + r_tid] = reg_w[ri][k_ci];
                    }
                }
                if (r_tid == k_rtid) {
                    #pragma unroll
                    for (int ci = 0; ci < k_ci; ++ci) {
                        s_row[buf][ci * 8 + c_tid] = reg_w[k_ri][ci];
                    }
                    if (c_tid < kc) {
                        s_row[buf][k_ci * 8 + c_tid] = reg_w[k_ri][k_ci];
                    } else if (c_tid == kc) {
                        const double d_kk = s_diag[k];
                        s_inv[buf] = __ddiv_rn(1.0, __dsqrt_rn(d_kk > 1e-30 ? d_kk : 1e-30));
                    }
                }
                __syncthreads();

                const double inv_d_fp64 = s_inv[buf];
                const float inv_d = static_cast<float>(inv_d_fp64);

                if (r_tid == k_rtid) {
                    #pragma unroll
                    for (int ci = 0; ci < k_ci; ++ci) {
                        reg_w[k_ri][ci] = __fmul_rn(s_row[buf][ci * 8 + c_tid], inv_d);
                    }
                    if (c_tid < kc) {
                        reg_w[k_ri][k_ci] = __fmul_rn(s_row[buf][k_ci * 8 + c_tid], inv_d);
                    } else if (c_tid == kc) {
                        reg_w[k_ri][k_ci] = inv_d;
                    }
                }

                float l_row[kRows];
                l_row[k_ri] = (r_tid > k_rtid)
                    ? __fmul_rn(s_col[buf][k_ri * 16 + r_tid], inv_d)
                    : 0.0f;
                #pragma unroll
                for (int ri = k_ri + 1; ri < kRows; ++ri) {
                    l_row[ri] = __fmul_rn(s_col[buf][ri * 16 + r_tid], inv_d);
                }

                if (c_tid == kc) {
                    if (r_tid > k_rtid) {
                        reg_w[k_ri][k_ci] = __fmul_rn(-l_row[k_ri], inv_d);
                    }
                    #pragma unroll
                    for (int ri = k_ri + 1; ri < kRows; ++ri) {
                        reg_w[ri][k_ci] = __fmul_rn(-l_row[ri], inv_d);
                    }
                }

                if (!(kb == kRows - 1 && kh == 1 && kc == 7)) {
                    if ((r_tid & 7) == c_tid) {
                        #pragma unroll
                        for (int ri = k_ri; ri < kRows; ++ri) {
                            if (ri > k_ri || r_tid > k_rtid) {
                                const int r = ri * 16 + r_tid;
                                const double ld = static_cast<double>(s_col[buf][r]) * inv_d_fp64;
                                s_diag[r] = __fma_rn(-ld, ld, s_diag[r]);
                            }
                        }
                    }

                    float p_col[kCols];
                    #pragma unroll
                    for (int ci = 0; ci < k_ci; ++ci) {
                        p_col[ci] = __fmul_rn(s_row[buf][ci * 8 + c_tid], inv_d);
                    }
                    p_col[k_ci] = (c_tid < kc)
                        ? __fmul_rn(s_row[buf][k_ci * 8 + c_tid], inv_d)
                        : ((c_tid > kc)
                            ? __fmul_rn(s_col[buf][k_ci * 8 + c_tid], inv_d)
                            : 0.0f);
                    #pragma unroll
                    for (int ci = k_ci + 1; ci < kCols; ++ci) {
                        p_col[ci] = __fmul_rn(s_col[buf][ci * 8 + c_tid], inv_d);
                    }

                    #pragma unroll
                    for (int ri = k_ri; ri < kRows; ++ri) {
                        const int r = ri * 16 + r_tid;
                        if (ri > k_ri || r_tid > k_rtid) {
                            const float neg_l = -l_row[ri];
                            #pragma unroll
                            for (int ci = 0; ci <= 2 * ri + 1; ++ci) {
                                const int c = ci * 8 + c_tid;
                                if ((ci < 2 * ri || c < r) && (ci != k_ci || c_tid != kc)) {
                                    reg_w[ri][ci] = __fmaf_rn(neg_l, p_col[ci], reg_w[ri][ci]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int ri = 0; ri < kRows; ++ri) {
        const int r = ri * 16 + r_tid;
        #pragma unroll
        for (int ci = 0; ci < kCols; ++ci) {
            const int c = ci * 8 + c_tid;
            if (ci <= 2 * ri + 1 && r >= c) {
                s_io[r * kLd + c] = reg_w[ri][ci];
            } else {
                s_io[r * kLd + c] = 0.0f;
            }
        }
    }
    __syncthreads();

    for (int idx = tid; idx < rank * rank; idx += kThreadsPerBlock) {
        out_sys[idx] = s_io[(idx / rank) * kLd + (idx % rank)];
    }
}

template <int rank>
__host__ __device__ constexpr int lu_cta_threads() {
    return 256;
}

// Bank-conflict-free SMEM row-major partial-pivoted LU factorization (GETRF):
// 256 threads = 16x16 2D tile (zero integer division in trailing updates),
// 1-based LAPACK pivots (1..rank) and 1-based singular status (0 ok, k+1 on zero pivot).
template <int rank>
__global__ __launch_bounds__(256) void compact_lu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ pivots,
    int* __restrict__ status) {
    static_assert(rank == 16 || rank == 32 || rank == 48 || rank == 64, "unsupported compact rank");
    using WorkScalar = LuWorkScalar;
    constexpr int kLd = rank + 1;
    constexpr int kTiles = rank / 16;

    __shared__ WorkScalar s_a[rank * kLd];
    __shared__ WorkScalar s_l[rank];
    __shared__ int s_pivots[rank];
    __shared__ int s_piv_row;
    __shared__ int s_info;
    __shared__ double s_inv_piv;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int r_tid = tid >> 4;  // 0..15
    const int c_tid = tid & 15;  // 0..15
    const int64_t s = blockIdx.x;
    const float* __restrict__ in_sys = input + s * rank * rank;
    float* __restrict__ out_sys = output + s * rank * rank;

    #pragma unroll
    for (int ri = 0; ri < kTiles; ++ri) {
        const int r = ri * 16 + r_tid;
        #pragma unroll
        for (int ci = 0; ci < kTiles; ++ci) {
            const int c = ci * 16 + c_tid;
            s_a[r * kLd + c] = static_cast<WorkScalar>(in_sys[r * rank + c]);
        }
    }
    if (tid == 0) {
        s_info = 0;
    }
    __syncthreads();

    for (int k = 0; k < rank; ++k) {
        if (warp_id == 0) {
            double best_val = -1.0;
            int best_row = k;
            for (int r = k + lane; r < rank; r += 32) {
                const double v = fabs(static_cast<double>(s_a[r * kLd + k]));
                if (v > best_val || (v == best_val && r < best_row)) {
                    best_val = v;
                    best_row = r;
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const double other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                const int other_row = __shfl_down_sync(0xffffffffu, best_row, offset);
                if (other_val > best_val || (other_val == best_val && other_row < best_row)) {
                    best_val = other_val;
                    best_row = other_row;
                }
            }
            if (lane == 0) {
                s_piv_row = best_row;
                s_pivots[k] = best_row + 1;
                const double d_piv = static_cast<double>(s_a[best_row * kLd + k]);
                if (d_piv == 0.0) {
                    if (s_info == 0) {
                        s_info = k + 1;
                    }
                    s_inv_piv = 0.0;
                } else {
                    s_inv_piv = __ddiv_rn(1.0, d_piv);
                }
            }
        }
        __syncthreads();

        const int piv = s_piv_row;
        if (piv != k) {
            if (tid < rank) {
                const WorkScalar tmp = s_a[k * kLd + tid];
                s_a[k * kLd + tid] = s_a[piv * kLd + tid];
                s_a[piv * kLd + tid] = tmp;
            }
            __syncthreads();
        }

        if (k < rank - 1) {
            const double inv_piv = s_inv_piv;
            if (inv_piv != 0.0) {
                if (tid > k && tid < rank) {
                    const WorkScalar l_rk = static_cast<WorkScalar>(
                        static_cast<double>(s_a[tid * kLd + k]) * inv_piv);
                    s_a[tid * kLd + k] = l_rk;
                    s_l[tid] = l_rk;
                }
                __syncthreads();

                #pragma unroll
                for (int ri = 0; ri < kTiles; ++ri) {
                    const int r = ri * 16 + r_tid;
                    if (r > k) {
                        const WorkScalar neg_l = -s_l[r];
                        #pragma unroll
                        for (int ci = 0; ci < kTiles; ++ci) {
                            const int c = ci * 16 + c_tid;
                            if (c > k) {
                                s_a[r * kLd + c] = fma_rn(neg_l, s_a[k * kLd + c], s_a[r * kLd + c]);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    #pragma unroll
    for (int ri = 0; ri < kTiles; ++ri) {
        const int r = ri * 16 + r_tid;
        #pragma unroll
        for (int ci = 0; ci < kTiles; ++ci) {
            const int c = ci * 16 + c_tid;
            out_sys[r * rank + c] = static_cast<float>(s_a[r * kLd + c]);
        }
    }
    if (tid < rank) {
        pivots[s * rank + tid] = s_pivots[tid];
    }
    if (tid == 0) {
        status[s] = s_info;
    }
}

// Compile-time-unrolled warp-shuffle GETRS with 16 RHS columns/CTA:
// 128 threads = 4 warps x 4 RHS cols/warp (c_sub = lane & 3) x 8 row-sublanes/col (r_sub = lane >> 2).
// Both triangular solves execute 100% in registers via __shfl_sync with ZERO CTA barriers in the loops.
template <int rank, bool transpose>
__global__ __launch_bounds__(128) void compact_getrs_kernel(
    const float* __restrict__ factors,
    const int* __restrict__ pivots,
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t columns) {
    static_assert(rank == 16 || rank == 32 || rank == 48 || rank == 64, "unsupported compact rank");
    using RhsScalar = LuWorkScalar;
    constexpr int kThreadsPerBlock = 128;
    constexpr int kLda = rank + 1;
    constexpr int kLdb = kNativeRhsTile + 1;  // 17 (bank-conflict-free)
    constexpr int kRowsPerThread = rank / 8;  // 2, 4, 6, 8

    __shared__ float s_lu[rank * kLda];
    __shared__ float s_rhs[rank * kLdb];
    __shared__ double s_inv_u[rank];
    __shared__ int s_piv[rank];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int r_sub = lane >> 2;  // 0..7
    const int c_sub = lane & 3;   // 0..3
    const int col = warp_id * 4 + c_sub;  // 0..15 RHS column within tile
    const int64_t s = blockIdx.x;
    const int64_t start = static_cast<int64_t>(blockIdx.y) * kNativeRhsTile;

    for (int idx = tid; idx < rank * rank; idx += kThreadsPerBlock) {
        s_lu[(idx / rank) * kLda + (idx % rank)] = factors[s * rank * rank + idx];
    }
    for (int idx = tid; idx < rank * kNativeRhsTile; idx += kThreadsPerBlock) {
        const int r = idx / kNativeRhsTile;
        const int c = idx % kNativeRhsTile;
        s_rhs[r * kLdb + c] = (start + c < columns) ? input[(s * rank + r) * columns + start + c] : 0.0f;
    }
    if (tid < rank) {
        s_piv[tid] = pivots[s * rank + tid];
    }
    __syncthreads();

    if (tid >= 64 && (tid - 64) < rank) {
        const int d_idx = tid - 64;
        const double u_kk = static_cast<double>(s_lu[d_idx * kLda + d_idx]);
        s_inv_u[d_idx] = (u_kk != 0.0) ? __ddiv_rn(1.0, u_kk) : 0.0;
    }
    if constexpr (!transpose) {
        if (tid < kNativeRhsTile) {
            for (int i = 0; i < rank; ++i) {
                const int piv = s_piv[i] - 1;
                if (piv != i) {
                    const float tmp = s_rhs[i * kLdb + tid];
                    s_rhs[i * kLdb + tid] = s_rhs[piv * kLdb + tid];
                    s_rhs[piv * kLdb + tid] = tmp;
                }
            }
        }
    }
    __syncthreads();

    RhsScalar reg_x[kRowsPerThread];
    #pragma unroll
    for (int q = 0; q < kRowsPerThread; ++q) {
        const int row = q * 8 + r_sub;
        reg_x[q] = static_cast<RhsScalar>(s_rhs[row * kLdb + col]);
    }

    if constexpr (!transpose) {
        // 1. Unit lower-triangular forward solve: L Y = P B (compile-time unrolled, zero CTA barriers).
        #pragma unroll
        for (int k_q = 0; k_q < kRowsPerThread; ++k_q) {
            #pragma unroll
            for (int k_sub = 0; k_sub < 8; ++k_sub) {
                const int k = k_q * 8 + k_sub;
                if (k < rank - 1) {
                    const RhsScalar y_k = __shfl_sync(0xffffffffu, reg_x[k_q], (k_sub << 2) | c_sub);
                    if (r_sub > k_sub) {
                        const int row = k_q * 8 + r_sub;
                        const RhsScalar l_ik = static_cast<RhsScalar>(s_lu[row * kLda + k]);
                        reg_x[k_q] = fma_rn(-l_ik, y_k, reg_x[k_q]);
                    }
                    #pragma unroll
                    for (int q = k_q + 1; q < kRowsPerThread; ++q) {
                        const int row = q * 8 + r_sub;
                        const RhsScalar l_ik = static_cast<RhsScalar>(s_lu[row * kLda + k]);
                        reg_x[q] = fma_rn(-l_ik, y_k, reg_x[q]);
                    }
                }
            }
        }

        // 2. Non-unit upper-triangular backward solve: U X = Y (compile-time unrolled, zero CTA barriers).
        #pragma unroll
        for (int k_q = kRowsPerThread - 1; k_q >= 0; --k_q) {
            #pragma unroll
            for (int k_sub = 7; k_sub >= 0; --k_sub) {
                const int k = k_q * 8 + k_sub;
                if (r_sub == k_sub) {
                    reg_x[k_q] = static_cast<RhsScalar>(static_cast<double>(reg_x[k_q]) * s_inv_u[k]);
                }
                if (k > 0) {
                    const RhsScalar x_k = __shfl_sync(0xffffffffu, reg_x[k_q], (k_sub << 2) | c_sub);
                    if (r_sub < k_sub) {
                        const int row = k_q * 8 + r_sub;
                        const RhsScalar u_ik = static_cast<RhsScalar>(s_lu[row * kLda + k]);
                        reg_x[k_q] = fma_rn(-u_ik, x_k, reg_x[k_q]);
                    }
                    #pragma unroll
                    for (int q = 0; q < k_q; ++q) {
                        const int row = q * 8 + r_sub;
                        const RhsScalar u_ik = static_cast<RhsScalar>(s_lu[row * kLda + k]);
                        reg_x[q] = fma_rn(-u_ik, x_k, reg_x[q]);
                    }
                }
            }
        }
    } else {
        // 1. Non-unit lower-triangular forward solve: U^T Y = B (compile-time unrolled, zero CTA barriers).
        #pragma unroll
        for (int k_q = 0; k_q < kRowsPerThread; ++k_q) {
            #pragma unroll
            for (int k_sub = 0; k_sub < 8; ++k_sub) {
                const int k = k_q * 8 + k_sub;
                if (r_sub == k_sub) {
                    reg_x[k_q] = static_cast<RhsScalar>(static_cast<double>(reg_x[k_q]) * s_inv_u[k]);
                }
                if (k < rank - 1) {
                    const RhsScalar y_k = __shfl_sync(0xffffffffu, reg_x[k_q], (k_sub << 2) | c_sub);
                    if (r_sub > k_sub) {
                        const int row = k_q * 8 + r_sub;
                        const RhsScalar u_ki = static_cast<RhsScalar>(s_lu[k * kLda + row]);
                        reg_x[k_q] = fma_rn(-u_ki, y_k, reg_x[k_q]);
                    }
                    #pragma unroll
                    for (int q = k_q + 1; q < kRowsPerThread; ++q) {
                        const int row = q * 8 + r_sub;
                        const RhsScalar u_ki = static_cast<RhsScalar>(s_lu[k * kLda + row]);
                        reg_x[q] = fma_rn(-u_ki, y_k, reg_x[q]);
                    }
                }
            }
        }

        // 2. Unit upper-triangular backward solve: L^T Z = Y (compile-time unrolled, zero CTA barriers).
        #pragma unroll
        for (int k_q = kRowsPerThread - 1; k_q >= 0; --k_q) {
            #pragma unroll
            for (int k_sub = 7; k_sub >= 0; --k_sub) {
                const int k = k_q * 8 + k_sub;
                if (k > 0) {
                    const RhsScalar z_k = __shfl_sync(0xffffffffu, reg_x[k_q], (k_sub << 2) | c_sub);
                    if (r_sub < k_sub) {
                        const int row = k_q * 8 + r_sub;
                        const RhsScalar l_ki = static_cast<RhsScalar>(s_lu[k * kLda + row]);
                        reg_x[k_q] = fma_rn(-l_ki, z_k, reg_x[k_q]);
                    }
                    #pragma unroll
                    for (int q = 0; q < k_q; ++q) {
                        const int row = q * 8 + r_sub;
                        const RhsScalar l_ki = static_cast<RhsScalar>(s_lu[k * kLda + row]);
                        reg_x[q] = fma_rn(-l_ki, z_k, reg_x[q]);
                    }
                }
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int q = 0; q < kRowsPerThread; ++q) {
        const int row = q * 8 + r_sub;
        s_rhs[row * kLdb + col] = static_cast<float>(reg_x[q]);
    }
    __syncthreads();

    if constexpr (transpose) {
        if (tid < kNativeRhsTile) {
            for (int i = rank - 1; i >= 0; --i) {
                const int piv = s_piv[i] - 1;
                if (piv != i) {
                    const float tmp = s_rhs[i * kLdb + tid];
                    s_rhs[i * kLdb + tid] = s_rhs[piv * kLdb + tid];
                    s_rhs[piv * kLdb + tid] = tmp;
                }
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < rank * kNativeRhsTile; idx += kThreadsPerBlock) {
        const int r = idx / kNativeRhsTile;
        const int c = idx % kNativeRhsTile;
        if (start + c < columns) {
            output[(s * rank + r) * columns + start + c] = s_rhs[r * kLdb + c];
        }
    }
}

}  // namespace native_solvers
}  // namespace memsolve_equilibrium
