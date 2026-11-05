# CUDA Implementation & Fast Path

This document describes the mixed-precision CUDA and Triton implementation in [`memsolve/ball/cuda.py`](../memsolve/ball/cuda.py) and [`csrc/ball/`](../csrc/ball/). The mathematical specification lives in [`reference.py`](../memsolve/ball/reference.py); the CUDA backend evaluates the exact same operator without materializing the token-sized orthonormal frame $P \in \mathbb{R}^{N \times r}$.

## Supported Configurations

`MemSolve.forward(..., implementation="cuda")` supports:
- **Inputs**: Contiguous CUDA tensors in `float16` or `bfloat16` (outputs match the input dtype).
- **Parameters**: `float32` parameters.
- **Ranks**: $r \in \{16, 32, 48, 64\}$ with any positive head dimension $d$ (even $d$ when RoVE is enabled).
- **Modes**: `equilibrium` (default) and `complement`.
- **Masks**: Arbitrary boolean `[B, N]` validity masks, including gaps or fully masked samples.

The fused input projection sanitizes invalid/masked input rows and casts to BF16 in a single pass, and masks invalid tokens in the projection output and VJP epilogues so `NaN`s in padded positions never pollute activations or weight gradients. You can also pass precomputed `valid_counts` (`max(valid_tokens, 1)` as FP32 `[B]`) across layers to avoid recomputing mask reductions.

For low-level integration, `cuda.fast_mix(projected, core_base_raw, eta_raw, ...)` takes pre-projected contiguous BF16 inputs `[B, N, H*r + D]` directly.

## Frame-Free Formulation

Instead of forming $P = A R^{-1} \in \mathbb{R}^{N \times r}$ in global memory, both forward and backward operate on compact $r \times r$ and $r \times d$ matrices per head:

```text
Forward (ordinary rank-space path, N > r):
  G = A^T A                                    [B, H, r, r]  Gram matrix
  Q = A^T C        (or A^T rot(C) with RoVE)   [B, H, r, d]  cross-statistic
  R^T R = I + G
  T = R^{-T}                                   [B, H, r, r]  upper-triangular Cholesky inverse factor
  Z = T Q                                      [B, H, r, d]  normalized state
  F = (I + J)^{-1} (I - J)                     [H, r, r]     shared equilibrium map (0 in complement)
  W = T^T (F - eta I) Z                        [B, H, r, d]  readout coefficients
  Y = eta C + A W  (or eta C + invrot(A W))    [B, N, D]     token readout
```

When physical sequence length $N \le r$, the base complement term solves the smaller $N \times N$ token system $\eta (I + A A^\top)^{-1} C$ while keeping the learned core correction in rank space, avoiding catastrophic cancellation when strong relations span very few tokens.

### Backward Pass

Let $L = R^\top = T^{-1}$, $M = F - \eta I$, and let $E = A^\top \mathrm{d}Y$ (or $A^\top \operatorname{rot}(\mathrm{d}Y)$) be the upstream readout-coefficient adjoint. With $U = T E$ and $H = U Z^\top$:

```text
  dF   = sum_B(H)                              (equilibrium only)
  deta = <dY, C> - trace(dF)
  dQ   = T^T M^T U
  dG   = T^T sym_lower(-M H^T - M^T H) T
```

where $\operatorname{sym\_lower}(X) = \frac{1}{2}\bigl(\operatorname{tril}(X) + \operatorname{tril}(X, -1)^\top\bigr)$, matching $\operatorname{sym}_\ell$ in Eq. (28) of the paper. For `equilibrium`, the shared core adjoint sums $H$ across the batch and reuses the saved partial-pivot LU factors of $I + J$ in a single transposed solve: $U_{\text{core}} = (I + J)^{-\top} \mathrm{d}F$ and $\mathrm{d}J = -U_{\text{core}} (F + I)^\top$.

## Kernel Architecture & Numerical Precision

1. **Native cuSOLVERDx & cuBLASLt Primitives (`csrc/ball/forward.cu`)**:
   - `compact_inverse`: One CTA per $(b, h)$ system forms $I + G$, runs shared-memory Cholesky (`POTRF`), and solves against the identity (`TRSM`) entirely on-chip, writing only $T = R^{-\top}$ in FP32 to global memory.
   - `compact_lu` / `compact_getrs`: Shared-memory partial-pivot LU factorization and transpose solve for the $H$ shared $r \times r$ core systems $(I + J)$.
   - `linear_weight_bias`: Fused BF16 weight + bias gradient GEMM via cuBLASLt `BGRADB` epilogues with a thread-local, CUDA-Graph-safe plan cache.

2. **Fused Triton Compact Chains (`memsolve/ball/cuda.py`)**:
   - **Forward (`_compact_chain`)**: Computes $Z = T Q$, applies $F - \eta I$ (by subtracting $\eta$ directly from the diagonal of $F$ in IEEE FP32 before the dot product for $r \le 32$, avoiding subtraction cancellation), and multiplies by $T^\top$ in a single kernel launch. In inference mode (`requires_grad=False`) at $r \le 32$, `STORE_FIRST=False` skips writing $Z$ to global memory altogether.
   - **Backward (`compact_chain_bwd_r32` / `compact_chain`)**: At $r \in \{16, 32\}$, fuses all four compact matrix products ($U = T E$, $H = U Z^\top$, $\mathrm{d}Z = (F^\top - \eta I) U$, and $\mathrm{d}Q = T^\top \mathrm{d}Z$) in registers without materializing $U$ in global memory, while `cholesky_product` and `compact_conjugate_tiled` evaluate the symmetric lower projection and $T^\top (\cdot) T$ conjugation products.
   - **Zero Register Spills**: Tile sizes and warp counts (`num_warps=4` or `8`) are tuned across all 12 Triton kernels so that every compiled specialization at $r \in \{16, 32, 48, 64\}$ compiles with **0 register spills** on SM80.

3. **Precision Design**:
   - Token projections and statistics multiply BF16 operands with FP32 accumulation.
   - All compact $r \times r$ and $r \times d$ solves and products remain in FP32 (using Triton's `tf32x3` three-pass IEEE FP32 emulation for Tensor Core dot products).
   - Token readout and input VJPs split sensitive FP32 compact matrices ($W$, $\mathrm{d}G$, $\mathrm{d}Q$) into a high BF16 word and a residual BF16 word ([Henry et al., 2019](https://arxiv.org/abs/1904.06376)), accumulating both products in FP32 so strong-relation cancellation does not lose precision.
   - Memory lifetimes are tightly managed: dead intermediates are freed immediately, and `maybe_clear_saved_tensors` releases autograd saved tensors before allocating the final token gradient (while respecting `retain_graph=True`).

## Exact Sequence Parallelism

`integrations.distributed.SequenceParallel` shards the sequence dimension across a process group with exact numerical equivalence to single-GPU execution:

```python
from integrations.distributed import SequenceParallel

sp = SequenceParallel.create(local_x.shape[1], group=sequence_group)
counts = sp.valid_counts(local_mask, batch_size=local_x.shape[0], device=local_x.device)
y = layer(local_x, local_mask, implementation="cuda",
          sequence_parallel=sp, valid_counts=counts)
local_loss.backward()
sp.reduce_gradients_(layer)
optimizer.step()
```

Each rank computes local $(G, Q)$ statistics on its shard; a single packed `all_reduce(SUM)` forms the global statistics in $\mathcal{O}(B H (r^2 + r d))$ communication volume—independent of sequence length $N$. Each rank then runs the compact solve and local token readout. Backward uses one symmetric compact `all_reduce(SUM)` plus `sp.reduce_gradients_(layer)` for replicated weights. For models with local depthwise convolution (`kernel=3`), `sp.halo(x, width=1)` exchanges 1-token boundary halos across adjacent ranks.

## Building the Native Extension

Requires PyTorch 2.14 (`+cu132`), CUDA 13.2, NVIDIA MathDx 26.06.1, C++20, and CMake $\ge 3.25$:

```bash
PYTHON=python MEMSOLVE_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh
```

Set `CUDA_HOME` and `MATHDX_ROOT` if your toolkit or MathDx headers are installed in non-default locations. `MEMSOLVE_CUDA_ARCHITECTURES` accepts a semicolon-separated list of SM targets (`80;86;87;89;90;100;120`).
