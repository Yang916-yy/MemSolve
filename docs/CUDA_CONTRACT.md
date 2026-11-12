# CUDA Implementation & Fast Path

This document describes the mixed-precision CUDA and Triton implementation in [`memsolve/ball/cuda.py`](../memsolve/ball/cuda.py) and [`csrc/ball/`](../csrc/ball/). The mathematical specification lives in [`reference.py`](../memsolve/ball/reference.py); the CUDA backend evaluates the exact same operator without materializing the token-sized orthonormal frame $P \in \mathbb{R}^{N \times r}$.

## Supported Configurations

`MemSolve.forward(..., implementation="cuda")` supports:
- **Inputs**: Contiguous CUDA tensors in `float16` or `bfloat16` (outputs match the input dtype).
- **Parameters**: `float32` parameters.
- **Ranks**: $r \in \{16, 32, 48, 64\}$ with any positive head dimension $d$ (even $d$ when RoVE is enabled).
- **Modes**: `equilibrium` (default) and `complement`.
- **Masks**: Arbitrary boolean `[B, N]` validity masks, including gaps or fully masked samples.
- **Readout recomputation (`recompute_readout: Literal["auto"] | bool = "auto"`)**: Configurable on `MemSolveConfig` and mutable at runtime via `layer.recompute_readout`. Under `"auto"` (default), the intermediate `[B, N, D]` mixed-token readout is reconstructed on the fly from compact factors during backward before the output-projection weight-gradient GEMM whenever sequence length $N > 512$ (and saved directly when $N \le 512$); `True` always recomputes and `False` always retains the readout buffer.
- **`torch.compile` & `CUDAGraph` compatibility**: All fast-path operations register with `torch.library.custom_op` (`memsolve::fast_mix_infer`, `memsolve::fast_mix_fwd`, `memsolve::fast_mix_bwd`, `memsolve::fast_mix_output_linear_fwd`, `memsolve::fast_mix_output_linear_bwd`, `memsolve::tensor_core_bmm{_backward}`, `memsolve::tensor_core_linear{_backward}`, `memsolve::compact_cholesky_inverse`, `memsolve::compact_lu_solve`) with `SymInt`-aware `.register_fake` kernels and `.register_autograd` rules, compiling end-to-end inside `torch.compile(fullgraph=True)` (static or `dynamic=True`) with zero Dynamo graph breaks and capturing cleanly inside `torch.cuda.CUDAGraph`. Static shapes (`has_static_value(N)`) resolve `recompute_readout="auto"` to concrete $N > 512$ prior to dispatch (while dynamic `SymInt` `"auto"` bounds temporary storage to `(B * 512, D)` and retains `mixed` iff $N \le 512$), short-sequence workspaces (`tok_inv_buf`, `res_buf`) are allocated only when $N \le r$ (using zero-element `(0,)` sentinels and zero extra `.copy_()` launches for $N > r$), and opaque linear forward ops cast FP32 weights to BF16 once in forward and save the BF16 buffer for backward while preserving exact parameter dtype on weight gradients.
- **Architectures**: Ampere-and-newer GPUs (`SM80`, `SM86`, `SM87`, `SM89`, `SM90`/`SM90a`, `SM100`, `SM120`) on both `x86_64` and `aarch64` (`sbsa-linux`, including Grace Hopper GH200 and GB200).

The fused input projection sanitizes invalid/masked input rows and casts to BF16 in a single pass, and masks invalid tokens in the projection output and VJP epilogues so `NaN`s in padded positions never pollute activations or weight gradients. You can also pass precomputed `valid_counts` (`max(valid_tokens, 1)` as FP32 `[B]`) across layers to avoid recomputing mask reductions.

For low-level integration, `cuda.fast_mix(projected, core_base_raw, eta_raw, ...)` takes pre-projected contiguous BF16 inputs `[B, N, H*r + D]` directly.

## Frame-Free Formulation

Instead of forming $P = A R^{-1} \in \mathbb{R}^{N \times r}$ in global memory, both forward and backward operate on compact $r \times r$ and $r \times d$ matrices per head:

```text
Forward (ordinary rank-space path, N > r):
  G = A^T A                                    [B, H, r, r]  Gram matrix
  Q = A^T C        (or A^T rot(C) with RoVE)   [B, H, r, d]  cross-statistic
  R^T R = I + G
  T = R^{-T}                                   [B, H, r, r]  lower-triangular Cholesky inverse factor
  Z = T Q                                      [B, H, r, d]  normalized state
  F = (I + J)^{-1} (I - J)                     [H, r, r]     shared equilibrium map (0 in complement)
  W = T^T (F - eta I) Z                        [B, H, r, d]  readout coefficients
  Y = eta C + A W  (or eta C + invrot(A W))    [B, N, D]     token readout
```

When physical sequence length $N \le r$, the base complement term solves the smaller $N \times N$ token system $\eta (I_N + A A^\top)^{-1} C$ while keeping the learned core correction in rank space, avoiding catastrophic cancellation when strong relations span very few tokens. Specifically, `short_stats` pads $A A^\top$ with zero off-diagonals and identity diagonal into a compact `[B, H, p, p]` buffer ($p \in \{16, 32, 48, 64\}$) for on-chip native Cholesky inversion ($T_{\text{tok}} (I_N + A A^\top) T_{\text{tok}}^\top = I_N$), and backward evaluates an explicit analytical FP32 VJP (`short_readout`, `short_bwd_prep`, `short_bwd_da`, `short_bwd_dc` plus `_compact_lu` / `_compact_getrs` for the shared equilibrium core) with no PyTorch autograd fallback.

### Backward Pass

Let $L = R^\top = T^{-1}$, $M = F - \eta I$, and let $E = A^\top \mathrm{d}Y$ (or $A^\top \operatorname{rot}(\mathrm{d}Y)$) be the upstream readout-coefficient adjoint. With $U = T E$ and $H = U Z^\top$:

```text
  dF   = sum_B(H)                              (equilibrium only)
  deta = <dY, C> - trace(dF)
  dQ   = T^T M^T U
  dG   = T^T sym_lower(-M H^T - M^T H) T
```

where $\operatorname{sym\_lower}(X) = \frac{1}{2}\bigl(\operatorname{tril}(X) + \operatorname{tril}(X, -1)^\top\bigr)$, matching $\operatorname{sym}_\ell$ in Eq. (39) (Appendix C.3) of the paper. For `equilibrium`, the shared core adjoint sums $H$ across the batch and reuses the saved partial-pivot LU factors of $I + J$ in a single transposed solve: $U_{\text{core}} = (I + J)^{-\top} \mathrm{d}F$ and $\mathrm{d}J = -U_{\text{core}} (F + I)^\top$.

## Kernel Architecture & Numerical Precision

1. **Native CUDA C++ Solvers & cuBLASLt Primitives (`csrc/ball/solvers.cuh`, `csrc/ball/forward.cu`, `csrc/ball/binding.cpp`)**:
   - By default (`MEMSOLVE_USE_MATHDX=OFF`, ABI contract version `17`), compact $r \times r$ linear algebra runs through self-contained on-chip CUDA C++ register/warp kernels in [`csrc/ball/solvers.cuh`](../csrc/ball/solvers.cuh) (with NVIDIA `cuSOLVERDx` in [`csrc/ball/forward.cu`](../csrc/ball/forward.cu) retained as an optional build mode via `MEMSOLVE_USE_MATHDX=ON`):
     - `compact_inverse`: One 128-thread CTA per $(b, h)$ system symmetrizes $G$, forms $I + G$, and executes a block-unrolled Gauss-Jordan Cholesky sweep (`compact_inverse_cta_kernel`) with IEEE FP64 diagonal accumulation (`__dsqrt_rn`, `__ddiv_rn`) on-chip, writing only the lower-triangular factor $T = R^{-\top}$ in FP32 to global memory.
     - `compact_lu` / `compact_getrs`: Bank-conflict-free shared-memory partial-pivot LU factorization (`lda = r + 1`, $< 36\text{ KB}$ static SMEM, 1-based pivots) and warp-shuffle register triangular solves (`N`/`T`) for the $H$ shared $r \times r$ core systems $(I + J)$, using hardware IEEE FP64 (`LuWorkScalar = double`) on datacenter SMs (`SM80`, `SM90`, `SM100`) and IEEE FP32 (`__fmaf_rn`) with FP64 pivot reciprocals elsewhere.
   - `linear_weight_bias` ([`csrc/ball/binding.cpp`](../csrc/ball/binding.cpp)): Fused BF16 weight + bias gradient GEMM via cuBLASLt `BGRADB` epilogues with a `32 MiB` workspace budget (`std::max(pool_ws, 32 MiB)`, reusing PyTorch's `at::cuda::getCUDABlasLtWorkspace()` when sufficient and transiently borrowing a stream-ordered buffer from `CUDACachingAllocator` otherwise) and a thread-local, bounded-LRU, CUDA-Graph-safe plan cache.

2. **Triton Kernels, Hopper `SM90a` Specializations & Zero Spills (`memsolve/ball/cuda.py`)**:
   - **Compact chains (`compact_chain`, `compact_chain_bwd_r32`, `cholesky_product`, `compact_conjugate_tiled`)**: Forward computes $Z = T Q$, applies $F - \eta I$ (subtracting $\eta$ directly from the diagonal of $F$ in IEEE FP32 before the dot product at $r \le 32$, avoiding subtraction cancellation), and multiplies by $T^\top$ in a single kernel launch; in inference mode (`requires_grad=False`) at $r \le 32$, `STORE_FIRST=False` skips writing $Z$ to global memory. At $r \in \{16, 32\}$, backward fuses all four compact products ($U = T E$, $H = U Z^\top$, $\mathrm{d}Z = (F^\top - \eta I) U$, and $\mathrm{d}Q = T^\top \mathrm{d}Z$) in registers without materializing $U$.
   - **Token-space & short-sequence suite**: Covers fused projections (`_biased_gemm`, `_masked_cast`), rank-space token passes (`_statistics`, `_chunk_statistics` summing across `2 <= tiles < 32` directly and dispatching `reduce_fwd` / `reduce_bwd` when `tiles >= 32` [$N > 8{,}192$], `_readout`, `_token_backward`), core/complement helpers (`core_forward`, `core_backward`, `complement`, `product`, `eta_trace`), and analytical $N \le r$ kernels (`short_stats`, `short_readout`, `short_bwd_prep`, `short_bwd_da`, `short_bwd_dc`), plus on-the-fly readout reconstruction for `recompute_readout`.
   - **Hopper `SM90a` Specializations**: On `SM90+` GPUs, token projections, statistics, readout, and token backward dispatch hardware TMA descriptors (`tl.make_tensor_descriptor` / `cp.async.bulk.tensor` when 16-byte aligned), native `M=64` transposed `WGMMA` (`TRANSPOSE_Q=True` accumulating $C^\top A$ in `[BD=64, BR]` layout when `BD >= 64` and `BR < 64`), and Programmatic Dependent Launch (`launch_pdl=True` / `griddepcontrol`).
   - **L2 Cache Persistence & Zero Register Spills**: Reused weights, biases, compact states, valid counts, and rotary tables carry `eviction_policy="evict_last"` hints while token activations keep default `.ca` L1+L2 caching with `GROUP_M` tile swizzling. Tile shapes and warp counts are tuned so every native CUDA C++ and Triton specialization across $r \in \{16, 32, 48, 64\}$ compiles with **`STACK: 0, LOCAL: 0`** (0 register spills) across both `SM80` and `SM90a`.

3. **Precision Design & Memory Lifetimes**:
   - Token projections and statistics multiply BF16 operands with FP32 accumulation, deferring $1/N$ ($G$) and $1/\sqrt{N}$ ($Q$) normalization to a single post-reduction scaling pass and evaluating $1/\sqrt{N}$ and $1/N$ via inline 1-step FMA Newton-Raphson refinement (`_short_rsqrt` / `_short_rcp`) with exact IEEE `sqrt.rn` / `div.rn` rounding (`0` `.approx` FP32 `sqrt`/`div`).
   - All compact $r \times r$ and $r \times d$ solves and products run in FP32 (using Triton's `tf32x3` three-pass IEEE FP32 emulation for Tensor Core dot products, paired with FP64-diagonal Cholesky and datacenter FP64 LU native solves).
   - Token readout and input VJPs split sensitive FP32 compact matrices ($W$, $\mathrm{d}G$, $\mathrm{d}Q$) into a high BF16 word and a residual BF16 word ([Henry et al., 2019](https://arxiv.org/abs/1904.06376)) and accumulate the low-order residual dot products first before adding the dominant high-word product, preventing floating-point absorption under strong-relation cancellation.
   - Memory lifetimes are tightly bounded: `recompute_readout` (`"auto"` default for $N > 512$) eliminates persistent storage of the `[B, N, D]` mixed-token readout tensor across forward/backward, dead intermediates are freed immediately, and `maybe_clear_saved_tensors` releases autograd saved tensors before allocating final token gradients (while respecting `retain_graph=True`).

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

By default (`MEMSOLVE_USE_MATHDX=OFF`), building the self-contained CUDA C++/cuBLASLt extension requires PyTorch 2.14, a matching CUDA toolkit ($\ge 12.0$), C++20, and CMake $\ge 3.25$ on `x86_64` or `aarch64` (`sbsa-linux`)—with **no external NVIDIA MathDx SDK required**:

```bash
PYTHON=python bash tools/build_cuda.sh
```

This builds a single unified multi-arch fatbinary `build/cuda/lib/memsolve_equilibrium.so` targeting `80;90a` by default (Ampere `SM80` + Hopper `SM90a` SASS plus embedded `compute_90` PTX, alongside `memsolve_equilibrium_sm80.so` and `memsolve_equilibrium_sm90.so` hardlinks). Set `CUDA_HOME` if your CUDA toolkit is in a non-default location, or override `MEMSOLVE_CUDA_ARCHITECTURES` with a semicolon-separated list from `80;86;87;89;90;90a;100;100a;103;103a;110;120;120a`.

To optionally compile with NVIDIA MathDx 26.06.1 (`cuSOLVERDx`, requiring CUDA 13.2 and `MATHDX_ROOT`), pass `MEMSOLVE_USE_MATHDX=ON`:

```bash
PYTHON=python MEMSOLVE_USE_MATHDX=ON MEMSOLVE_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh
```
