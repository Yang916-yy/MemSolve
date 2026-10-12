# CUDA contract

The current no-RankRotary operator supports `equilibrium` (default) and
`complement`. Model contract **26**, native ABI **16**, and the installed
PyTorch build must match. The source equations are owned by
[`reference.py`](../memsolve/ball/reference.py); CUDA never changes the model or
silently falls back to the reference implementation.

## Supported boundary

`MemSolve.forward(..., implementation="cuda")` accepts CUDA FP16/BF16 inputs,
FP32 parameters, ranks 16/32/48/64, and positive head dimensions. The public
output follows the input dtype. Boolean masks may contain gaps or completely
masked samples. Packed projections are masked before statistics, normalization
uses `max(valid_tokens, 1)`, and the final output is masked after its bias.
No-skew and no-complement component ablations are reference-only.

`cuda.fast_mix(projected, core_base_raw, eta_raw, valid_counts=None)` accepts
contiguous CUDA BF16 `[B,N,H*R+D]` packed inputs. The base is FP32 `[H,R,R]`
for equilibrium or empty `[H,R,0]` for complement; eta is FP32 `[H]`.
Optional counts are contiguous FP32 `[B]`, non-differentiable metadata.
Direct callers must provide positive finite counts, zero masked packed rows,
and mask invalid upstream gradients. The model wrapper handles this contract.
Unsupported shapes, layouts and dtypes raise errors.

## Computation without token-sized P

Both modes share statistics, readout and input-gradient kernels. With normalized
relation A, content C, and `J = L L^T + Omega`, the ordinary rank-space path is

```text
G = A^T A
Q = A^T C                        compact cross-statistic, not a query projection
R^T R = I + G
Z = solve(R^T, Q)
F = solve(I+J, I-J)                equilibrium: one map per head
Delta = F Z                       equilibrium
Delta = 0                         complement
W = solve(R, Delta - eta Z)
Y = eta C + A W
```

This is exactly the reference with `P=A R^-1`. F is rebuilt every forward,
shared across samples, and never cached across optimizer updates. Complement
omits core construction and LU entirely. No content-conditioned generator or
materialized-frame CUDA implementation remains.

For physical `N <= r`, the base uses the smaller token system
`eta solve(I+A A^T, C)`. It avoids subtracting nearly equal terms when strong
relations span a short sequence. The learned correction still uses the compact
rank system. This is an algebraic shape choice with systems no larger than
64-by-64, not an empirical mode or performance heuristic. Neither branch
constructs P or an unbounded N-by-N attention matrix.

The ordinary branch has an explicit analytic first-order VJP. Complement uses
`H=solve(I+G, Q)` and differentiates this SPD solve directly, rather than
backpropagating through intermediate Cholesky coordinates. For the adjoint E
of the mathematical coefficient `W=-eta H`, set `T=solve(I+G, E)`; then
`dQ=-eta T`, `dG=eta sym(T H^T)`, and subtract `<E,H>` from the eta adjoint.
Here `sym(X)=(X+X^T)/2`. This is the same derivative as the original formula
and keeps the token VJP shared with equilibrium. Equilibrium's shared core
map adjoint sums across samples and reuses saved partial-pivot LU factors in a
transposed solve. For `F=(I+J)^-1(I-J)`, adjoint E gives
`U=(I+J)^-T E` and `dJ=-U(F+I)^T`. Cholesky and triangular-solve VJPs stay in
compact space. The short branch retains its PyTorch autograd owner and the same
implicit LU-backed core solve. Retained first-order backward is supported;
higher-order differentiation is not.

## Precision and scheduling

- Token projections/statistics use BF16 multiplicands with FP32 accumulation.
  Projection biases are added in FP32 before the final cast.
- Compact Gram matrices, Cholesky, LU, triangular solves, core maps and their
  adjoints are stored/accumulated in FP32. Batched compact products use Triton's
  explicit `tf32x3` decomposition, independently of ambient TF32 policy. The
  small shared-generator products and factorization/solve arithmetic retain
  IEEE FP32. This does not round compact state to BF16 or FP16.
- Token readout and input VJPs split sensitive FP32 compact operands into a
  BF16 leading word and residual word, accumulating both products in FP32.
  Rounding these operands to a single BF16 word fails the strong-relation
  fixture. This follows the multiword arithmetic idea of
  [Henry et al.](https://arxiv.org/abs/1904.06376).
- Eta uses the tail-safe interior-tanh value and analytic derivative, including
  the derivative `1-eps` at zero. Packed outputs/input gradients are BF16;
  parameter gradients remain FP32.

Triton handles tiled token statistics/readout/VJPs, compact products and
compact elementwise fusion. Small shared-core products use PyTorch library
operations. NVIDIA cuSOLVERDx supplies partial-pivot GETRF/GETRS, Cholesky POTRF,
and batched TRSM/POTRS. In the ordinary rank-space branch, one CTA per system
forms I+G, factors it once, and solves RHS panels while the factor stays in
shared memory. Equilibrium composes POTRF with its first TRSM; complement
composes POTRF with POTRS. The factor is saved once for the analytic backward.
The cancellation-sensitive short-token branch retains its existing owner.
cuBLASLt supplies the joint projection weight/bias gradient via its BGRADB
epilogue. The native interface exposes `contract_version`, `compact_lu`, `compact_getrs`,
`compact_trsm`, `compact_potrs`, `compact_factor_solve`,
`compact_cholesky_backward` and `linear_weight_bias`;
there is no full-mixer native forward/backward ABI. Direct native solves reject
autograd-enabled inputs because their derivatives belong to the Python owner.
Factorization status remains on GPU, without a CPU status poll.

For at most 256 physical tokens, statistics fit in one bounded token tile;
there is no intermediate Gram/cross-state reduction. Longer sequences retain
parallel partial statistics and reduction. Both modes share this schedule.
Shared-head maps are loaded directly by compact-product CTAs without expanding
them across the batch. TRSM uses 32-column RHS panels, 128 threads and the actual
rank (including rank 48); dense row/column-major inputs avoid layout copies.
Biased large projections use 128x128 output tiles with eight warps; small
projections retain compact tiles. These are scheduling choices, not model modes.

The input VJP at ranks 16/32 uses 128-token by 32-channel tiles, trading
channel width for token reuse. Ranks 48/64 retain 32-token by 64-channel tiles
to limit live register state. Nonmultiple tails are masked. Compact shared-map
products fuse the subtraction of eta times state into their FP32 epilogue.
The shared-map adjoint sums products across batch/channel blocks before its
final reduction; it does not allocate a full [B,H,r,r] product.

The Cholesky adjoint applies the symmetric lower-triangle projection and both
triangular solves in one CTA. cuSOLVERDx left/transposed and right/nontransposed
TRSM reuse the same shared-memory factor, returning a contiguous FP32 Gram
adjoint without a global intermediate or transpose copy. Compact-product
epilogues apply negation, symmetric averaging, or subtraction into an unsaved
adjoint. Eta's inner-product VJP uses bounded FP32 partial reductions rather
than materializing the full batch/head/rank/channel elementwise product.

Complement saves H=(I+G)^-1 Q only once, without a second -eta H tensor. The
readout uses eta C - eta A H, and the relation VJP applies -eta to its H product.
This is algebraically the same coefficient readout, with FP32 scaling after the
compensated product; no division by eta is used, including at eta=0. Short-lived
Gram/cross statistics and coefficient adjoints are released after their last
use. Saved tensors needed for repeated backward are not mutated or cleared.

The CUDA projection fuses weight and bias gradients when both consume the same
BF16 upstream tensor and both parameters require gradients. FP16/FP32 upstream
bias gradients retain their original values; they are not rounded to BF16 to
enable fusion. The library GEMM uses BF16 inputs and FP32 accumulators, outputs,
and bias reduction, regardless of ambient AMP/reduced-precision settings. Its
workspace is allocator-owned and CUDA Graph compatible. Host plans are cached
per thread/device/shape/input alignment, with a bounded 256-entry cache. Each
call refreshes the bias pointer; captured graphs retain their own launch
arguments, and the cache owns no tensor storage. The workspace limit remains
32 MiB; allocation uses the selected algorithm's actual requirement. Reducing
that limit did not preserve performance across tested layouts.
Reference projections remain independent of this native optimization. Forward
projection bias fusion follows the
[Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
Upstream attribution is retained in [NOTICE](../NOTICE).

Warm up each forward/backward shape on a side stream before CUDA Graph capture.
Replay reads current inputs and parameters; saved factors are not mutated by
backward. Graph removes repeated host scheduling, not GPU arithmetic or graph
pool storage. See [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs).

## Numerical validation and remaining boundary

The FP64 reference is the oracle. Standard deterministic fixtures require
forward relative L2 error <=0.5% and input/parameter gradient error <=3%, with
finite outputs. Narrow-head and eta-tail gradient checks retain a 1% budget.
Tests cover all supported ranks, masks, short/long shapes, nonzero learned
cores, and Graph replay after input/parameter updates. These fixture budgets
are not universal bounds on arbitrarily ill-conditioned problems.

An existing extended high-amplitude fixture has a 1% forward budget, predating
this mode cleanup. Tolerances are not loosened by removing the generator or
renaming the retained modes.

TODO: when physical N exceeds rank but only a few tokens are valid, strong
relations can cause cancellation in the rank-space complement. A historical
BF16 probe with N=197, r=32, one valid token and relation scale 64 had 9.93%
relative output error but maximum absolute error 5.97e-7; at scale 8 the relative
error was 0.187%. This applies to the shared base in both modes. Revisit its
stable evaluation and gradients before very sparse-mask workloads. Current
224px mean-pooling image models use 196 valid patches. No sparse-mask threshold,
FP64 production path, or fallback has been introduced.

## Build and runtime artifacts

The local build uses PyTorch 2.14.0+cu132, CUDA 13.2, MathDx 26.06.1, C++20 and
CMake >=3.25. Triton is supplied by the matching PyTorch environment.

```bash
PYTHON=python MEMSOLVE_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh
```

Run this command in the intended Python environment. `CUDA_HOME` defaults to
`/usr/local/cuda-13.2`; `MATHDX_ROOT` defaults to
`/opt/nvidia/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06`. Override them for
other installations. `MEMSOLVE_CUDA_ARCHITECTURES` accepts a semicolon-separated
list; the build script defaults to all seven targets when it is unset.

Device LTO builds one executable image per target. The loader chooses
`memsolve_equilibrium_sm<arch>.so` and enforces a matching architecture per
process. Targets are SM80/86/87/89/90/100/120 (SM121 selects SM120); this update
was built and exercised on SM80/A800. Other targets require hardware validation.
Mixed architectures in one process are rejected. Native kernels verify the
compiled architecture before launch.

`tools/package_cuda_runtime.py` requires all seven artifacts and records exact
source/native versions, Torch/CUDA versions, and the C++ ABI. Packaged artifacts
must match these checks. It removes build-host RPATH/RUNPATH and rejects GLIBC
requirements above 2.31. All earlier native artifacts must be rebuilt for ABI 16;
historical training source snapshots keep their own binaries and remain separate.

## Upstream sources and precision decisions

- [NVIDIA cuSOLVERDx Cholesky composition](https://docs.nvidia.com/cuda/cusolverdx/get_started/posv.html):
  factor once and solve RHS panels in the same shared-memory lifetime.
- [CUTLASS parallelized reductions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#parallelized-reductions):
  partition the shared-map gradient's combined batch/channel reduction while
  keeping enough CTAs; no CUTLASS kernel source is vendored for this operation.
- [cuBLASLt plan reuse](https://docs.nvidia.com/cuda/cublas/index.html#heuristics-cache):
  query an algorithm once and refresh runtime pointers at execution.
- [NVIDIA cuSOLVERDx POTRS](https://docs.nvidia.com/cuda/cusolverdx/get_started/potrs.html):
  two triangular solves within one shared-memory lifetime for complement.
- [NVIDIA cuBLASLt epilogues](https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t):
  BGRADB computes projection bias gradients alongside the weight-gradient GEMM.
  The implementation calls the library API rather than copying its GEMM kernels.
- [NVIDIA cuSOLVERDx TRSM](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/trsm.html)
  and the installed MathDx 26.06.1 `07_BLAS/trsm_batched_block.cu` example:
  library-owned triangular arithmetic with cooperative load/solve/store and
  RHS panels. The Cholesky adjoint composes left/transposed and
  right/nontransposed solves with the same shared-memory factor. No new
  hand-written triangular solver is introduced.
- [FLA chunk output](https://github.com/fla-org/flash-linear-attention/blob/78a39922d56562336e22e5a429b039f34ef2bf7f/fla/ops/common/chunk_o.py)
  and [compact triangular products](https://github.com/fla-org/flash-linear-attention/blob/78a39922d56562336e22e5a429b039f34ef2bf7f/fla/ops/utils/solve_tril.py):
  bounded tiles, independent CTA scheduling, selective local fusion and explicit
  product precision. The kernel organization is adapted to MemSolve's global
  statistics; its noncausal operator is not replaced by an FLA recurrence.
- [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html):
  tiled Tensor Core products with FP32 accumulators and fused epilogues.
- [Mesa precision repair](https://github.com/fla-org/flash-linear-attention/pull/1152)
  and [small-gradient issue](https://github.com/fla-org/flash-linear-attention/issues/1283):
  precision follows each tensor's range and numerical role. Mesa's normalized
  Q/K can use FP16, while V-containing state and small gradients need the BF16
  exponent range, and sensitive systems stay FP32. MemSolve has no normalized
  Q/K and does not materialize bounded P. Its BF16 relation/content/gradient
  streams are unbounded, while its bounded compact maps already use FP32.
  No BF16-to-FP16 conversion is adopted merely by analogy. Regression tests
  include loss scales from 2^-24 to 2^12 without relaxing error tolerances.

Model equations, initialization, parameter counts and checkpoint contract 26
are unchanged. Native ABI 16 requires rebuilding the extension. Measurements
and register-resource checks for this update cover SM80/A800 only.
