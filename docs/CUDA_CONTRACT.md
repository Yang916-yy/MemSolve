# CUDA contract

The current no-RankRotary operator supports `equilibrium` (default) and
`complement`. Model contract **26**, native ABI **17**, and the installed
PyTorch build must match. The source equations are owned by
[`reference.py`](../memsolve/ball/reference.py); CUDA never changes the model or
silently falls back to the reference implementation.

## Supported boundary

`MemSolve.forward(..., implementation="cuda")` accepts CUDA FP16/BF16 inputs,
FP32 parameters, ranks 16/32/48/64, and positive head dimensions. The public
output follows the input dtype. Boolean masks may contain gaps or completely
masked samples. Projection GEMM epilogues mask packed outputs and final outputs
after bias.
Inputs are sanitized before projection so invalid NaNs cannot enter weight
gradients. Normalization uses `max(valid_tokens, 1)`. Optional precomputed FP32
valid counts can be reused across layers.
These counts must equal `max(valid_tokens, 1)` for the same mask (globally
when sequence parallelism is active).
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
T = inverse(R^T)                 construct once; retain FP32
Z = T Q
F = solve(I+J, I-J)                equilibrium: one map per head
Delta = F Z                       equilibrium
Delta = 0                         complement
W = T^T (Delta - eta Z)
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

The ordinary branch has one inverse-factor forward/backward organization for
both modes, without rank- or mode-dependent solver selection. With lower
`L=R^T`, `M=F-eta I`, coefficient adjoint `E`, `U=T E`, and `H=U Z^T`:

```text
dF = sum_batch(H)
deta = direct_content_adjoint - trace(dF)
L^T dL = -M H^T - M^T H
dQ = T^T M^T U
dG = T^T sym_lower(L^T dL) T
```

Here `sym_lower(X)=(tril(X)+tril(X,-1)^T)/2`. H is computed once and reused
for all three adjoints, then overwritten by the symmetric lower projection;
no saved tensor is mutated. Complement sets F=0, so the projection simplifies
to the symmetric expression in H and eta and needs no learned-core VJP.
Its stored coefficient is `T^T Z` without -eta; shared token kernels apply
-eta after the compensated product, including eta=0 without division.
Both modes save Z and the coefficient. This common implementation uses more
saved content-state storage than the previous complement-only SPD VJP.

Equilibrium's shared core map adjoint sums across samples and reuses saved
partial-pivot LU factors in a transposed solve. For `F=(I+J)^-1(I-J)`, adjoint E
gives `U=(I+J)^-T E` and `dJ=-U(F+I)^T`. The short branch retains its PyTorch
autograd owner and the same implicit LU-backed core solve. Retained first-order
backward is supported; higher-order differentiation is not.

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
- Eta uses native tanh for its value and an exponential analytic derivative,
  including the derivative `1-eps` at zero. The value avoids cancellation in
  `1-exp(-2*abs(raw))` near zero; the derivative avoids saturation in
  `1-tanh(raw)^2` in the tails. The reference uses the same mathematical map
  with an explicit autograd derivative. Core-factor softplus uses libdevice
  `log1p(exp(x))`, retaining the reference's threshold at 20, rather than
  rounding `1+exp(x)` before taking its logarithm.
  Packed outputs/input gradients are BF16;
  parameter gradients remain FP32.

Triton handles tiled token statistics/readout/VJPs, compact products and
compact elementwise fusion. Small shared-core products use PyTorch library
operations. NVIDIA cuSOLVERDx supplies partial-pivot GETRF/GETRS, Cholesky POTRF,
and TRSM device operations. In the ordinary rank-space branch, one CTA
per system forms I+G, factors it once, and solves identity RHS panels while
the factor stays in shared memory. Only `T=chol(I+G)^-1` is written to global
memory: neither the identity nor the Cholesky factor is materialized there.
Both modes use this `compact_inverse` primitive, followed by FP32 products.
The inverse factor replaces the saved Cholesky factor, not an additional
persistent copy. `||T||_2 <= 1` follows from `I+G >= I`, but this bound alone
does not justify low-precision storage in cancellation-sensitive expressions.
The cancellation-sensitive short-token branch retains its existing owner.
cuBLASLt supplies the joint projection weight/bias gradient via its BGRADB
epilogue. The native interface exposes `contract_version`, `compact_lu`, `compact_getrs`,
`compact_inverse` and `linear_weight_bias`;
there is no full-mixer native forward/backward ABI. Direct native solves reject
autograd-enabled inputs because their derivatives belong to the Python owner.
Factorization status remains on GPU, without a CPU status poll.

For at most 256 physical tokens, statistics fit in one bounded token tile;
there is no intermediate Gram/cross-state reduction. Longer sequences retain
parallel partial statistics and reduction. Both modes share this schedule.
Shared-head maps are loaded directly by compact-product CTAs without expanding
them across the batch. Inverse-factor TRSM uses 32-column identity panels, 128 threads and the actual
rank (including rank 48); no RHS tensor or layout copy is materialized.
Biased large projections use 128x128 output tiles with eight warps; small
projections retain compact tiles. These are scheduling choices, not model modes.
Compact GEMMs use 16-wide inner tiles for odd multiples of 16 and otherwise
32-wide tiles. This avoids padding inner dimensions 16/48 to 32/64 while
retaining the same FP32/TF32x3 algorithm for every rank and both modes.

The input VJP at ranks 16/32 uses 128-token by 32-channel tiles, trading
channel width for token reuse. Ranks 48/64 retain 32-token by 64-channel tiles
to limit live register state. Nonmultiple tails are masked. Compact shared-map
products fuse the subtraction of eta times state into their FP32 epilogue.
The shared-map adjoint reduces the already-needed H across samples. Its
[B,H,r,r] storage is temporary and reused for the Cholesky adjoint. The
symmetric lower projection is fused with H's transformation; two compact
products apply the saved inverse factor on either side. Gram/cross statistics,
shared-core temporaries, and compact adjoints are released after their last
use. Saved tensors required for repeated backward are never overwritten.
The fused Cholesky adjoint accumulates its two products serially with 16-wide
inner tiles, four warps and one stage. This shortens operand lifetimes; SM80
resource checks at ranks 16/32/48/64 show no register spills in this kernel or
the tested compact GEMMs. Larger warp counts were slower in module benchmarks
and are not retained. Other architectures still need resource/performance
validation.

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

## Exact sequence parallelism

From the source checkout, `integrations.distributed.SequenceParallel` provides
an explicit layout and gradient-SUM helper. Initialize `torch.distributed`,
place identical model weights on each rank, and shard consecutive tokens of
the **same samples** across the sequence group. Uneven nonempty shard lengths
are supported; batch/head/channel dimensions must agree. The global sequence
must exceed rank, although individual shards may be shorter than rank.

```python
from integrations.distributed import SequenceParallel

sp = SequenceParallel.create(local_x.shape[1], group=sequence_group)
# Compute once per batch; reuse across layers. Clamp happens after global SUM.
counts = sp.valid_counts(local_mask, batch_size=local_x.shape[0], device=local_x.device)
y = layer(local_x, local_mask, implementation="cuda",
          sequence_parallel=sp, valid_counts=counts)
# Normalize by the GLOBAL target/token count, not each shard's local count.
local_loss.backward()
sp.reduce_gradients_(layer)  # once after accumulation, before clipping/step
optimizer.step()
```

Each rank computes local G/Q contributions; one packed SUM collective forms
the global sufficient statistics. Compact systems are redundantly solved on
each rank. Readout stays local. Backward uses another compact SUM collective;
input gradients include cross-shard coupling. Core/eta adjoints are divided by
the sequence world size because they already contain all shards, while
projection adjoints remain local. The final **SUM**, not average, of replicated
parameter gradients then produces the unsharded derivative. Do not also apply
DDP averaging over the same sequence group. An independent data-parallel group
may average separate examples. Every trainable parameter passed to the helper
must participate; freeze unused parameters. Zero gradients before the next
accumulation window, and call the helper only once per window.

The context stores shape metadata and the process group, not activations or
parameter/gradient tensors. Create it before Graph capture. Counts and the
layout may be reused when valid; tensor masks/counts must be updated when the
batch changes. Forward, backward, and parameter SUM support CUDA Graph capture
after synchronized warmup on the capture stream. NCCL/PyTorch manage collective
stream dependencies; the implementation does not poll GPU values on the CPU.
Reset or release captured CUDA Graphs before `destroy_process_group()`, including
exception cleanup: NCCL graphs retain communicator references until released.

`sp.halo(x, width=1)` exchanges only boundary tokens with differentiable PyTorch
functional collectives. `_CenteredShortConv(..., sequence_parallel=sp)` uses
this halo before native Conv1d, masks before/after, and pads only the global
ends. Full BERT Trainer entrypoints are not automatically sharded: callers
must also provide global absolute position offsets, globally normalized losses,
and the parallel context through their encoder. Operator and short-convolution
contracts are tested independently against an unsharded FP64 oracle.

The core collective payload is O(BH(r²+rd)), independent of token count;
parameter gradient reduction and convolution halos are additional. This is
intended for long sequences. Communication makes N=128 slower on two A800s;
it should not replace data parallelism in the current short MLM recipe.

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
requirements above 2.31. All earlier native artifacts must be rebuilt for ABI 17;
historical training source snapshots keep their own binaries and remain separate.

## Upstream sources and precision decisions

- [Selective triangular inversion (IPDPS 2017)](https://arxiv.org/abs/1612.01855):
  reuse an inverse factor to expose parallel GEMMs instead of repeatedly
  traversing triangular dependencies. Both modes use this organization.
- [Cholesky differentiation (Murray)](https://arxiv.org/abs/1602.07527):
  organize the compact VJP around the symmetric lower projection. MemSolve's
  reuse of H across core, eta, and Gram adjoints follows its own algebra.
- [cuBLASDx dependent GEMM fusion](https://docs.nvidia.com/cuda/cublasdx/examples.html):
  informed a larger compact-chain fusion prototype. It regressed wide ranks
  on SM80, so only profitable local fusion is retained.
- [TSQR](https://arxiv.org/abs/0806.2159): an equivalent positive-diagonal tree QR
  of the relation blocks, adding the identity regularizer once, was tested.
  The mature PyTorch QR prototype was slower than the tiled Gram/inverse path;
  it is not a second production algorithm.

- [NVIDIA cuSOLVERDx Cholesky composition](https://docs.nvidia.com/cuda/cusolverdx/get_started/posv.html):
  factor once and solve RHS panels in the same shared-memory lifetime.
- [cuBLASLt plan reuse](https://docs.nvidia.com/cuda/cublas/index.html#heuristics-cache):
  query an algorithm once and refresh runtime pointers at execution.
- [NVIDIA cuBLASLt epilogues](https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t):
  BGRADB computes projection bias gradients alongside the weight-gradient GEMM.
  The implementation calls the library API rather than copying its GEMM kernels.
- [NVIDIA cuSOLVERDx TRSM](https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/trsm.html)
  and the installed MathDx 26.06.1 `07_BLAS/trsm_batched_block.cu` example:
  library-owned triangular arithmetic with cooperative load/solve/store and
  implicit identity RHS panels. Both modes use the resulting inverse factor
  for forward and backward products. No hand-written triangular solver is
  introduced; the superseded standalone solve/adjoint interfaces were removed.
- [FLA chunk output](https://github.com/fla-org/flash-linear-attention/blob/78a39922d56562336e22e5a429b039f34ef2bf7f/fla/ops/common/chunk_o.py)
  and [compact triangular products](https://github.com/fla-org/flash-linear-attention/blob/78a39922d56562336e22e5a429b039f34ef2bf7f/fla/ops/utils/solve_tril.py):
  bounded tiles, independent CTA scheduling, selective local fusion and explicit
  product precision. The kernel organization is adapted to MemSolve's global
  statistics; its noncausal operator is not replaced by an FLA recurrence.
- [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html):
  tiled Tensor Core products with FP32 accumulators and fused epilogues.
- [Triton libdevice integration](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html):
  use mature tanh/log1p primitives for compact scalar maps. Their SM80 module
  forward/backward benchmarks changed latency by less than 0.7% in either
  direction and did not change peak allocated memory. Explicitly rounded
  normalization and deferred readout compensation were also tested; neither
  consistently improved full-operator error, so neither is adopted.
- [Mesa precision repair](https://github.com/fla-org/flash-linear-attention/pull/1152)
  and [small-gradient issue](https://github.com/fla-org/flash-linear-attention/issues/1283):
  precision follows each tensor's range and numerical role. Mesa's normalized
  Q/K can use FP16, while V-containing state and small gradients need the BF16
  exponent range, and sensitive systems stay FP32. MemSolve has no normalized
  Q/K and does not materialize bounded P. Its BF16 relation/content/gradient
  streams are unbounded, while its bounded compact maps already use FP32.
  No BF16-to-FP16 conversion is adopted merely by analogy. Regression tests
  include loss scales from 2^-24 to 2^12 without relaxing error tolerances.
  FP16 inverse-factor storage passed ordinary full-operator checks but failed
  the existing tighter compact-system oracle. FP32 storage is retained.

Model equations, initialization, parameter counts and checkpoint contract 26
are unchanged. Native ABI 17 requires rebuilding the extension. Measurements
and register-resource checks for this update cover SM80/A800 only.
