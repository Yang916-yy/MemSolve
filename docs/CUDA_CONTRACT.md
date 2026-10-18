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
gradients. Input sanitization and BF16 conversion share one pass. The input
projection's VJP applies its mask in the GEMM store, eliminating a separate
masked-gradient tensor. This evaluates the same `D_out (X_safe W^T + b)`
and its VJP; invalid inputs and upstream gradients are still sanitized before
weight-gradient GEMMs. FP16/FP32 bias gradients retain their original precision.
Normalization uses `max(valid_tokens, 1)`. Optional precomputed FP32
valid counts can be reused across layers.
These counts must equal `max(valid_tokens, 1)` for the same mask (globally
when sequence parallelism is active).
No-skew and no-complement component ablations are reference-only.

`cuda.fast_mix(projected, core_base_raw, eta_raw, valid_counts=None, ...)` accepts
contiguous CUDA BF16 `[B,N,H*R+D]` packed inputs. The base is FP32 `[H,R,R]`
for equilibrium or empty `[H,R,0]` for complement; eta is FP32 `[H]`.
Optional counts are contiguous FP32 `[B]`, non-differentiable metadata.
Direct callers must provide positive finite counts, zero masked packed rows,
and mask invalid upstream gradients. The model wrapper handles this contract.
Unsupported shapes, layouts and dtypes raise errors.

Optional `rotary_cos` and `rotary_sin` are a matched pair of contiguous FP32
tables `[1 or B,N,head_dim/2]` on the same device, without gradients. The model
constructs them from its RoVE configuration and position IDs. Its optional
`rotary_factors=(cos, sin)` argument reuses these tables instead of accepting
position/context inputs; BERT shares them within one forward call. Rotary head
widths must be even; positions and frequency scaling are defined in the
[core contract](CORE_CONTRACT.md#optional-rotary-value-encoding).

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

With RoVE, the cross-statistic becomes `Q=A^T rot(C)`, while Gram, the compact
solve and shared-core map are unchanged. Readout uses
`Y=eta C+invrot(A W)`. Backward similarly rotates the upstream content gradient
for cross-statistics and inversely rotates the indirect content gradient.
The direct eta adjoint remains the unrotated inner product of upstream and C.
Rotations are fused into token kernels and evaluated in FP32; rotated content
is not rounded into a token-sized BF16 buffer. Products involving a rotated
FP32 operand use the same multiword BF16 strategy as sensitive compact products.
The short-token branch applies the equivalent rotation around its stable
token-system solve. Native compact primitives and ABI 17 are unchanged.

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

Short statistics use at most 256 tokens per tile without RoVE, or 128 with
RoVE to limit live FP32 rotation operands. At N >= 512, when the resulting grid
supplies at least two CTAs per SM, each CTA accumulates 512 tokens in 64-token
steps. Otherwise the existing parallel partial-statistics schedule is used.
This regrouping computes the same Gram, cross-statistic and direct eta adjoint;
it changes only FP32 summation order. RoVE high/low-word products retain separate
FP32 accumulators. At lengths divisible by 512, the long schedule reduces the
partial buffers fourfold relative to 128-token tiles. Both core modes use the
same device-aware schedule, including exact sequence-parallel local statistics.
Shared-head maps are loaded directly by compact-product CTAs without expanding
them across the batch. Inverse-factor TRSM uses 32-column identity panels, 128 threads and the actual
rank (including rank 48); no RHS tensor or layout copy is materialized.
Projection GEMMs use Inductor-derived grouped scheduling and contiguous-load
hints. Large projections use 128x128 output tiles with four warps: inner tiles
are 64 wide with three stages for K >= 512, otherwise 32 wide with four stages.
Small projections use 64x64x32 tiles, four warps and three stages. Tile choices
are fixed before launch; there is no runtime autotuning inside CUDA Graphs.
All schedules compute the same AB+bias with FP32 accumulation and bias addition,
apply the token mask, then round once at the output store. There is no dense
FP32 projection scratch or change to the saved tensors/backward contract.
See `THIRD_PARTY_NOTICES.md` for the upstream source and license.
Compact GEMMs use 16-wide inner tiles for odd multiples of 16 and otherwise
32-wide tiles. This avoids padding inner dimensions 16/48 to 32/64 while
retaining the same FP32/TF32x3 algorithm for every rank and both modes.

Without RoVE, the input VJP at ranks 16/32 uses 128-token by 32-channel tiles;
ranks 48/64 use 128-token by 16-channel tiles. RoVE uses 128-token by 16-channel
tiles at all four ranks. Its three BF16 products reuse one FP32 accumulator.
This reduces the register spills introduced by fused rotations, without changing
the mathematical terms or their precision. RoVE's direct relation-gradient term
is computed before loading the content
and cross-adjoint operands. This shortens their overlapping live ranges: on
SM80, r48/r64 token-VJP spill counts drop from 66/62 to 26/24 at 128 tokens,
64 content channels and four warps. Remaining spills are measured rather than
eliminated by indiscriminately increasing warp count. Nonmultiple tails are masked. Compact shared-map
products fuse the subtraction of eta times state into their FP32 epilogue.
The shared-map adjoint reduces the already-needed H across samples. Its
[B,H,r,r] storage is temporary and reused for the Cholesky adjoint. The
symmetric lower projection is fused with H's transformation; two compact
products apply the saved inverse factor on either side. Gram/cross statistics,
shared-core temporaries, and compact adjoints are released after their last
use. The compact VJP reuses its fresh U buffer for the state and cross adjoints,
and its fresh H buffer for both Gram-adjoint products. These GEMMs read each
CTA's complete input region before storing: right-buffer reuse requires a square
left map; left-buffer reuse is limited to the single-CTA square output. Neither
saved inverse factors nor saved states/core tensors are overwritten.
Autograd saved-tensor ownership is released with `maybe_clear_saved_tensors`,
following PyTorch AOTAutograd; `retain_graph=True` keeps the saved tensors intact.
Projection backward computes weight/bias gradients before allocating the input
gradient, releasing the saved input after its last use. The mixer drops its
local compact tape before allocating the token gradient. Repeated backward and
input/parameter version counters are covered by contract tests.
Inference releases the unused backward tape before allocating token readout;
training retains it unchanged.
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
must also provide globally normalized losses and the parallel context through
their encoder. RoVE automatically uses global shard offsets; masked shards
beyond the training window also require explicit global context lengths.
An absolute-position encoder needs global position offsets at its embedding
layer. Operator and short-convolution
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
- [Mamba SSD chunk-state](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_state.py):
  informed the chunk-local K-loop for long-sequence statistics. The implementation
  is independently written for MemSolve's Gram/cross/eta equations; it does not
  import Mamba's decay, causality or low-precision state storage.
  FLA-inspired separate relation/content VJPs and channel-tiled readout were also
  tested. Their additional scheduling did not give a consistent full-operator
  gain, so the fused VJP and readout remain. Narrower fused VJP channel tiles
  improved wide ranks without adding a kernel.
- [Inductor shared-A BMM](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/templates/triton_bmm_shared_a.py.jinja):
  packing several samples behind a shared core improved individual wide-rank
  products but regressed r32. Full-operator gains were small, so this prototype
  is not retained. Shared maps continue to be loaded without batch expansion.
- [PyTorch AOTAutograd lifecycle](https://github.com/pytorch/pytorch/blob/main/torch/_functorch/_aot_autograd/runtime_wrappers.py)
  and [autograd's conditional release](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/python_function.cpp):
  release saved-tensor ownership during ordinary backward, while honoring
  `retain_graph`. No unconditional clear-on-access or donated saved buffer is used.
- Low-word-first and four-product BF16 compensation were compared against FP64.
  The former gave negligible accuracy benefit and more spills; the latter reduced
  internal FP32 error but slowed the token VJP with almost no change after BF16
  output rounding. Neither replaces the existing three-product rotated dot.
  Direct reciprocal-count normalization compiled to the same tested result;
  no additional precision gain is claimed from that rewrite.
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

The unrotated model retains its equations, initialization, parameter counts and
checkpoint contract 26. RoVE adds explicit transport settings to that contract
without changing the native ABI. Artifacts older than ABI 17 must be rebuilt.
Measurements and register-resource checks cover SM80/A800 only.


## Measured memory and resource changes

On 2026-10-18, A800/SM80 tests used Torch 2.14.0+cu132 and Triton 3.8.0,
BF16 inputs and FP32 parameters. The following are **incremental peak allocated
memory for one full MemSolve module forward/backward**, including both
projections, not whole-Transformer memory. B=512, N=128, width=768, heads=12;
all rows use equilibrium. The comparison isolates the lifecycle/scratch-reuse
pass after the projection and long-statistics optimizations were already present.

| Rank | RoVE | Before | After |
| ---: | --- | ---: | ---: |
| 32 | off | 676 MiB | 604 MiB |
| 32 | on | 772 MiB | 700 MiB |
| 64 | off | 1062 MiB | 966 MiB |
| 64 | on | 1158 MiB | 1062 MiB |

Complement showed similar savings. Across 24 module cases, outputs were bitwise
unchanged; unrotated gradients were also bitwise unchanged. With RoVE, maximum
input/parameter-gradient relative L2 change was 0.0037%, within the existing
FP64 oracle limits. Graph timing used 60 warmup replays and seven alternating
20-replay rounds. RoVE cases reduced module training latency by roughly
0.6–3.3%; these measurements are not whole-network throughput claims.
The corresponding r48/r64 RoVE token-VJP spill counts fell from 66/62 to 26/24.

The preceding long-statistics pass reduced Gram/cross partial buffers fourfold
when the 512-token schedule applies. For B=8, N=4096, heads=12, rank=32 and
64 content channels, those forward partials decreased from 36 to 9 MiB.
Other stages can still determine the full-module peak. No storage dtype or
compensation term was removed to obtain these savings. Detailed probes and
raw training artifacts remain outside the repository.
