# CUDA Contract

The CUDA runtime implements the unrotated operator in three core modes.
The native RHS-tiled component uses contract version 11:

- `core_mode=DYNAMIC`, `STATIC`, or `ZERO`;
- `skew_coupling=True` and `scalar_complement=True`;
- learned per-head one-ULP interiorized tanh complement;
- the soft frame, shared compact state, dynamic accretive generator, and
  direct equilibrium readout defined by `lsso/ball/reference.py`.

No-skew and no-complement are PyTorch-only ablations. CUDA rejects
them explicitly. It never falls back to the reference implementation, retains
an old ABI, or redefines the operator mathematics outside `reference.py`.

## Build environment

The current native build targets PyTorch 2.14.0+cu132, CUDA Toolkit 13.2,
MathDx 26.06.1 (CUDA 13 package), and C++20. CMake 3.25 is sufficient
for consuming the installed MathDx package with device LTO; building MathDx
itself has different requirements. See NVIDIA
[installed-package requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html).
The projection runtime is Triton 3.8.0 supplied by this PyTorch environment.
Use `LSSO_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh` for an A800.
The current unrotated implementation uses native ABI 11 and checkpoint contract 13.
Historical rotated checkpoints and results retain their original contracts.

## Public And Native Boundaries

`LSSO.forward(..., implementation="cuda")` is an explicit request. It accepts
CUDA activations in `torch.float16` or `torch.bfloat16`. Both projections use
BF16 multiplicands and FP32 accumulation. The input projection produces a
contiguous BF16 packed tensor; the native mixer returns BF16 coordinates.
The output projection adds its bias in FP32 and returns `x.dtype`.
Parameters remain FP32. FP32/FP64 public inputs are reference-only.

The native `projected` input must be contiguous CUDA BF16 with shape
`[B, N, H * R + D]`. FP16 and FP32 packed coordinates are rejected.
`core_base_raw`, `core_drive_weight`, and `eta_raw` must
also be contiguous FP32 CUDA tensors on the same device. When the public caller supplies a
`valid_mask`, it zeros masked packed coordinates and supplies contiguous FP32
`valid_counts[B] = max(n_valid, 1)`; this is also metadata and is not
differentiable. The native ABI rejects unsupported dtypes, layouts, shapes,
ranks, and devices explicitly;
`lsso.ball.cuda.fast_mix()` rejects gradient-bearing metadata before dispatch.
It does not interpret a token mask itself: direct `fast_mix()` callers must
zero padded packed coordinates, provide strictly positive finite valid counts,
and mask invalid upstream output gradients.

The supported native shape range is rank in `{16, 32, 48, 64}` and any positive
head dimension within ordinary CUDA allocation and launch limits. Rank and head
dimension are independent; rank may exceed head dimension. The public CUDA path
accepts `valid_mask[B, N]`,
including gapped padding and an
entirely masked sample. Masking retains the one generic physical `[B, N]`
schedule: invalid coordinates are zero and every per-sample normalization uses
`sqrt(n_valid)`. It is therefore a semantic input of the current operator, not
a dense-only scheduling restriction or a fallback path.

The private native ABI exposes:

- `forward_inference`, which returns BF16 pre-output coordinates;
- `forward_train`, which returns those coordinates plus a private FP32 tape and
  one-based `int32` LU pivots, matching MathDx partial-pivot LU factors;
- `backward`, which consumes the tape through the first-order autograd
  boundary owned by `lsso.ball.cuda.fast_mix()`. Its contiguous upstream
  tensor is BF16; kernels widen individual loads to FP32. FP32 upstream
  tensors are rejected by the native ABI.

`forward_inference` and `forward_train` are not differentiable public
operators. Direct use with autograd-enabled inputs raises an error, and
higher-order gradients are unsupported.

## Numerical Contract

`reference.py` owns the mathematical and canonical numerical definition. The
same reference path run with FP64 tensors is the oracle for CUDA validation.
CUDA uses a fixed mixed BF16/FP16/FP32 contract:

- token projections and token-statistics GEMMs use BF16 multiplicands and
  FP32 accumulators; compact factors/products/solves remain FP32. PyTorch
  reduced-precision BF16 reductions are disabled inside projection operations;
- Gram/factorization, one-ULP interiorized eta, LU and triangular
  solves, sensitive backward statistics, and parameter gradients remain FP32;
- packed activations, native output and packed-input gradients are BF16.

A finite bound alone is insufficient to justify FP16: eta must retain its
strict interior margin, and factor/solve state remains sensitive to rounding.
No data-dependent format switching, clipping, or recovery fallback is added.
BF16 extends the representable range but does not make arbitrary scales safe;
FP16 public outputs and input gradients still have FP16 range limits.

The factor Gram is algebraically still `F F^T`; native CUDA evaluates it with
FP32 FMA rather than quantizing `F` to BF16 first. This removes the dominant
small-matrix factor rounding error without adding a public numerical variant.

The complement's scalar backward reduction reconstructs its frame-content
term in FP32 from the recorded frame. This avoids narrow-head cancellation
from the BF16 compact-state tape; it changes neither the forward result nor
the Tensor Core boundaries of the compact operator.

The accepted CUDA accuracy envelope against the FP64 oracle is relative L2
error at most `5e-3` for forward outputs and at most `3e-2` for input and
parameter gradients; all compared tensors must be finite. Tests do not require
pointwise FP32 equality or finite-difference agreement from the reduced
precision path. The general gradient limit is relaxed from 1% to 3%
for BF16 rounding: the expanded native shape sweep measured up to 1.418%,
and the reference ZERO-mode fixture measured 2.357%. The forward limit remains
0.5%; narrow-head and complement-tail eta checks retain their 1% limits.
These are deterministic fixture budgets, not universal error bounds.

### TODO: cancellation with very few valid tokens

The rank-space no-P readout still evaluates the shared base as
`eta C - eta A (I + A^T A)^-1 A^T C`. With padded length greater than rank,
very few valid tokens, and large relation features, its two terms can nearly
cancel. This affects the base in all three modes; removing materialized P
does not eliminate finite-precision subtraction error.

A 2026-09-19 BF16 Zero-mode probe at physical length 197, rank 32, one valid
token and random relation features multiplied by 64 measured 9.93% relative
output L2 error, but maximum absolute error was 5.97e-7 and error L2 divided
by content L2 was 1.25e-7. At scale 8, relative error was 0.187%.
These are synthetic forward measurements, not training or gradient guarantees.
The current fixed-resolution ImageNet recipe uses all 197 tokens; image
augmentation does not reduce valid token counts. Defer this sparse-mask case
for that experiment, and revisit before workloads with very small valid
supports. TODO: evaluate an algebraically equivalent stable shared-base
calculation and its gradients without adding empirical dispatch thresholds,
raising production precision, or relaxing validation tolerances.

The FP32-reduction policy follows the
[PyTorch 2.11 numerical accuracy guidance](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms):
disabling reduced-precision BF16 reductions avoids intermediate truncation.
The implementation also disables ambient autocast around these GEMMs so an
FP16 caller cannot silently narrow BF16 operands.

The frame retains the reference detached scaling and identity augmentation, and
the accretive parameterization preserves the normal-training-domain solve.
There is deliberately no recovery fallback or synchronous solver-info poll for
inputs outside that domain.

Unbiased BF16 linear outputs and BF16 activation VJPs use FP32 accumulation
and write BF16 directly. Other unbiased outputs/VJPs keep their FP32 result
followed by the requested cast. Parameter VJPs remain FP32.

Biased CUDA linear outputs requesting FP16 or BF16 fuse `A B + bias` in a
blocked Triton GEMM: BF16 multiplicands, FP32 accumulation, FP32 bias addition,
then one final store conversion. FP16 output never rounds through BF16.
The same projection autograd owner and first-order VJP remain in use. General
FP32 outputs use PyTorch `aten.addmm.dtype`. Alternative FP32 summation
orders can differ by an output rounding bin; bitwise agreement is not universal.

The fused kernel follows the [Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
It uses the Triton runtime supplied by Linux CUDA PyTorch (tested with Triton
3.6.0 and SM120), imported lazily so CPU reference imports do not require it.
The first call compiles a kernel; warm up on the capture stream before CUDA
Graph capture. Token counts are runtime arguments, avoiding one compilation
for every image resolution. This adds a JIT projection boundary; the native
MathDx mixer remains a precompiled artifact with ABI 11. No new parameters,
checkpoint version, precision guards, or fallback implementation are added.


## Default CUDA scheduling without an explicit frame

`cuda.fast_mix()` always uses the common no-frame Triton/PyTorch implementation
for training and inference. Dynamic, Static and Zero share this path for every
supported rank and batch size, including ImageNet shapes. The model contract
13 is unchanged; native ABI 11 adds compact LU factor/solve entry points. Native materialized-frame entry points
remain available for explicit low-level comparisons; the public path never
selects them as an optimization or fallback.

The no-frame schedule uses, in real arithmetic,

```
R^T R = I + A^T A
Z = solve(R^T, A^T C)
Delta = solve(I+K, (I-K) Z)        # Dynamic
Delta = solve(I+K, I-K) Z         # Static, one map per head
Delta = 0                        # Zero
V = solve(R, Delta - eta Z)
Y = eta C + A V
```

Here A includes length/valid-count normalization. This is the same frame
`P=A R^-1` as the reference, without constructing or saving token-sized P.
Statistics and token VJPs read the existing BF16 packed projection in tiles;
only compact factors/state and partial statistics are retained. The ordinary
rank-space path records tensor values for an explicit first-order VJP, not
a private autograd graph. The three modes share the statistics, readout and input VJP.
Static's differentiable map is rebuilt per forward, never cached across updates.

There are no rank-48, mode, batch-size or sequence-length performance heuristics.
For the shared base, `N <= r` selects the equivalent smaller token system;
otherwise the rank system is used. Both forms eliminate P. Feature dimensions
are processed in tiles of at most 128 channels, including wider heads, rather
than redirecting those shapes to a materialized-frame implementation.

The no-frame statistical dots use exact BF16 inputs with FP32 accumulation.
Compact factors, general partial-pivot solves, products and their VJPs are FP32
with ambient AMP and TF32 disabled locally. The transformed readout coefficient
and compact input adjoints must not be rounded to one BF16 word: that fails the
large-relation fixture. Instead, token products split each FP32 compact operand
into a BF16 leading word and a BF16 residual word and sum their two FP32 dot
results. The token operand is already BF16. This follows the multiword arithmetic
idea in [Henry et al.](https://arxiv.org/abs/1904.06376); it is not full FP32
multiplication, and the existing forward/gradient error budgets still apply.
There is no persistent FP64 training path or global precision-setting change.

For `N <= r`, the base term is evaluated as
`eta * solve(I + A A^T, C)` instead of subtracting `eta * A R^-1 Z` from
`eta * C`. The smaller system is at most 64-by-64. This avoids cancellation
when a strong relation nearly spans a short sequence, including Zero and eta
VJPs. The learned correction remains `A solve(R, Delta)`. These are exact
algebraic identities, selected by shape; they never construct P or an unbounded
N-by-N attention matrix. Statistics for ordinary longer sequences do not use
the native detached-scale workspace; the validated envelope covers relation
scales from 0.001 to 64, raw core scales through 4, eta tails +/-9.5, all supported
ranks, and gapped/all-masked inputs. Finite fixture errors are not universal
bounds for arbitrarily ill-conditioned inputs or vanishing residuals.

The rank-space compact VJP is analytic and reuses the saved LU factors.
The smaller token-system branch retains its PyTorch autograd owner and uses
the same LU-backed implicit core solve. Repeated VJPs with `retain_graph=True`
are supported; higher-order differentiation is not. Warm up
each shape and its backward on a side stream before CUDA Graph capture. Graph
replay includes fresh input and parameter reads, all GPU kernels, and graph-pool
storage. It avoids repeated Python/autograd scheduling but does not remove
unnecessary GPU work or make memory overhead zero; see the
[PyTorch CUDA Graph documentation](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs).

## Native RHS-tiled schedule

All supported ranks use one tiled generic workspace schedule. It builds the
FP32 relation/soft-frame state, computes compact tiles, fuses dynamic-coordinate
generation with accretive factor construction and LU factorization, then solves
the equilibrium and performs the BF16/FP32 readout. There are no phase tables
or position coordinates. All modes share a base-plus-correction coefficient:
`correction = 2U - Z` for learned cores, zero for ZERO, followed by
`correction - eta Z` in the token readout and frame VJP.

STATIC factors one matrix per head. When `B >= 4` and `B * head_dim >= 4 * rank`,
it solves against the identity once per head, stores `S = solve(I+K, I)`, and
applies `U = S Z` across samples. The transpose action uses the same stored S;
it does not repeat per-sample core solves. This is equivalent to the reference
correction map `M = 2S - I`. Smaller batches retain shared LU with per-sample RHS
solves to avoid materializing a map that cannot amortize its construction.
For STATIC backward, matrix adjoints are reduced across the batch before the
shared parameterization VJP. The reduction uses scratch storage that is dead
until the later frame VJP; it does not alias the parameter-gradient output.
No detached or cross-forward factor cache is used.

ZERO skips core construction and LU solves. Its compact correction is zero,
so it stores neither U, core coordinates, LU factors nor pivots. The first
backward statistic reduction also produces its state adjoint `-eta * d_t`,
avoiding a separate equilibrium-adjoint launch. The frame, compact statistics,
content VJP and token readout remain shared with the learned-core paths.

Training stores one token-sized FP32 frame: materialization overwrites `B`
with `P` in place. Backward reconstructs `B` from BF16 packed relations,
FP32 length normalization, and saved detached scale.
The expensive triangular frame solve is not recomputed. Inference also reuses
the token region and omits the training-only compact coordinates. Head dimensions use runtime 32-column RHS tiles with
zero-filled tails, so they have no second shape whitelist; very small or very
large heads trade throughput or memory for the same operator semantics.

Frame materialization evaluates the same `P = B L^{-T}` using 128-token
right-hand-side panels at every compiled rank. The cuSolverDx triangular solve
and its operands remain FP32; only the panel schedule changes. Zero-filled
token tails preserve arbitrary sequence lengths. Panel sizes 64 and 128 were
compared against the previous rank-dependent 32/64 schedule on SM120; 128
reduced long-sequence mixer time. The choice follows NVIDIA's
[cuSolverDx performance guidance](https://docs.nvidia.com/cuda/cusolverdx/get_started/performance.html)
to tune block resources and preserve enough independent CTAs, using the existing
[TRSM implementation](https://docs.nvidia.com/cuda/cusolverdx/get_started/trsm.html).
This is an internal schedule, with no new model variant or dependency.

The fused frame/relation VJP uses 64-token local GEMM/TRSM panels for
`r=16/32/48`, and retains 32-token panels for `r=64`. Each CTA still owns 64
tokens, independently across batch, head, and token range. The wider local
panel replaces two serial subpanels without materializing `D_P` globally:
`D_P = G F^T + X D_U^T`, `D_B = D_P L^{-1} + 2 B C`, then the relation normalization
VJP. Tensor Core operand boundaries and the FP32 solve remain unchanged.
SM120 measurements favored wider panels below rank 64; the rank-64 candidate
regressed and was not adopted. Reducing the block to 128 threads or splitting
it into independent 32-token CTAs did not establish a consistent hotspot
improvement and was also not adopted. These are fixed internal scheduling
choices, with no runtime autotuning or numerical fallback.

Backward forms both `P_tile^T grad_output_tile` and the reconstructed
`P_tile^T content_tile` with one shared cuBLASDx FP32 GEMM implementation.
Each 64-token frame tile is loaded once per statistic; 32-column RHS panels
reuse it in shared memory. Multiplicands and accumulators remain FP32,
including the content reconstruction used by the complement VJP. The
ascending-token-tile reduction and compact-state storage are unchanged;
within-tile GEMM and scalar complement reductions may reorder FP32 sums.
This uses the shared-memory GEMM pattern illustrated by NVIDIA's
[cuBLASDx FP32 example](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/cuBLASDx),
also shipped with the pinned MathDx 26.06.1 package. This native statistic kernel does not use Triton or replace the accretive solve.

Forward compact-state formation groups four adjacent 32-token GEMMs per CTA,
accumulating their results in FP32 before writing one partial. Groups remain
independent across batch, head, RHS panel, and token range; a separate kernel
reduces them in ascending group order. This evaluates the same `P^T C` with
the same BF16 operands, while changing FP32 summation grouping. The producer
workspace holds at most sixty-four group partials at a time, so long sequences
do not make its transient allocation grow without bound. This selective fusion
is informed by FLA's [chunked state accumulation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_h.py)
and [split state schedule](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_h_split.py):
retain independent token groups rather than fuse the entire sequence into one
CTA. The implementation uses the existing cuBLASDx GEMM; it imports no FLA
code or recurrent semantics. The QR-frame VJP builds its compact adjoint
directly from the saved sufficient statistics; token-level VJPs retain BF16 operands with FP32
accumulators.

For sequences of at most 256 tokens, each cross-state RHS CTA accumulates
all (at most eight) token tiles and writes compact state directly. This skips
the partial allocation and reduction launch. Larger sequences keep independent
groups. This is algebraically the same contraction, with a different FP32
summation grouping when the sequence exceeds 128 tokens; forward and VJP
oracle checks cover both sides of the 256-token boundary.

For inputs with at least 32 relation tiles, frame Gram construction uses
independent 32-token FP32 partials and an ascending-token reduction before the
same Cholesky factorization. This improves long-sequence occupancy without a
public scheduling variant. Because the future `P` region now holds live `B`,
all calls use a temporary lower-triangle workspace of
`B * H * ceil(N / 32) * R * (R + 1) / 2` FP32 values, released before the
cross-state producer workspace. Shorter sequences retain the single-CTA Gram
reduction.

For long inputs (`N >= 4096`) with fewer than 128 batch/head systems, scale
preparation uses 256-token producers, a maximum reduction, and parallel
normalization. It preserves detached scaling and identity augmentation. Long
Gram reductions group 32 partials before the final factorization when there
are at least 128 token tiles and fewer than 128 systems. Short or highly batched
inputs retain the original lower-launch-count schedule. The hierarchical sums
remain FP32 and deterministic, but their grouping can change rounding.

There is no public scheduling switch: these are equivalent internal schedules.


The core GEMM keeps its FP32 accumulator separate while reusing dead BF16 A/B
storage for the subsequent factor/solver workspace. Barriers separate the
lifetimes. A trial that freed statistic partials early and delayed packed-gradient
allocation did not lower measured peak memory, so it was not retained.

The additional 64x relation-scale test uses a 1% forward budget: its measured
0.6513% error is identical before and after B reconstruction. Existing standard
fixtures retain 0.5%. This extended fixture does not establish a universal
error bound. No existing gradient tolerance is relaxed by these optimizations.

## Artifacts

CUDA 13.2 device-LTO with the cuSolverDx fatbin builds one executable image per
device-link invocation. `tools/build_cuda.sh` therefore produces strict
artifacts named `lsso_equilibrium_sm80.so`,
`lsso_equilibrium_sm86.so`, `lsso_equilibrium_sm87.so`,
`lsso_equilibrium_sm89.so`, `lsso_equilibrium_sm90.so`,
`lsso_equilibrium_sm100.so`, and `lsso_equilibrium_sm120.so`, rather than
claiming one universal binary.

SM80 is the minimum architecture because the complete contract requires native
BF16 Tensor Core operations. The CUDA 13.2 toolchain upgrade is built and validated on SM80 (A800).
Earlier scheduling optimizations were validated on SM120. Other targets and
SM120 under the new toolchain still require their own hardware validation.

`lsso.ball.cuda.load(device=...)` selects the artifact matching the requested
device. It serializes loading and binds one native operator implementation per
process, so heterogeneous SMs in one process are rejected explicitly. The
build script removes each selected SM's CMake directory and output artifact
before configuration, preventing a cached Torch_DIR from linking a newly
selected Python environment against an old libtorch. The loaded artifact
verifies both its compiled SM and native contract version before launching
kernels.

The runtime packaging tool requires all seven files. A wheel built for this
source must carry native contract 11; older released v0.6.3 wheels must not be
mixed with current source. Its generated
metadata is checked before loading: LSSO version, native contract, exact Torch
version, CUDA version, and PyTorch's C++ ABI must all match. Release packaging
removes every build-host RPATH/RUNPATH and rejects ELF artifacts requiring a
GLIBC version above 2.31, so the packaged CUDA 13.2 runtime can load on
Ubuntu 20.04 and newer x86_64 systems with the matching PyTorch runtime.

## Validation and performance scope

The 2026-09-12 retained update fuses biased low-precision projections. Trials
using 32- or 128-token backward statistics did not establish reliable overall
improvement and were withdrawn; the default backward statistic tile remains
64 tokens. Do not infer peak-memory savings from partial-buffer sizes alone.

On SM120, same-process alternating measurements of the complete biased mixer
at `B=2,N=4096,D=768,H=12,R=48` reduced eager forward-plus-backward time from
3.162 to 2.937 ms for FP16 and 3.084 to 2.826 ms for BF16. This excludes
optimizer updates and first-use compilation. Small eager shapes did not
uniformly improve. These are exploratory operator measurements, not updated
formal results or Mask R-CNN/UperNet throughput.

The publication check passed 342 tests with 6 skips. Optional integration
packages were absent, and other GPU architectures were not executed. Existing
formal results retain their recorded contract-6 provenance in `results/`.
For reproducible new measurements, record source commit, bias, dtype, rank,
shape, GPU, warmup, eager/Graph mode, and the actual live-allocation peak.


## 2026-09-17 toolchain migration validation

The SM80 artifact built successfully with the installed CMake 3.25.2, CUDA
13.2, MathDx 26.06.1 (cuBLASDx 0.7.1 / cuSolverDx 0.5.0), and PyTorch
2.14.0+cu132. PyTorch's ATen headers now require C++20; both host and CUDA
compilation use that standard. Triton 3.8.0 exercises the existing biased
projection kernel without changing its math.

On A800, core/CUDA/repository checks passed 195 tests. Integration and experiment
checks passed 146 tests with 11 optional-dependency skips, including four new
Assembly101 data/evaluation checks. Python 3.10 experiment imports now use the
TOML backport where the standard-library `tomllib` is unavailable.

Additional biased, single-head, width-64 native comparisons against the FP64
reference covered FP16 at N=16384/rank=16 and BF16 at N=55296/rank=32, with
nonzero dynamic parameters. Forward relative L2 errors were 0.348% and 0.345%;
the largest parameter-gradient error was 0.738%. Existing tolerances were not
relaxed. These finite fixtures do not establish a universal error bound.

The old toolchain was not benchmarked on this machine. New-stack exploratory
latencies therefore do not establish an upgrade speedup. Only SM80 was built;
this validation did not produce a seven-architecture release wheel.

## Private core-mode encoding

Dynamic uses base `[H,R,R]` and drive `[H,Dh,R]`. Static uses the same base
and an empty drive `[H,Dh,0]`. Zero uses empty base `[H,R,0]` and drive
`[H,Dh,0]`. These placeholders are private ABI metadata, not model parameters.
All modes share `eta C + P (correction - eta Z)`. For learned cores,
`correction = 2U-Z`; for Zero it is exactly zero. ABI 10 changes private tape
layout and Static factor representation, while model checkpoint contract 13
and parameter ownership are unchanged.

## Scheduling sources

The separation between parallel token producers and compact consumers follows
FLA's [shape-dependent local fusion](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gated_delta_rule/chunk_fwd.py).
Its unit-triangular solver is not used for the general LSSO core. Small direct
solves continue to use MathDx; the shared-memory lifetime pattern follows the
[cuBLASDx examples](https://docs.nvidia.com/cuda/cublasdx/0.7.0/examples.html).
No FLA source has been copied into the operator. Performance probes and
unselected implementations are retained outside the source tree.

## 2026-09-19 implementation audit

The no-frame token backward now uses a fixed 32-token tile for all core modes
and ranks. Its algebra and mixed-precision products are unchanged. Statistics
and readout retain their separate 128-token tiles. This follows the bounded
live-state and selective-fusion approach of
[FLA's token backward](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/common/chunk_o.py),
without copying its causal operator or introducing a model-path heuristic.

An A800 profile at B=512, N=197, H=6, rank=48 and head dimension 64 localized
the previous regression to token backward: about 17.9 ms of a 33.4 ms profiled
layer step. Triton reported 3116 spills for the 128-token/4-warp kernel.
The 32-token/4-warp version reported zero spills and took about 0.80 ms in
an isolated kernel probe. Reducing to 64 tokens was insufficient. Compiler
register/spill diagnostics matter here; removing a token-sized tensor alone
does not establish an efficient implementation.

The upstream reuse review found:

- Cholesky, triangular solves and compact GEMMs already call PyTorch's mature
  implementations. Keep these until a fused replacement wins an end-to-end
  forward/backward comparison under the same numerical contract.
- At the start of this audit the hand-written Gauss-Jordan solve repeated elimination in backward
  and for each RHS tile. Upstream
  [LU factorization](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.lu_factor_ex.html)
  and [adjoint LU solve](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.lu_solve.html)
  can reuse factors. A local FP32 probe at batch-head count 3072 and 64 RHS
  columns reduced rank-48 forward-plus-adjoint solve time from 2.63 to 1.98 ms
  when cuSOLVER was explicitly selected, but rank 16 slowed from 0.437 to
  0.465 ms. The default PyTorch backend failed Graph capture at rank 32 on
  this installation. This experiment is not installed as a global backend
  switch or a new shape-based dispatcher.
- The existing native cuSOLVERDx partial-pivot LU factor/solve components were
  selected as reusable building blocks for the compact no-frame system.
  [GETRS supports transposed solves](https://docs.nvidia.com/cuda/cusolverdx/get_started/getrs.html),
  so a dedicated compact boundary can retain the forward factors for backward.
  The ABI-11 implementation below now adopts this boundary after Graph,
  gradient and latency validation.
- [FLA solve_tril](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/utils/solve_tril.py)
  computes a unit-lower-triangular inverse. The LSSO core system is general
  dense and cannot use it directly. The QR-coordinate triangular factor is
  also non-unit; converting it merely to use this inverse is not justified
  over the existing triangular solve.

Same-process paired CUDA Graph measurements of the complete Dynamic LSSO
layer (including projections and loss backward, excluding MLP, optimizer and
DDP) used B=512, N=197, BF16 activations and FP32 parameters on A800:

| Shape | Previous 128-token backward | Retained 32-token backward |
|---|---:|---:|
| T / rank 32 | 6.474 ms | 6.668 ms |
| T / rank 48 | 11.728 ms | 12.203 ms |
| S / rank 48 | 33.010 ms | 16.018 ms |
| B / rank 48 | 67.453 ms | 33.565 ms |

The measured T cases trade about 3–4% latency for the uniform bounded tile;
S/B rank 48 improve about 50%. The T/S/B rank sweep did not change PyTorch's
measured peak tensor allocation. Register spill elimination is not a claim
of reduced allocator-visible activation memory. Timings use 50 Graph warmup
replays followed by five groups of ten replays and report the median.

The retained change passed 263 core/CUDA tests without relaxing tolerances.
The sparse-mask cancellation TODO above remains open. Local probes and
profiler artifacts remain outside the repository; no task results were added.

## ABI 11: compact LU reuse, analytic VJP and local fusion

The public rank-space no-frame implementation now uses cuSOLVERDx GETRF with
partial pivoting and GETRS. Dynamic factors one system per sample/head;
Static factors one per head and solves for its shared map. Every RHS panel
and the backward transposed solve reuse those factors. Zero skips the core
factor/solve entirely. The hand-written Triton Gauss-Jordan kernel was removed.
The private `compact_lu` entry returns factors, one-based int32 pivots and
GPU status; `compact_getrs` consumes them without mutation. Neither entry is
a standalone differentiable operator. There is no CPU status poll, global
PyTorch backend switch or new shape-based fallback. ABI 11 requires rebuilding
the native artifact; model checkpoint contract 13 is unchanged.

For `M=I+K`, the Dynamic correction obeys
`Delta=M^-1 (I-K) Z`. Given its adjoint `E`, backward computes
`U=M^-T E`, `dK=-U(Delta+Z)^T`, and the direct `dZ=2U-E`.
The generator VJP then adds the Dynamic `K(Z)` contribution to `dZ`, before
propagating through the triangular solve and Cholesky factor. Static first
sums the map adjoint across samples, then performs one adjoint solve per head.
The Cholesky VJP uses the symmetric copy of the lower triangle of `L^T dL`,
scaled by one half, with two triangular solves. This avoids a private compact
autograd graph in the ordinary rank-space path. The short token-space base
continues to use its existing autograd owner and the LU-backed core solve.

Local Triton kernels fuse coordinate normalization with generator-factor
preparation, `I +/- K` assembly, the bounded-complement value/derivative,
`Delta-eta Z`, the generator coordinate VJP, and the Cholesky VJP's triangular
symmetrization. FP32 GEMMs and Cholesky/triangular solves remain library calls.
Temporary adjoints are released after their last use; no saved factors are
modified, including on repeated first-order backward calls. The token tile
remains 32 and the numerical acceptance budgets are unchanged.

An independent gradient test found that the installed PyTorch clamp boundary
subgradient made the old reference return zero derivative at `eta_raw=0`.
The smooth mapping's derivative is `1-eps` there. Explicit sign-conditioned
exponent inputs now preserve that derivative without overflowing the inactive
branch. This repairs a reference derivative, not the mathematical operator;
nonzero eta initialization and the tail-safe mapping remain unchanged.

On A800, Dynamic single-layer forward/backward Graph timings used B=512,
N=197, BF16 activations, FP32 parameters, and the same 50-warmup/five-by-ten
measurement method as above. Baseline is the preceding 32-token implementation.
These measurements exclude the surrounding network, optimizer and DDP.

| Shape | Before | After | Peak allocated before | Peak allocated after |
|---|---:|---:|---:|---:|
| T / rank 16 | 3.763 ms | 3.249 ms | 604.8 MiB | 583.8 MiB |
| T / rank 32 | 6.660 ms | 5.530 ms | 720.4 MiB | 672.4 MiB |
| T / rank 48 | 12.203 ms | 8.953 ms | 871.3 MiB | 789.3 MiB |
| T / rank 64 | 17.198 ms | 14.271 ms | 1043.0 MiB | 924.0 MiB |
| S / rank 16 | 6.317 ms | 5.661 ms | 1225.6 MiB | 1187.5 MiB |
| S / rank 32 | 10.069 ms | 8.889 ms | 1377.8 MiB | 1294.6 MiB |
| S / rank 48 | 16.008 ms | 13.256 ms | 1564.6 MiB | 1428.5 MiB |
| S / rank 64 | 21.776 ms | 20.092 ms | 1772.4 MiB | 1582.1 MiB |
| B / rank 16 | 13.913 ms | 12.919 ms | 2438.9 MiB | 2362.7 MiB |
| B / rank 32 | 21.671 ms | 19.396 ms | 2727.0 MiB | 2560.8 MiB |
| B / rank 48 | 33.630 ms | 28.095 ms | 3076.8 MiB | 2808.2 MiB |
| B / rank 64 | 45.160 ms | 41.777 ms | 3485.7 MiB | 3103.2 MiB |

The full suite passed 450 tests with 9 skips, including pivoted ordinary and
transposed solves, singular-status reporting, independent FP64 compact VJPs,
nonzero Dynamic state dependence, masks, Graph parameter updates, and retained
backward. Native compilation and latency measurements cover SM80 only. The
extremely sparse-mask cancellation TODO remains open. No formal task training
or new accuracy results are implied by these operator measurements.
