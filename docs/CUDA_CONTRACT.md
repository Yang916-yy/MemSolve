# CUDA contract 19

Implements [model contract 24](CORE_CONTRACT.md), source version 0.14.0.
The fast path uses PyTorch CUDA linear algebra and Triton kernels. No MemSolve
native library, MathDx build, CMake or separate runtime wheel is needed.
The name change introduces `memsolve` / `MemSolve` / `MemSolveConfig`; operator
parameters and equations remain unchanged; contract 23 changes readout rounding
and the output projection's mixed-precision boundary. Model contract 24 removes
Q/K rotation from the public model; its CUDA local packing uses the existing
no-rotation branch. Learned vision positions are added outside the operator.

## Runtime

- Local validation: PyTorch `2.14.0+cu132`, Triton 3.8, SM80/A800.
- `memsolve.ball.cuda.load(device=...)` validates CUDA/Triton/device support.
- FP16/BF16 public activations; contiguous FP32 parameters on the same device.
- BF16 packed `[B,N,2*H*r+D]` at `fast_mix`; rank 16/32/48/64.
- Positive batch/length/head width. Optional FP32 valid counts with excluded rows zeroed.
- One no-P algorithm for every shape; unsupported contracts fail explicitly.
- Supported architecture checks: SM80/86/87/89/90/100/120 (SM121 maps to SM120).
  Devices other than SM80 still require hardware-specific validation.

## Device-aware token launches

Token statistics, raw/normalized readout and their token adjoints use a bounded
set of launch schedules. These change token tiles, warp counts and pipeline
stages only. The compact solve, dtypes, compensated products, normalization,
gating and mathematical VJP remain the same. Changing tiles can change FP32
reduction order; candidates are checked against the independent reference.

The normal path first tries the existing schedule. On its first eager call it
compiles the candidate, checks the compiled shared-memory requirement against
the actual device's opt-in limit, and lets Triton validate remaining launch
resource limits. Resource-incompatible schedules are replaced by smaller
equivalent schedules. Candidate workspace allocation failures are recorded and
skipped too: smaller token tiles can require larger global partial buffers.
If none fits, the operation fails explicitly: it never
changes precision or silently switches to the reference implementation.

For performance tuning, warm up representative **forward and backward** calls
before CUDA Graph capture, on an otherwise idle GPU:

```python
from memsolve.ball import cuda

with cuda.autotune():
    output = model(x, implementation="cuda")
    output.float().square().mean().backward()
model.zero_grad(set_to_none=True)
# Do the usual side-stream warmup, then capture/replay the training graph.
records = cuda.launch_report()
```

Tuning uses Triton's CUDA Graph benchmark utility and includes partial-buffer
reductions in timing. A cache hit performs neither benchmarking nor resource
probing. The process-local cache separates device index/UUID/capability/resource
limits, workload dimensions, optional mask/gate paths and relevant dtypes.
It retains scalar metadata only, not tensors. Reports include each attempted
configuration, shared-memory bytes, registers, spills, partial workspace bytes,
resource rejection reasons and (when tuned) timings. They can be serialized to
JSON for review; plans are not automatically persisted/reloaded across processes.

An uncached shape encountered inside CUDA Graph capture raises an actionable
warmup error. Autotuning must happen before capture; existing captured graphs
keep their selected kernels even if later eager calls are tuned. Concurrent GPU
work can distort timing, so tuning is explicit rather than an unconditional
cost on every new training shape. The simple rotary/packing kernels and vendor
linear algebra are not part of this search. Resource checks do not substitute
for numerical and performance validation on other GPU architectures.

Example with a complete mixer and resource report:

```bash
python -m benchmarks.benchmark_ball --batch 32 --length 196 --dim 384 \
    --heads 6 --rank 32 --dtype bfloat16 --mode train --graph --autotune \
    --launch-report /tmp/memsolve-launches.json
```

The bounded configuration search follows the approach in
[Mamba-2's chunk scan](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py);
resource handling follows
[Triton's autotuner](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py).
The launch coordinator is local because token tile sizes also determine partial
buffer allocation and reduction work. The upstream benchmarking utility is
imported; no Mamba kernel or alternate operator equation is vendored.

## Selective fusion and precision

1. Packed Q/K/V and the two gate projections use BF16 Tensor Core linear maps.
   FP32 bias addition precedes output rounding. The biased GEMM uses grouped
   row tiles and a 32-wide reduction tile, including narrow gate projections.
2. Centered Q/K convolution remains native PyTorch/cuDNN with BF16 operands
   and FP32 master weights. Both modalities use channels-last convolution:
   a 1D sequence becomes a 1 x N grid with a 1 x k filter, exactly the same
   centered convolution. V bypasses the filter. A single Triton pass packs
   Q/K/V and zeros invalid outputs. Its VJP unpacks and masks in one pass.
   The public model supplies no rotation tables; the retained low-level
   rotation primitive is not used by the vision or language forward path.
3. Independent token tiles accumulate K^T K and K^T V with BF16 products and
   FP32 accumulation, then reduce partial statistics. A fused kernel adds the
   unit ridge and produces the column-major layout for vendor Cholesky.
4. PyTorch computes L = chol(I+K^T K/n) and F = solve_triangular(L,I) once.
   The compact FP32 triangular inverse is reused in forward/backward GEMMs.
   No inverse of T or token-sized P is formed. All compact GEMMs use IEEE FP32.
   Shared T products use a standard tiled GEMM that reads the head map
   directly, removing the batch-expanded FP32 copy otherwise made by matmul.
5. Query readout, head RMSNorm, its shared gain, sigmoid gating and output
   conversion are fused. RMS/gain/sigmoid/multiply stay FP32 until one FP16
   readout store. Wo uses FP16 forward operands with FP32 accumulation/bias.
   The readout and Wo share an autograd boundary; the saved FP16 activation
   never forces the incoming gradient to FP16. Wo backward uses BF16
   multiplicands and FP32 accumulation; its readout VJP stores BF16. Casts of
   saved forward operands are fused into GEMM loads. The small dWo matrix uses
   parallel split-K and FP32 partial reduction for long token batches.
   Backward rematerializes the readout and emits Q/gate gradients plus partial coefficient/gain gradients.
   Neither the normalized activation nor its token-sized FP32 adjoint is saved.
   Storing the gain VJP before the matrix products and reloading the small
   coefficient shortens register live ranges; 32-token/two-warp backward
   tiles remain efficient without the spills seen in a single large fusion.
6. The normalized readout and its coefficient VJP multiply an exactly BF16
   query by an FP32 operand. Decompose the latter into three BF16 residual
   components and accumulate low-to-high. This retains the component omitted
   by ordinary BF16x3, which matters when RMS backward cancels radial changes.
   Other token products retain Triton's BF16x3 decomposition, with FP32
   coefficient/adjoint storage and accumulation. No global precision setting
   is left changed, and no data-dependent precision branch is introduced.
7. Compact implicit differentiation and token work remain separate. K/V
   gradients write the remaining slices of the same packed gradient. The
   Cholesky-adjoint product is tiled into at most 32 x 32 outputs and mirrors
   lower blocks, avoiding full-rank register pressure.

The optional `fast_mix(..., norm_weight=..., gate_logits=...)` fuses the gate
and returns BF16 for standalone use. `output_weight`, optional `output_bias`
and `output_dtype` additionally include Wo, retaining FP16 only inside the
autograd boundary. Without logits it returns the normalized readout, and
without a norm weight it returns the unnormalized FP32 readout. Gate logits
require a norm weight and contiguous BF16 `[B,N,H*d]`. Wo requires a norm weight
and contiguous FP32 square weights. These primitives share the same equations.

## Compact equivalence

Let B = K^T V/sqrt(n), L L^T = I+K^T K/n and F=L^-1. Then

```text
Z = F B
M = F^T T Z
O = Q M / sqrt(n)
```

For an incoming coefficient adjoint E, set U=F E and dZ=T^T U. The VJP is

```text
dB = F^T dZ
per_sample_dT = U Z^T
dT = sum_batch(per_sample_dT)
A = -T (per_sample_dT)^T - dZ Z^T
S = sym(Phi(A))
dGram = F^T S F
```

Phi keeps the lower triangle and halves the diagonal; sym averages a matrix
with its transpose. This follows from L^T dL = -T(dT)^T-dZ Z^T in the ordinary
Cholesky VJP. It holds for nonsymmetric, singular or zero learned T as well as
T=I. Reusing F replaces six triangular solves per training step with one;
it does not treat the factorization as constant during differentiation.
The reference continues to use triangular solves as an independent oracle.

## Range audit

- Q/K/V, compact states, learned T and upstream adjoints have no fixed
  FP16-safe range. They are not cast directly to FP16.
- Gram statistics, Cholesky, triangular inverse and compact adjoints stay
  FP32. Quantizing the Gram can lose the unit ridge in sensitive directions.
- Since I+K^T K/n >= I, ||F||_2 <= 1. This bound permits range-safe storage but
  does not establish adequate FP16 solve accuracy; F is kept in FP32.
- RMS/sigmoid/RoPE arithmetic stays in FP32 registers. The fast path avoids
  full token-sized FP32 buffers for those operations. Parameters, gradients
  of master weights and optimizer moments remain FP32.
- The non-affine RMS value satisfies abs(xhat_i) <= sqrt(d); sigmoid cannot
  increase its magnitude. After the learned gain the bound is abs(gamma_i)*sqrt(d).
  Readout/Wo use FP16 forward operands in the ordinary training envelope; gamma
  and Wo are not clipped or reparameterized. The forward-only FP16 boundary
  must not be reused for backward: upstream gradients can be arbitrarily small.
- The former single-token (`r=16,d=7`) FP64 core-gradient failure is covered
  by the original, unchanged tolerance. Residual-compensated readout and
  coefficient-adjoint products make that regression test pass. This is a
  checked boundary, not a guarantee of small relative error at every zero
  gradient or for arbitrarily ill-conditioned inputs.

Forward/gradient tests cover all four ranks, zero/general T, gapped/empty masks,
short sequences, large key scales, small values, gate saturation, odd head
widths, and both local geometries. CUDA Graph tests change inputs and Delta
between replays. Performance probes and temporary artifacts stay outside Git.

## Upstream provenance

Post-normalization gate fusion follows the forward/VJP decomposition in
[FLA layernorm_gated.py](https://github.com/fla-org/flash-linear-attention/blob/8024667ab58fdd8986587147fca71fecc017f977/fla/modules/layernorm_gated.py).
MemSolve uses sigmoid instead of SiLU and keeps normalization/gating arithmetic
in FP32 through the readout store. Convolution remains native cuDNN;
FLA's causal short-convolution semantics are not substituted for centered filters.
The compact inverse uses ordinary PyTorch `solve_triangular` and cuBLAS GEMMs.


The packed Q/K rotary kernel follows the adjacent-pair load and conjugate
backward algorithm in
[FLA rotary.py, commit 864a87f6ce5be8828bef81eb22baafd41937cdf2](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/modules/rotary.py).
It removes causal offsets/variable-length indexing and specializes to MemSolve's
packed Q/K/V layout and fixed FP32 tables. The axial frequency schedule follows
[RoPE-ViT](https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py).

The grouped RMSNorm forward/backward is adapted from
[FLA layernorm.py, commit 954438d1fcb5e1bb05c22f9908de9c5c2df74ae5](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/modules/layernorm.py).
The specialization removes residual/bias/LayerNorm branches and retains grouped
RMS equations, FP32 reductions and partial weight-gradient reduction. The MIT
copyright and permission notice are in `NOTICE`.

The projection GEMM and grouped program ordering follow the
[official Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
The FP32-input dot decomposition is supplied by
[Triton `tl.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html),
following [Henry et al.](https://arxiv.org/abs/1904.06376); the same algorithm
family appears in [OpenXLA](https://github.com/openxla/xla/blob/e33f93fb7220d408811afdc926cf10baaf49c64e/xla/backends/gpu/codegen/triton/dot_algorithms.cc).
The readout rematerialization applies the IO-saving principle of
[FlashAttention](https://arxiv.org/abs/2205.14135) to the MemSolve readout; no
softmax-attention kernel or equation is substituted for ridge regression.
Token scheduling uses separate parallel statistics and readout stages, following
FLA's selective-fusion approach. The within-sample tile traversal and small-tile
warp choices also draw on
[FLA chunk_o.py, commit 864a87f6ce5be8828bef81eb22baafd41937cdf2](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/common/chunk_o.py).
They are adapted to packed QKV and measured on SM80 rather than importing FLA's
full autotuner. No causal state recurrence, UT delta-rule
transform, or GDN2 gating code is used: those implement a different operation.

The readout forward precision follows the range-based rationale of
[FLA Mesa's normalized Q/K FP16 outputs](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/mesa_net/chunk.py),
without adding Q/K normalization to MemSolve. Its
[state-precision fix](https://github.com/fla-org/flash-linear-attention/pull/1152)
and [small-gradient report](https://github.com/fla-org/flash-linear-attention/issues/1283)
motivate keeping solve arithmetic FP32 and backward intermediates out of FP16.
Parallel dWo reduction uses the standard
[CUTLASS split-K strategy](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#parallelized-reductions),
implemented in the existing Triton projection GEMM; no CUTLASS dependency is added.

## Graph and verification

Use ordinary fused AdamW on Delta; set the optimizer's `capturable=True` when
capturing its step. Forward forms T = I + Delta on each execution. The CUDA
primitive accepts an effective T and optional gate logits under CUDA contract 19.
Warm up on the capture stream before capture. CUDA Graph replay updates inputs
and the learned core; no detached map is cached across forwards. The ImageNet
entrypoint captures forward/backward and keeps the optimizer outside the graph.
Compile-graph mode compiles the surrounding encoder and leaves the CUDA mixer
as an explicit boundary.

Run `python -m pytest tests/core tests/cuda`, the vision integration checks,
and `python tools/check_repository.py`. Timing probes and generated reports
belong outside the repository. Old native-path timings and older published panels
do not measure this changed architecture.

To time a complete BF16 image-shaped mixer (projections, Q/K convolution,
RoPE, memory solve, normalization, gate, output projection and backward):

```bash
python benchmarks/benchmark_ball.py --batch 128 --length 196 --dim 384 \
  --heads 6 --rank 32 --spatial-shape 14 14 --dtype bfloat16 --mode train --graph
```

Omit `--spatial-shape` for a sequence. The benchmark reports its execution
mode and allocated-memory peak; it excludes the encoder FFN, optimizer and
data pipeline. Register/spill checks and timing comparisons must use the same
workload and GPU. Zero spills on tested shapes is not an all-shape guarantee.
