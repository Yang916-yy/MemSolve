# CUDA contract 17

Implements [model contract 19](CORE_CONTRACT.md), source version 0.11.0.
The fast path uses PyTorch CUDA linear algebra and Triton kernels. No Ridgon
native library, MathDx build, CMake or separate runtime wheel is needed.

## Runtime

- Local validation: PyTorch `2.14.0+cu132`, CUDA 13.2 runtime, SM80/A800.
- `ridgon.ball.cuda.load(device=...)` validates CUDA/Triton/device support.
- FP16/BF16 public activations; contiguous FP32 parameters on the same device.
- BF16 packed `[B,N,2*H*r+D]` at `fast_mix`; rank 16/32/48/64.
- Positive batch/length/head width. Optional FP32 valid counts with excluded rows zeroed.
- One no-P algorithm for every shape; unsupported contracts fail explicitly.
- Supported architecture checks: SM80/86/87/89/90/100/120 (SM121 maps to SM120).
  Devices other than SM80 still require hardware-specific validation.

## Selective fusion and precision

1. Packed Q/K/V projection uses the existing BF16 Tensor Core linear map.
2. Independent token tiles accumulate K^T K and K^T V with BF16 products and
   FP32 accumulation, then reduce partial statistics.
3. PyTorch owns Cholesky and triangular solves of the key Gram. The effective
   T = I + Delta multiplies the compact state; no learned-core factorization is performed.
   A fused kernel adds the identity and produces the column-major system layout.
4. Query readout uses Tensor Cores without materializing P. Triton's upstream
   `bf16x3` decomposition handles FP32 compact coefficients and adjoints, with
   FP32 storage and accumulation. This is a multi-product approximation, not
   a single BF16 rounding of those tensors or a claim of bitwise FP32 products.
   No global TF32/AMP precision setting is left changed.
5. The model fuses query readout, per-head RMSNorm (epsilon `1e-6`), shared `[D]` affine weights and BF16
   output conversion. Backward recomputes the FP32 readout from the saved Q
   and compact coefficient, then computes the RMS VJP, Q gradient, partial
   coefficient gradient and partial affine gradient together. Affine gradients
   are summed over batches, heads and token tiles, matching the shared gain. Neither the raw
   readout nor its FP32 token-sized adjoint is stored. K/V gradients follow the
   compact-system VJP and write the other slices of the same packed gradient.
   Standalone
   unnormalized `fast_mix` and `head_rms_norm` primitives remain available.
6. Backward separates compact implicit differentiation from token work, and
   computes distinct Q, K and V gradients. No old shared-A gradient is reused.

Compact operations remain FP32. The explicit reference and independently
constructed FP64 original-coordinate system validate both forward and every
parameter/input gradient. Precision choices do not change the mathematical
readout or introduce a shape-dependent model.

The biased projection uses larger GEMM tiles for large row counts and groups
eight row tiles before moving through columns to improve operand reuse in L2.
FP32 bias addition precedes the final output rounding. Token kernels traverse
heads and token tiles within each sample before moving to another sample.
Forward statistics use 256-token tiles and omit the sum kernels when there is
only one partial. This is the same K^T K/n and K^T V/sqrt(n) reduction, with a
different floating-point summation order. The normalized readout adjoint uses
32-token tiles and two warps at every rank; the K/V adjoint uses two warps at
ranks 16/32 and four at ranks 48/64. These are scheduling choices, not precision
or mathematical dispatch rules.

## Compact backward equivalence

Write the forward state as Z=L^-1 B, coefficient M=L^-T T Z, and its incoming
adjoint as E. Set U=L^-1 E and dZ=T^T U. The ordinary reverse pass gives
dB=L^-T dZ, per-sample dT=U Z^T, and dL=-M U^T-dB Z^T. Therefore

    L^T dL = -(T Z) U^T - dZ Z^T = -T (dT)^T - dZ Z^T.

The fused kernel reuses the per-sample dT before its batch reduction and forms
sym(Phi(L^T dL)) directly, where Phi keeps the lower triangle and halves its
diagonal. Both matrix products use IEEE FP32, including the value-dimension
reduction. The two remaining triangular solves produce the original Cholesky
VJP. No dL tensor or multiplication of dL by L^T is needed. This identity holds
for general learned T; it does not assume symmetric T or its initialization.

## Range audit

- Q, K, V, the compact state/coefficient and upstream adjoints have no fixed
  FP16-safe range. A bound relative to V or its gradient is not an absolute bound.
- Gram statistics, Cholesky, triangular solves and their adjoints stay FP32.
  Quantizing the Gram can lose the unit ridge or perturb sensitive directions.
- From H=I+K^T K/n=L L^T >= I, ||L^-1||_2 <= 1. This makes its values bounded,
  but does not guarantee accurate solves after rounding; factor inverses are
  not materialized in the selected implementation.
- Each normalized feature before its affine gain has absolute value <=sqrt(d).
  Its FP16 storage is range-safe at ordinary head widths, but RMS backward
  subtracts nearly aligned terms. The fused implementation instead recomputes
  that feature in FP32, reducing rounding in the epsilon-sensitive VJP.
- The master Delta parameter, effective T = I + Delta, optimizer updates and
  moments use FP32. T is not bounded by a norm constraint.
  This audit adds no FP16 activation stage, clipping, or data-dependent fallback.

TODO: the single-token fused-readout oracle case (`r=16`, `d=7`) has a
near-radial core gradient. Under contract 18 (epsilon `1e-6`), maximum absolute
error is about `3.32e-6`, above the unchanged `1e-6` absolute allowance; the
FP64 core-gradient norm is only `6.52e-7`, so relative error is large. The same
discrepancy reproduces with the previous per-head weight layout using repeated
shared weights at the same epsilon, and also occurs with CUDA readout followed
by PyTorch RMSNorm. Sharing affine weights is therefore not its cause; the
finite-precision readout remains to be investigated. The test remains failing,
including under model contract 19 with its original tolerance; this boundary must not be described as fully
validated. Image-shaped oracle and graph checks pass. No shape fallback or
precision promotion is added. Contract 17's epsilon `1e-5` had maximum absolute
core-gradient error `1.59e-6`; those are measurements of different norm settings.

## Upstream provenance

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
[FlashAttention](https://arxiv.org/abs/2205.14135) to the Ridgon readout; no
softmax-attention kernel or equation is substituted for ridge regression.
Token scheduling uses separate parallel statistics and readout stages, following
FLA's selective-fusion approach. The within-sample tile traversal and small-tile
warp choices also draw on
[FLA chunk_o.py, commit 864a87f6ce5be8828bef81eb22baafd41937cdf2](https://github.com/fla-org/flash-linear-attention/blob/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/common/chunk_o.py).
They are adapted to packed QKV and measured on SM80 rather than importing FLA's
full autotuner. No causal state recurrence, UT delta-rule
transform, or GDN2 gating code is used: those implement a different operation.

## Graph and verification

Use ordinary fused AdamW on Delta; set the optimizer's `capturable=True` when
capturing its step. Forward forms T = I + Delta on each execution. The CUDA
primitive still accepts an effective T, so its algorithm contract remains 17.
Warm up on the capture stream before capture. CUDA Graph replay updates inputs
and the learned core; no detached map is cached across forwards. The ImageNet
entrypoint captures forward/backward and keeps the optimizer outside the graph.
Compile-graph mode compiles the surrounding encoder and leaves the CUDA mixer
as an explicit boundary.

Run `python -m pytest tests/core tests/cuda`, the vision integration checks,
and `python tools/check_repository.py`. Timing probes and generated reports
belong outside the repository. Old native-path timings and older published panels
do not measure this changed architecture.
