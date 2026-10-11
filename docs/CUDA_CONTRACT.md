# CUDA contract

The current no-RankRotary operator supports `equilibrium` (default) and
`complement`. Model contract **26**, native ABI **12**, and the installed
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
R^T R = I + A^T A
Z = solve(R^T, A^T C)
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

The ordinary branch has an explicit analytic first-order VJP. Its shared core
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
  adjoints stay FP32 with ambient AMP/TF32 disabled locally.
- Token readout and input VJPs split sensitive FP32 compact operands into a
  BF16 leading word and residual word, accumulating both products in FP32.
  Rounding these operands to a single BF16 word fails the strong-relation
  fixture. This follows the multiword arithmetic idea of
  [Henry et al.](https://arxiv.org/abs/1904.06376).
- Eta uses the tail-safe interior-tanh value and analytic derivative, including
  the derivative `1-eps` at zero. Packed outputs/input gradients are BF16;
  parameter gradients remain FP32.

Triton handles tiled token statistics/readout/VJPs and compact elementwise
fusion. FP32 matrix products, Cholesky and triangular solves use PyTorch library
operations. NVIDIA cuSOLVERDx supplies partial-pivot GETRF/GETRS. The native
interface exposes only `contract_version`, `compact_lu` and `compact_getrs`;
there is no full-mixer native forward/backward ABI. Direct native solves reject
autograd-enabled inputs because their derivatives belong to the Python owner.
Factorization status remains on GPU, without a CPU status poll.

The input VJP uses 32-token tiles to limit live register state. Feature tiles
are at most 128 channels; nonmultiple tails are masked. Biased projection
fusion follows the [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
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
PYTHON=/cywang/vision-venv/bin/python MEMSOLVE_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh
```

Device LTO builds one executable image per target. The loader chooses
`memsolve_equilibrium_sm<arch>.so` and enforces a matching architecture per
process. Targets are SM80/86/87/89/90/100/120 (SM121 selects SM120); this cleanup
was built and exercised on SM80/A800. Other targets require hardware validation.
Mixed architectures in one process are rejected. Native kernels verify the
compiled architecture before launch.

`tools/package_cuda_runtime.py` requires all seven artifacts and records exact
source/native versions, Torch/CUDA versions, and the C++ ABI. Packaged artifacts
must match these checks. It removes build-host RPATH/RUNPATH and rejects GLIBC
requirements above 2.31. ABI 11 artifacts must be rebuilt for ABI 12; historical
training source snapshots keep their own binaries and remain separate.
