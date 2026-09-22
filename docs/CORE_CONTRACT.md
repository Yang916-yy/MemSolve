# Core Contract

The operator accepts x[B,N,D] and an optional boolean valid_mask[B,N]. It returns y[B,N,D]. Batch size
B and sequence length N must be positive. The reference accepts `float16`,
`bfloat16`, `float32`, and `float64`; the native CUDA implementation accepts
only `float16` and `bfloat16` public inputs. Outputs match the input dtype. A
sequence may be entirely masked; its output is zero.

For each head, let A be the relation coordinates after
normalization by sqrt(n_valid), and let C be the masked content coordinates.
The QR soft frame and shared compact state are

~~~
P = qr_soft_frame(A)
Z = P^T C.
~~~

For DYNAMIC mode, the compact coordinates are generated from that same state:

~~~
R = R0 + Z W_drive / sqrt(n_valid).
~~~

STATIC uses R = R0. ZERO owns no compact coordinates and applies the zero
compact-core branch directly:

~~~
Y = eta (C - P Z).
~~~

For DYNAMIC and STATIC, define

~~~
L = tril(R, -1) + Diag(softplus(diag(R) + softplus_inverse(1)))
Omega = triu(R, 1) - triu(R, 1)^T
K = L L^T + Omega
U = solve(I + K, Z).
~~~

The symmetric part of I + K is I + L L^T, so the equilibrium is unique. The
per-head complement and token output before the output projection are

~~~
eps_d = finfo(calculation_dtype).eps
eta = (1 - eps_d) tanh(eta_raw)
Y = eta C + P [2 U - (1 + eta) Z].
~~~

Equivalently, M = 2 (I + K)^-1 - I and

~~~
Y = eta C + P (M - eta I) P^T C.
~~~

The reference computes this as a Zero base plus a learned correction:

~~~
Delta = solve(I + K, (I - K) Z)
Y = eta (C - P Z) + P Delta.
~~~

The identity `solve(I + K, I - K) = 2 (I + K)^-1 - I` proves
equivalence. The final readout combines the two frame products as
`eta C + P (Delta - eta Z)` so it does not materialize two token-sized
products. ZERO has `Delta = 0` (equivalently `K = I`, not `K = 0`).
For STATIC, compute `M = solve(I + K, I - K)` once per head and apply
`Delta = M Z` across the batch. The solve remains in the autograd graph;
there is no detached or cross-forward cache. Dynamic correction RHS products,
static correction products and their VJPs use IEEE FP32 on CUDA, while FP64
diagnostics retain ordinary differentiable FP64 operations. For incoming
correction gradient `G`, the CUDA reference uses the equivalent VJP
`V = solve((I + K)^T, G)`, `dZ = 2 V - G`, and
`dK = -V (Delta + Z)^T`, avoiding redundant differentiation through both
occurrences of K. The backward solve reuses the forward LU factors.
Static broadcasts its shared correction matrix over the batch; its matrix
gradient accumulates contributions from all samples. This algebraic rewrite preserves the operator; removal of rotation changes
the checkpoint contract. Native CUDA forms the learned correction as `2U-Z` and shares the final
`correction-eta Z` coefficient with Zero, whose correction is exactly zero.
Its Static schedule can apply a once-per-head solved map instead of repeated
RHS solves; all schedules are checked against this reference.

At R = 0, L = I, Omega = 0, U = Z / 2, and M = 0. DYNAMIC and STATIC both
start at this compact point; DYNAMIC additionally starts with W_drive = 0.
The correction is zero at initialization but has a nonzero core derivative;
the implementation does not skip or detach it when its value is zero.

In exact arithmetic, the QR frame, accretive generator, and eta
parameterization make the frozen token mixer contractive. The fixed one-ULP
interior scale keeps the realized FP32 and FP64 complement strictly inside the
unit interval. The reference evaluates `tanh` through sign-specific stable
logistic identities, so its tail gradient remains nonzero when a direct FP32
`tanh` forward would round to `+/-1`, without overflowing in the opposite
inactive branch. Like every finite-precision exponential, this does not claim
meaningful tail gradients for astronomically extreme raw coordinates.

The operator contains no feature rotation or position-coordinate interface.
External absolute or spatial position embeddings belong to the surrounding model.

The frame, compact-state storage, accretive factor Gram `F F^T`, and solve
calculations use FP32 unless the input is FP64. CUDA evaluates the factor Gram
with IEEE FP32 FMA; its small size makes avoiding a second factor quantization
worthwhile. Ordinary projections and eligible compact contractions use BF16
multiplicands with FP32 accumulation. Packed activations and the pre-output
boundary use BF16. Sensitive eta and solve state stay FP32.
FP16 and BF16 public inputs are accepted by CUDA; the result matches the input
dtype. FP32/FP64 inputs remain available on the reference path. Invalid tokens
are zeroed before every compact statistic.

## Serialized and numerical boundaries

The current model `_extra_state` contract is version **13**. This is separate
from native CUDA ABI **11** and the ImageNet runner envelope format **7**.
Loading requires every saved operator-contract field to match, including model
geometry and ablations. Missing or mismatched contracts fail even under
`strict=False`; older weights need explicit validation before migration.

Biased low-precision CUDA projections add FP32 bias to an FP32 accumulator
before a single FP16/BF16 store. They use a lazily compiled Triton GEMM;
parameter gradients retain the existing FP32 reduction boundaries. Summation
order can change rounding, so algebraic equivalence is not a promise of
bitwise-identical training. See [CUDA implementation details](CUDA_CONTRACT.md).

The contraction statement freezes the input-conditioned frame and generator.
It does not bound the complete input Jacobian, which also differentiates those
quantities. Implementation benchmarks do not establish downstream accuracy.

## Explicit reference ablations

`skew_coupling=False` sets Omega to zero in K = L L^T + Omega while
retaining the accretive factor. `scalar_complement=False` fixes eta to zero
with no learned complement update. Both flags are recorded in the model
checkpoint contract and require `implementation="reference"`. The native
default continues to require both flags enabled.


The default CUDA path uniformly uses a no-frame implementation of these same
identities: `R^T R=I+A^T A`, `Z=R^-T A^T C`, and
`Y=eta C + A R^-1(correction-eta Z)`. It shares compact and token code across
core modes. For `N<=r`, the base is computed as `eta (I+A A^T)^-1 C` to avoid
subtractive cancellation. This changes scheduling and rounding, not the model.
See `CUDA_CONTRACT.md` for the primal/dual dimension choice and the fixed mixed-precision policy.

The frame is soft: `PP^T = A(I+A^T A)^-1 A^T`, so the common base is
`eta (I+A A^T)^-1 C`, not an orthogonal projection onto a null space.
Dynamic, Static and Zero all share this base. With `K=I`, the correction
vanishes in any mode; the trainable core derivatives need not vanish.
