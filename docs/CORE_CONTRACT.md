# MemSolve accretive-equilibrium operator

MemSolve uses the no-RankRotary relation/content operator restored from
`e769a9a`, with two modes: `equilibrium` (default) and `complement`.
Model checkpoint contract **26** and native CUDA ABI **17** require matching
source and artifacts. Old mode names are not aliases. Historical checkpoints
and their source snapshots retain their original contracts.

## Definition

For each head, n is the number of valid tokens. A packed projection produces
relation B and content C. There is no independent query projection, Q/K
convolution, output RMSNorm or channel gate. Optional RoVE transports content
between token-local coordinate systems; the relation frame is unchanged.

```text
A = B / sqrt(n)
R^T R = I + A^T A                  positive-diagonal upper factor
P = A R^-1                        soft frame
Z = P^T C
L = tril(U,-1) + diag(softplus(diag(U) + log(expm1(1))))
Omega = triu(U,1) - triu(U,1)^T
J = L L^T + Omega
F = solve(I + J, I - J)
eta = interior_tanh(eta_raw)
Y = eta (C - P Z) + P F Z         equilibrium
Y = eta (C - P Z)                 complement
output = concat_heads(Y) Wo + bo
```

In `equilibrium`, U is learned independently per head and shared across
samples. F is solved once per head in each forward and reused across the
batch; it is not cached across optimizer updates. The token operator remains
input-dependent through P. This mode was previously called Static.

`complement` omits U, J and the compact core solve. It retains learned relation
and content projections, eta, and the output projection. It is a nonzero,
input-dependent content compensation operator. Its equation matches J=I
(F=0), not J=0. P is a soft frame, so I-PP^T is not generally an orthogonal
projector. This mode was previously called Zero.

The former Dynamic content-conditioned core generator has been removed.
`equilibrium` starts with U=0, giving J=I and F=0; eta starts at 0.9 per head.
The two modes therefore have the same initial formula when their remaining
weights match, but equilibrium has trainable core gradients at initialization.
No initialization, projection or training recipe changes accompany the rename.

The reference obtains P via reduced QR of `[A; I]`. CUDA uses an equivalent
compact factorization and readout without storing token-sized P. Both modes
mask padded rows and normalize by valid-token count. Completely masked samples
return zero. At fixed head width and rank, sequence cost is linear; matrix
factorization sizes depend on rank, not unbounded token length.

## Optional rotary value encoding

With `MemSolveConfig(rotary=True)`, each head uses a full-channel, adjacent-pair
rotation O_i at token position p_i. Let M=F-eta I (F=0 for complement):

```text
C_rot[i] = O_i C[i]
Z = P^T C_rot
correction = P M Z
Y[i] = eta C[i] + O_i^T correction[i]
output = concat_heads(Y) Wo + bo
```

The inverse rotation precedes the output projection. It covers the entire
global correction, including `-eta P P^T`, and leaves the direct content term
as `eta C` without a numerical rotation round trip. Both modes use the same
transport. No relation rotation, new learned parameter or independent query
is introduced. Head width must be even.

At a fixed relation frame, this is orthogonal conjugation of the complete
token/content operator. It preserves that operator's singular values and
spectrum; it does not bound the Jacobian of the input-dependent full network.
Token interactions contain `O_i^T O_j = O(p_j-p_i)`. This relative-position
property holds for a fixed frequency scale, including a common position shift.

The default uses base 10000, a training window of 128, and frequency-only YaRN
beyond that window (`rotary_scaling="yarn"`, beta_fast=32, beta_slow=1).
Amplitude remains one. `rotary_scaling="none"` disables interpolation.
Context length defaults to each sample's first-to-last valid-token span, or N
without a mask; an all-masked sample uses one. Padding therefore does not change
the frequencies of a sample. Explicit `rotary_context_length` accepts a positive
integer or integer tensor `[B]` to select a common or per-sample window.
Position IDs accept `[N]`, `[1,N]` or `[B,N]` and do not determine the scale.
For masked sequence parallelism beyond the training window, supply global
context lengths explicitly; the unmasked default is the global sequence length.
Default positions include the shard offset. Within the training window (or with
scaling disabled), frequencies are independent of context length, so the
default shared position IDs need only one table for the batch.

Phase/trigonometric arithmetic uses FP32 (FP64 in the oracle). CUDA fuses
content transport into statistics and readout instead of materializing rotated
BF16 content. Enabling RoVE adds its settings to the checkpoint contract;
the unrotated contract and visual defaults retain their existing meaning.

The transport follows [RoVE](https://arxiv.org/abs/2606.11275) and its
[standard implementation](https://github.com/AGarciaCast/RoVE/blob/8e3eb84efb7c920289aba776fbdb189f93b79e01/model/pos_methods/rove.py).
YaRN uses the upstream evaluation frequency ramp, without an attention
temperature or amplitude multiplier. These settings define an implementation,
not a demonstrated long-context accuracy claim for MemSolve.

## Ownership and model shells

`reference.py` owns these equations; `model.py` owns learned parameters.
The default visual shell has residual CPE, GELU, token-LN mean pooling,
no LayerScale and linear stochastic depth. S uses MLP ratio 4.125; T/B use 4.
See [architecture](ARCHITECTURE.md) and [CUDA contract](CUDA_CONTRACT.md).
Historical Base used RankRotary and a different training protocol; its result
is not an exact reproduction of this configuration. Experiment records retain
their original labels rather than being relabeled as current measurements.
