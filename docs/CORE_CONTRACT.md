# MemSolve accretive-equilibrium operator

MemSolve uses the no-RankRotary relation/content operator restored from
`e769a9a`, with two modes: `equilibrium` (default) and `complement`.
Model checkpoint contract **26** and native CUDA ABI **17** require matching
source and artifacts. Old mode names are not aliases. Historical checkpoints
and their source snapshots retain their original contracts.

## Definition

For each head, n is the number of valid tokens. A packed projection produces
relation B and content C. There is no independent query projection, Q/K
convolution, internal positional rotation, output RMSNorm or channel gate.

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

## Ownership and model shells

`reference.py` owns these equations; `model.py` owns learned parameters.
The default visual shell has residual CPE, GELU, token-LN mean pooling,
no LayerScale and linear stochastic depth. S uses MLP ratio 4.125; T/B use 4.
See [architecture](ARCHITECTURE.md) and [CUDA contract](CUDA_CONTRACT.md).
Historical Base used RankRotary and a different training protocol; its result
is not an exact reproduction of this configuration. Experiment records retain
their original labels rather than being relabeled as current measurements.
