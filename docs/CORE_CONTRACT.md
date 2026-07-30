# Operator Equations and Core Specification

MemSolve implements a low-rank accretive-equilibrium mixer with two operating modes: `equilibrium` (default) and `complement`. Checkpoint metadata uses contract version **26**, and the native CUDA extension uses ABI version **17**.

## Forward Equations

Given $n$ valid tokens in a sequence, a single packed linear layer projects the input into per-head relations $B \in \mathbb{R}^{n \times r}$ and content $C \in \mathbb{R}^{n \times d}$:

```text
A      = B / sqrt(n)
R^T R  = I + A^T A                                    (Cholesky factor, positive diagonal)
P      = A R^-1                                       (soft frame, P^T P = (I + G)^-1 G)
Z      = P^T C                                        (compressed memory state)
L      = tril(U, -1) + diag(softplus(diag(U) + c_0))  (c_0 = log(expm1(1)))
Omega  = triu(U, 1) - triu(U, 1)^T
J      = L L^T + Omega                                (strictly accretive generator, J + J^T >= 2 L L^T)
F      = (I + J)^-1 (I - J)                           (Cayley transform, ||F||_2 < 1)
eta    = (1 - eps) * tanh(eta_raw)                    (bounded complement weight in (-1, 1))

Y      = eta * (C - P Z) + P F Z                      (equilibrium mode)
Y      = eta * (C - P Z)                              (complement mode)
output = concat_heads(Y) W_o + b_o
```

### Design Notes

- **Shared per-head generator**: In `equilibrium` mode, the unconstrained parameter $U \in \mathbb{R}^{r \times r}$ is learned per head and shared across the batch. The $r \times r$ Cayley map $F$ is solved once per head during each forward pass.
- **Initialization**: Setting $U = 0$ yields $L = I$, $\Omega = 0$, $J = I$, and $F = 0$, while $\eta$ is initialized to $0.9$. Both `equilibrium` and `complement` therefore start from the exact same input-output mapping, while `equilibrium` immediately receives nonzero gradients on $U$.
- **Masking and complexity**: Invalid (padded) tokens are zeroed before statistics accumulation, and $n = \max(\text{valid\_tokens}, 1)$ normalizes the relation scale. The reference backend forms $P$ explicitly via a thin QR factorization of $[A; I_r]$; the CUDA backend factors $I_r + A^\top A$ directly in rank space without materializing the $n \times r$ matrix $P$, keeping time and memory linear in sequence length $N$.

## Rotary Value Encoding (RoVE)

Setting `MemSolveConfig(rotary=True)` enables position-aware content transport based on [RoVE](https://arxiv.org/abs/2606.11275). For each token position $p_i$, a block-diagonal 2D rotation $O_i$ acts on adjacent channel pairs of the content stream while leaving the relation stream $A$ untouched. Letting $M = F - \eta I_r$ (with $F = 0$ in `complement` mode):

```text
C_rot[i]      = O_i C[i]
Z             = P^T C_rot
correction[i] = (P M Z)[i]
Y[i]          = eta * C[i] + O_i^T correction[i]
```

Because $O_i^\top O_j = O(p_j - p_i)$, any pair of tokens interacts through their relative displacement $p_j - p_i$, and the compact operator is orthogonally conjugated without altering its singular values or norm bounds. The direct path $\eta C[i]$ bypasses the rotation round-trip to avoid unnecessary floating-point rounding.

### 1D and 2D Axial Frequencies

- **1D sequences**: Uses a standard base-10,000 geometric frequency schedule (`head_dim` must be even). By default, sequences longer than `rotary_train_length=128` apply frequency-only YaRN interpolation (`rotary_scaling="yarn"`, `beta_fast=32`, `beta_slow=1`, unit amplitude); set `rotary_scaling="none"` to disable frequency scaling.
- **2D image grids**: `axial_rotary_position_factors` splits the head channels evenly between horizontal and vertical coordinates, $O(x, y) = \mathrm{diag}(R(x), R(y))$ (`head_dim` must be divisible by 4), so pairwise transport depends only on $(x_j - x_i, y_j - y_i)$.

## Module Layout

- [`memsolve/ball/reference.py`](../memsolve/ball/reference.py): PyTorch reference implementation, FP64 test oracle, and rotary helpers.
- [`memsolve/ball/model.py`](../memsolve/ball/model.py): `MemSolve` `nn.Module`, parameter initialization, checkpoint validation, and diagnostics.
- [`memsolve/ball/config.py`](../memsolve/ball/config.py): `MemSolveConfig` and `CoreMode` definitions.
- [`memsolve/ball/cuda.py`](../memsolve/ball/cuda.py): Mixed-precision Triton + cuSOLVERDx/cuBLASLt fast path (see [CUDA_CONTRACT.md](CUDA_CONTRACT.md)).
