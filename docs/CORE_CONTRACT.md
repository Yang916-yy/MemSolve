# Q/K/V ridge memory and learned query readout

Model contract **19**, source version **0.11.0**. Independent Q/K/V projections,
a shared query map T = I + Delta, and per-head RMSNorm define one operator.
There is no input-conditioned core generator, reflected readout, internal value
skip, projection-local convolution, rotation, or mode selector.

## Definition

For one head, n valid tokens, key/query rank r and value width d:

```text
Q = X Wq + bq                 [n,r]
K = X Wk + bk                 [n,r]
V = X Wv + bv                 [n,d]
Ak = K / sqrt(n), Aq = Q / sqrt(n)
R^T R = I + Ak^T Ak           upper Cholesky, positive diagonal
Pk = Ak R^-1, Pq = Aq R^-1
Z = Pk^T V
T = I + core_delta           [r,r], shared across samples
O = Pq T Z
Y = concat_h[RMSNorm_h(O)] Wo + bo
```

One packed projection stores independent Q/K/V in the layout
`[Q_all_heads, K_all_heads, V_all_heads]`. `core_delta` has shape `[H,r,r]`
and initializes to zero. Forward forms only the compact T = I + Delta.
There is no norm constraint, normalized raw proxy or inverse of T.
Each token/head RMSNorm has epsilon `1e-6` and a learned channel gain gamma
of shape `[d]`, shared across heads and initialized to ones, without a bias.
This follows the **full-model** defaults in
[FLA GatedDeltaNetConfig](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/gated_deltanet/configuration_gated_deltanet.py),
which [GatedDeltaNetBlock](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/gated_deltanet/modeling_gated_deltanet.py)
passes to the mixer. The standalone layer constructor defaults to `1e-5`,
but that value is overridden when constructing the full model. Gain shape and
initialization follow [FLA RMSNorm](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm.py).
Sources inspected 2026-09-23. Only affine parameters are shared; normalization
statistics remain independent for each token and head. Gradients of gamma
sum over tokens, batches and heads. This is an upstream convention, not a
claim of optimality or improved accuracy. Contract 16 had per-head gains;
contract 17 adopted the standalone epsilon `1e-5`. Contract 18 corrects that
to the full-model epsilon `1e-6`, retaining shared gains. Contract 19 replaces
the constrained T parameter with an unconstrained identity-centered Delta.
Older contracts are rejected rather than silently converted.
The encoder owns CPE, residual connections, MLPs and pooling.

## Key initialization

Since source 0.8.1, K projection entries are initialized independently from
Normal(0, 1/dim), where the second argument denotes variance; K bias is zero.
For unit-variance normalized inputs, this sets the expected mean eigenvalue of
K^T K/n to one, keeping the initialization scale independent of model width.
This is a statistical initialization criterion, not an accuracy guarantee.

`Ridgon.init_weights()` owns this rule and only initializes the K slice. It runs
on direct construction and after timm's depth-first initialization of child
Linear modules. Q/V and the output projection keep their existing initializer
(default Linear outside vision; timm std=0.02 and zero bias in the vision
factory). Delta, RMSNorm gains and encoder-owned CPE are not changed by this method.
No key rescaling or normalization is added to the forward pass.

Loading a checkpoint restores its saved K; it does not reinitialize that tensor.
The K-only initialization change in source 0.8.1 retained model contract 15.
The historical source 0.9.0 introduced constrained T under contract 16.
Current contract 19 uses unconstrained Delta initialized to zero; older
contracts are rejected on load.

## Exact memory interpretation

Define the ridge memory and the adjusted query by

```text
Mstar = argmin_M 0.5 ||V-Ak M||_F^2 + 0.5 ||M||_F^2
      = (I+Ak^T Ak)^-1 Ak^T V
Qeff = Aq R^-1 T R
O = Qeff Mstar
```

This equals the operator above, including gradients through R. Qeff and Mstar
are explanatory variables, not additional runtime computations. The memory is
the exact global ridge solution; T changes its query readout in the coordinates
set by the input's key statistics. Arbitrary T is not claimed to be the inverse
Hessian of that ridge objective. Intention already studies global ridge/query
readout; DeltaNet supplies the online error-correction connection and MesaNet
the regression/TTT connection. This operator is not an exact bidirectional
expansion of DeltaNet's ordered recurrence.

R, Z and the memory remain input-dependent even though T is shared. For finite
inputs, I+Ak^T Ak is positive definite in exact arithmetic. Singular T is valid
because it is never inverted. T has no fixed spectral or Frobenius bound.
Positive definiteness of the ridge system does not establish a contraction
of the normalized mixer or network.

## Identity-centered parameterization and training

Delta initializes to zero, so T starts at I and the initial readout is exactly
Aq (I+Ak^T Ak)^-1 Ak^T V. More generally,

```text
O = O_ridge + Pq Delta Pk^T V
O_ridge = Aq (I+Ak^T Ak)^-1 Ak^T V
```

This is an algebraic decomposition, not two runtime readouts. The reference
and CUDA primitives both receive the effective T and evaluate the same compact
products as before. Since dT/dDelta is the identity, dLoss/dDelta = dLoss/dT.
The parameter count and identity initial function are unchanged from contract 18.

Use ordinary [PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
without model-specific hooks. Delta belongs to the regular weight-decay group.
The [ViT³-derived ImageNet recipe](IMAGENET_VIT3.md) uses fused AdamW with
weight decay 0.05. The [DeiT III-derived recipes](IMAGENET_DEIT3.md) use fused
LAMB for pretraining and fused AdamW for resolution fine-tuning, with
recipe-specific decay. For AdamW, if u is the bias-corrected adaptive gradient
update, the effective map changes as

```text
Delta_next = (1 - lr * wd) Delta - lr * u
T_next = I + (1 - lr * wd) (T - I) - lr * u
```

Thus decoupled weight decay pulls T toward I, not zero. This is an identity
prior in the update; it is not an assertion that AdamW exactly optimizes an
L2-regularized objective or that learned Delta necessarily improves accuracy.
There is no gradient projection, step retraction, moment transport or custom
optimizer state. Standard optimizer state_dict handles resume and AMP skips.

Any real T is representable as I + Delta. Q -> s Q, T -> T/s remains an exact
scale freedom of the unregularized forward function. This parameterization
does not remove that freedom; it prioritizes ordinary optimization and an
identity-centered decay rule over a sphere constraint. No L2 normalization is
added to Q, K or V. The new optimization trajectory differs from contract 18,
so old training checkpoints are rejected.

## Equivalent implementation without P

```text
G = K^T K / n
Bkv = K^T V / sqrt(n)
F F^T = I + G                 F = R^T
Z = solve_triangular(F, Bkv)
M = solve_triangular(F^T, T Z)
O = Q M / sqrt(n)
```

Only key-statistic Cholesky and triangular solves remain. No J, L, Omega,
shared-core LU or inverse-map adjoint is evaluated. This algorithm covers
N<r, N=r and N>r and never materializes token-sized P.

For upstream E at O, token adjoints are

```text
dM = Q^T E / sqrt(n)
dQ = E M^T / sqrt(n)
dK = K (dG+dG^T) / n + V dBkv^T / sqrt(n)
dV = K dBkv / sqrt(n)
```

The compact adjoint differentiates both triangular solves and Cholesky.
If dU is the gradient at U=T Z, dT=sum_samples(dU Z^T) and dZ=T^T dU.
The reference and CUDA operator return ordinary derivatives with respect to T;
the model passes them unchanged to Delta through the identity addition.

## Masking and precision

Boolean [B,N] masks exclude rows before/after biased projection. Counts clamp
to one for entirely masked samples. Invalid outputs are zeroed after output
projection; excluded NaNs cannot enter neighboring rows. Position operations
must separately honor masks.

FP64 is the mathematical oracle. Production projections use BF16 operands and
FP32 accumulation. Statistics, factors, compact products, raw readout, RMS
reductions and optimizer updates use FP32. Normalized outputs round to BF16
before Wo. Parameters remain FP32; public output dtype follows input dtype.
Old model contracts are rejected, including under strict=False. Historical
paper/results retain their recorded operators, not this contract.
