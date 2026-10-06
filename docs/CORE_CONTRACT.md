# Q/K/V ridge memory and learned query readout

Model contract **23**, source version **0.14.0**. Independent Q/K/V projections,
centered Q/K depthwise convolutions, a shared query map T = I + Delta, and
per-head RMSNorm and a low-rank sigmoid output gate define one operator.
There is no input-conditioned core generator, reflected readout, internal value
skip, V convolution or mode selector. Vision adds fixed axial 2D RoPE after
Q/K convolution; 1D models retain their existing external position embeddings.

## Definition

For one head, n valid tokens, key/query rank r and value width d:
convolution uses the original sequence/grid coordinates, with padding masked;
the displayed matrices below retain only the valid rows.

```text
Q = DWConv_q(X Wq + bq)       [n,r]
K = DWConv_k(X Wk + bk)       [n,r]
V = X Wv + bv                 [n,d]
Q,K = RoPE_2D(Q), RoPE_2D(K)  vision only, before both Gram and KV statistics
Ak = K / sqrt(n), Aq = Q / sqrt(n)
R^T R = I + Ak^T Ak           upper Cholesky, positive diagonal
Pk = Ak R^-1, Pq = Aq R^-1
Z = Pk^T V
T = I + core_delta           [r,r], shared across samples
O = Pq T Z
Gout = sigmoid((X Wdown) Wup + bg)  [n,D], D = H*d
Y = (concat_h[RMSNorm_h(O)] * Gout) Wo + bo
```

DWConv is centered, stride one, zero padded, bias free, and has no activation.
Sequence models default to independent width-3 filters per Q/K channel. Vision defaults to
3×3 filters on the explicit patch grid, including rectangular grids. Q and K
share a grouped convolution launch, not weights; V bypasses the convolution.
The added work is linear in token count. The ridge equations use the filtered
Q/K, so the memory interpretation and positive-definite system are unchanged.

Filters initialize to one at the center and zero elsewhere, preserving the
initial projection function and K variance while allowing every tap to learn.
This is an initialization choice, not a claim of an optimal local prior.
`qk_conv_kernel_size=k` is a positive odd integer (default 3); padding is k//2.
A kernel of size 1 is a learned channel scale without neighbor mixing.
Parameters add `2*H*r*k` per 1D layer or `2*H*r*k*k` per 2D layer. The core config
selects `qk_conv_dim=1` (default) or `2`; integrations fix this from the data layout,
and expose the kernel size separately. 2D forward requires
`spatial_shape=(height, width)` with `height*width=N`. There are no prefix tokens.
Both backends use native PyTorch convolution. CUDA evaluates the 1D case as a
channels-last 1 x N image with a 1 x k filter, preserving its centered
neighborhood and padding. See the depthwise definition in
[PyTorch Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html).
[FLA ShortConvolution](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/conv/short_conv.py)
is causal; its boundary semantics are not reused for this bidirectional filter.

Vision follows the fixed axial convention in
[RoPE-ViT](https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py):
each head has r/4 adjacent channel pairs for horizontal position, followed by
r/4 pairs for vertical position, with frequencies `100**(-4*j/r)`. Rank must
be divisible by four. Coordinates are integer patch indices `(x,y)` in the
actual row-major `(height,width)` grid; frequencies are shared across heads.
No frequency training, normalized coordinates, resolution interpolation or
additional learned position table is used. V is unchanged. In column-vector
notation, `q_p = Rotation(p) sum_delta D_delta q_raw[p+delta]` (and likewise K).
The same rotated K enters both K^T K and K^T V. Orthogonal rotation preserves
each key norm and hence the trace initialization criterion, while changing
its feature covariance. The ridge system remains positive definite. The
general T-corrected readout is not claimed to depend only on relative offsets.

Phases are computed in FP32 (FP64 for the oracle), independently of AMP. The
reference rotates pairs in that dtype and rounds to the projection dtype once.
CUDA uses the same equations in registers. Derived phase tables are cached by
both grid dimensions, device and dtype; caches are excluded from checkpoints
and DDP broadcasts. Warmed tables stay alive across resolution changes so a
captured training graph can be replayed after evaluation on another grid.

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
Contract 20 adds the centered Q/K convolutions.
Contract 21 adds vision RoPE after convolution and removes encoder CPE.
Contract 22 adds the low-rank sigmoid output gate and configurable odd Q/K kernel size.
Older contracts are rejected rather than silently converted.
The encoder owns residual connections, MLPs and pooling.

## Coordinate shifts and the learned core

Axial RoPE follows [RoPE-ViT](https://arxiv.org/abs/2403.13298). Its rotations
preserve each Q/K row norm, hence the trace of the key Gram, but do not preserve
the Gram's full spectrum when different rows receive different rotations.
The unit ridge still makes `I + K^T K/n` positive definite.

The following is a deduction for this operator, not a property established by
the RoPE-ViT experiments. For `T = I`, the readout reduces to
`O = Q (I + K^T K/n)^-1 K^T V/n`. Shifting all position coordinates by the
same offset rotates Q and K by a common orthogonal matrix, leaving this readout
unchanged when content and valid-token membership are held fixed.

For a general learned `T`, write `O = Pq T Pk^T V`. Under that same coordinate
shift, the Cholesky-whitened frames transform as `Pq' = Pq U`, `Pk' = Pk U`
for an orthogonal `U` depending on the key statistics and shift. The new output
is `Pq U T U^T Pk^T V`; an unconstrained shared T need not satisfy `U T U^T = T`.
Thus the current learned correction does not guarantee coordinate-origin
invariance. This applies to both Axial and Mixed RoPE. It neither invalidates
the solve nor establishes a loss of classification accuracy. Full-image
translation also changes convolution boundary effects, which this argument
deliberately excludes.

## Output selection

The gate receives the same masked mixer input X as the projections, before
convolution and RoPE. Vision supplies Pre-LN activations; BERT retains its
Post-LN scaffold. Wdown has shape [D,m] and Wup [m,D], with
`output_gate_rank=m` (default 32), independent of the memory rank r. There is
no intermediate activation, down bias, head sharing or sequence normalization.
The up bias has shape [D], independently of the QKV/output `bias` setting.
The gate adds `2*D*m+D` parameters per layer and linear work in sequence length.

Gate location and sigmoid follow [Gated Attention](https://arxiv.org/abs/2505.06708)
and its [official implementation](https://github.com/qiuzh20/gated_attention/blob/main/modeling_qwen3.py).
The post-RMS placement follows [FLA Gated DeltaNet](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py),
which uses a SiLU gate instead. Low-rank factorization is our cost choice, not
a claim that either upstream establishes rank 32 or a MemSolve accuracy gain.
The reference uses Tensor Core GEMMs and PyTorch sigmoid/multiply/autograd.
CUDA fuses sigmoid selection with the readout and RMSNorm, retaining the
reference's BF16 rounding boundaries in both forward and backward. The
compact solver remains outside TorchInductor; no extra native dependency
is introduced.

Both gate factors keep ordinary Linear/framework weight initialization; the
up bias is zero. Initial gates are near 0.5, not identity-preserving. No factor
of two, bias saturation, zero-initialized factor or checkpoint conversion is
introduced. Sigmoid is soft suppression, not guaranteed exact sparsity or
sparse execution. Multiplication follows head RMSNorm so normalization cannot
undo suppression, and precedes Wo so channels are selected before mixing heads.
The gate changes the emitted features; it does not change the ridge objective,
key statistics, memory solve or learned shared T.

## Key initialization

Since source 0.8.1, K projection entries are initialized independently from
Normal(0, 1/dim), where the second argument denotes variance; K bias is zero.
For unit-variance normalized inputs, this sets the expected mean eigenvalue of
K^T K/n to one, keeping the initialization scale independent of model width.
This is a statistical initialization criterion, not an accuracy guarantee.

`MemSolve.init_weights()` owns this rule and initializes the Q/K filters to
identity. It runs on direct construction and after timm's depth-first initialization of child
Linear modules. Q/V, both gate factors and the output projection keep their existing initializer
(default Linear outside vision; timm std=0.02 and zero bias in the vision
factory). Delta and RMSNorm gains are not changed by this method.
No key rescaling or normalization is added to the forward pass.

Loading a checkpoint restores its saved K; it does not reinitialize that tensor.
The K-only initialization change in source 0.8.1 retained model contract 15.
The historical source 0.9.0 introduced constrained T under contract 16.
Current contract 23 uses unconstrained Delta initialized to zero and identity
Q/K filters; older contracts are rejected on load.

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
The T reparameterization in contract 19 preserved contract 18's parameter count
and identity initial function; contract 20 additionally learns Q/K filters.

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

CUDA reuses the compact inverse of the Cholesky factor in FP32 GEMMs,
including its analytic VJP; the reference retains direct triangular solves.
These evaluate the same equations. No J, L, Omega,
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

Boolean [B,N] masks exclude rows before/after biased projection and after Q/K
convolution. Counts clamp to one for entirely masked samples. Invalid outputs are zeroed after output
projection; excluded NaNs cannot enter neighboring rows. Gaps retain their
original coordinates; compacting a gapped sequence changes its neighborhoods.
Learned convolutions are not equivariant to arbitrary token permutations.
Other position operations must separately honor masks.

FP64 is the mathematical oracle. Production QKV/gate projections use BF16
operands and FP32 accumulation. Q/K convolution uses BF16 activations and cast
weights with FP32 master parameters. Statistics, factors, compact products,
raw readout, RMS reductions and optimizer updates use FP32. RMS gain, sigmoid
and gating multiplication share FP32 arithmetic without an intermediate BF16
rounding. The gated readout rounds once to FP16 and Wo uses FP16 forward
operands, FP32 accumulation and FP32 bias before the public output cast.

The non-affine head-normalized readout is bounded by sqrt(head_dim); the gated
affine value is bounded by abs(gamma_i)*sqrt(head_dim). This is range-safe for
ordinary trained gains, not an unconditional bound on learned gamma or Wo.
FP16 forward intermediates stay inside a custom autograd boundary. Wo backward
uses BF16 operands/storage for the readout VJP and FP32 accumulation, so small
gradients are not forced through FP16. Operand casts are fused into the GEMM;
large reductions for dWo use split-K with FP32 partial sums. RMS and solve
backward arithmetic stays FP32; QKV/logit gradients store BF16. Master parameter
gradients remain FP32. CPU reference arithmetic stays FP32, and FP64 inputs
retain the mathematical oracle. Public output dtype follows input dtype.
Old model contracts are rejected, including under strict=False. Historical
paper/results retain their recorded operators, not this contract.
