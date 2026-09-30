# Architecture

Dependencies point inward: experiments and framework integrations call the
single Ridgon model, which calls either the mathematical reference or the
explicit CUDA implementation.

| Owner | Responsibility |
| --- | --- |
| `ridgon/ball/config.py` | Dimensions, heads, rank and projection bias |
| `ridgon/ball/reference.py` | Q/K/V memory equations, FP64 oracle and precision boundaries |
| `ridgon/ball/model.py` | Independent packed projections, identity-plus-delta shared query map, shared RMS channel weights, masks, checkpoint contract 19 |
| `ridgon/ball/cuda.py` | CUDA contract 17 validation, token tiling, compact analytic VJP, FLA-derived grouped RMSNorm |
| `integrations/timm.py` | ViT³-style vision encoder, packed SwiGLU, CPE, token-LN mean pooling and AMP boundaries |
| `integrations/openmmlab.py` | Dense framework registration and padded-image plumbing |
| `integrations/transformers.py` | HF BERT/MLM registration, packed SwiGLU, Post-LN adapter, padding and checkpoint metadata |

There is one public operator and no mode selector. The learned core is
sample-independent, while the key Gram and key/value memory remain
input-dependent. The operator has no projection-local convolution or internal
position mechanism. The vision encoder adds residual CPE and token-LayerNorm mean pooling.

No custom native library is built or loaded. Token and normalization kernels separately
JIT-compile through Triton on first use. CPU reference import needs no Triton.
The shared-core LU implementation and its build/package tools have been removed.

See [mathematics](CORE_CONTRACT.md), [CUDA](CUDA_CONTRACT.md) and
[ViT³-derived ImageNet training](IMAGENET_VIT3.md) and
[DeiT III-derived ImageNet training](IMAGENET_DEIT3.md).
