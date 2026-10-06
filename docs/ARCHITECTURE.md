# Architecture

Dependencies point inward: experiments and framework integrations call the
single MemSolve model, which calls either the mathematical reference or the
explicit CUDA implementation.

| Owner | Responsibility |
| --- | --- |
| `memsolve/ball/config.py` | Dimensions, heads, rank, projection bias, Q/K convolution dimension/kernel size and output gate rank |
| `memsolve/ball/reference.py` | Q/K/V memory equations, FP64 oracle and precision boundaries |
| `memsolve/ball/model.py` | Independent packed projections, identity-plus-delta shared query map, shared RMS channel weights, masks, checkpoint contract 23 |
| `memsolve/ball/cuda.py` | CUDA contract 19 validation, token tiling, compact analytic VJP, FLA-derived grouped RMSNorm |
| `integrations/timm.py` | ViT³-derived vision encoder, packed SwiGLU, patch-grid plumbing, token-LN mean pooling and AMP boundaries |
| `integrations/openmmlab.py` | Dense framework registration and padded-image plumbing |
| `integrations/transformers.py` | HF BERT/MLM registration, packed SwiGLU, Post-LN adapter, padding and checkpoint metadata |

There is one public operator and no mode selector. The learned core is
sample-independent, while the key Gram and key/value memory remain
input-dependent. The operator applies centered depthwise convolution to Q/K
before the solve: width 3 for sequences, 3×3 for an explicit image patch grid.
V remains tokenwise. For vision, the core rotates Q/K with fixed axial 2D RoPE
after convolution and before statistics. The vision encoder uses token-LayerNorm
mean pooling and has no residual CPE. Phase tables are derived non-checkpoint state.

After per-head RMSNorm, a low-rank sigmoid channel gate selects the readout
before Wo. Its two linear maps reuse the projection GEMMs. PyTorch owns the
reference; CUDA fuses the gate and its gradients into the normalized readout.
Gate rank defaults to 32, independently
of memory rank. Centered convolution kernel size defaults to 3 and is configurable
as a positive odd integer; integrations choose dimension from their data layout.

No custom native library is built or loaded. Token and normalization kernels separately
JIT-compile through Triton on first use. CPU reference import needs no Triton.
The shared-core LU implementation and its build/package tools have been removed.

See [mathematics](CORE_CONTRACT.md), [CUDA](CUDA_CONTRACT.md) and
[ViT³-derived ImageNet training](IMAGENET_VIT3.md) and
[DeiT III-derived ImageNet training](IMAGENET_DEIT3.md).
