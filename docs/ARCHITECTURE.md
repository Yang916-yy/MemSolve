# Architecture

Dependencies point inward:

~~~
experiments / integrations / benchmarks
                    |
                 lsso.ball
                    |
          PyTorch reference or strict CUDA op
~~~

config.py owns the two supported ablation axes, reference.py owns the QR frame,
accretive generator, and equilibrium mathematics, and model.py owns parameters
and nn.Module behavior. Framework adapters may not implement operator math.

## Execution and state ownership

| Owner | Responsibility |
| --- | --- |
| `lsso/ball/config.py` | Public geometry, DYNAMIC/STATIC/ZERO validation |
| `lsso/ball/reference.py` | Canonical math, FP64 oracle, mixed-precision projections and their VJPs; lazy Triton biased GEMM |
| `lsso/ball/model.py` | Parameters, validity masks, model checkpoint contract 13 |
| `lsso/ball/cuda.py` | Native loading/ABI 11 validation, unified no-frame Triton kernels and first-order autograd adapters |
| `csrc/ball/` | Precompiled per-SM MathDx mixer and native forward/backward storage |
| `integrations/timm.py` | Shared vision encoder with masked CPE and CLS pooling |
| `integrations/openmmlab.py` | Dense-task framework registration, feature maps and padded-image plumbing |

The CUDA execution path combines a Python projection boundary with a
common no-frame Triton/PyTorch mixer. The compact LU factor/solve calls reuse precompiled cuSOLVERDx kernels.
The materialized-frame native mixer remains an explicit low-level comparison
implementation, outside public dispatch.
Native artifact loading does not compile CUDA source; projections and the
no-frame token kernels separately JIT-compile through Triton on first use.
CPU reference imports remain independent of Triton. Unsupported native
contracts fail explicitly rather than selecting another operator.

Consult [the core contract](CORE_CONTRACT.md), [CUDA contract](CUDA_CONTRACT.md),
and [downstream protocol](DOWNSTREAM_PROTOCOLS.md) for their respective boundaries.
