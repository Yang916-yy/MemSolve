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
| `lsso/ball/cuda.py` | Strict native loading, ABI 9 validation and autograd adapter |
| `csrc/ball/` | Precompiled per-SM MathDx mixer and native forward/backward storage |
| `integrations/timm.py` | Shared DeiT III encoder |
| `integrations/openmmlab.py` | Dense-task framework registration, feature maps and padded-image plumbing |

The CUDA execution path combines a Python projection boundary with a
precompiled native mixer. Native artifact loading does not compile CUDA source;
the biased projection separately JIT-compiles through Triton on first use.
CPU reference imports remain independent of Triton. Unsupported native
contracts fail explicitly rather than selecting another operator.

Consult [the core contract](CORE_CONTRACT.md), [CUDA contract](CUDA_CONTRACT.md),
and [downstream protocol](DOWNSTREAM_PROTOCOLS.md) for their respective boundaries.
