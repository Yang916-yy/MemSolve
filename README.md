# Ridgon

**Ridgon: From Sequential Memory Updates to Global Equilibrium**

Ridgon builds a compact ridge-regression memory from keys and values and reads it
with independent queries transformed in the key-statistic coordinates. At a
fixed rank its token work scales linearly with sequence length.

The current model uses **independent Q/K/V, a shared query map
T = I + Delta with zero-initialized learned Delta, and per-head RMSNorm**. Key statistics and memory
depend on each input. The vision encoder uses residual CPE and per-token final
LayerNorm followed by mean pooling, without CLS or LayerScale. The vision
FFN uses packed SwiGLU (gate width 1024 for S).

This refactor changes the architecture. The [paper](paper/main.pdf) and
[recorded results](results/README.md) concern earlier source contracts. They
are historical evidence, not accuracy or contraction claims for this model.
No new classification result is claimed by this source revision.

The current package is `ridgon-operator`, imported as `ridgon`; the public
classes are `Ridgon` and `RidgonConfig`. This replaces the former `lsso`
namespace without compatibility aliases. Historical artifacts retain LSSO
labels. The paper has the new title and an explicit working-draft note;
its technical body still needs revision for the current model.

## Quick start

```python
import torch
from ridgon import Ridgon, RidgonConfig

layer = Ridgon(RidgonConfig(dim=192, num_heads=3, rank=16)).cuda()
x = torch.randn(8, 65, 192, device="cuda", dtype=torch.bfloat16)
y = layer(x)  # mathematical reference

from ridgon.ball import cuda
cuda.load(device=x.device)
y = layer(x, implementation="cuda")
```

Train with ordinary fused AdamW:

```python
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, fused=True)
# loss.backward(); optimizer.step()
```

The stored `core_delta` starts at zero. Forward uses `T = I + core_delta`;
weight decay on Delta pulls T toward I. No optimizer hooks or norm constraint
are needed. Repository launchers include Delta in their regular decay group.

The CUDA path supports rank 16/32/48/64 and boolean validity masks. BF16
projections feed FP32 statistics/compact products and a shared no-P readout.
Parameters remain FP32. Unsupported contracts fail explicitly.

Install with CUDA PyTorch 2.14 and its matching Triton runtime:

```bash
python -m pip install -e '.[vision]'
```

No custom CUDA extension, MathDx or CMake build is required. Current model
checkpoint contract: **19**; CUDA algorithm contract: **17**; source: **0.11.0**.
Old model checkpoints are rejected. Local GPU validation targets SM80/A800
with PyTorch `2.14.0+cu132`.

## Documentation

| Topic | Document |
| --- | --- |
| Equations, parameterization and masks | [Core contract](docs/CORE_CONTRACT.md) |
| CUDA precision, upstream reuse and Graph execution | [CUDA contract](docs/CUDA_CONTRACT.md) |
| Code ownership | [Architecture](docs/ARCHITECTURE.md) |
| ImageNet, ViT³-derived training recipe | [ImageNet](docs/IMAGENET_VIT3.md) |
| ImageNet, DeiT III 400/800 + 20 epochs | [DeiT III](docs/IMAGENET_DEIT3.md) |
| Sequence training | [Sequence experiments](docs/SEQUENCE_EXPERIMENTS.md) |
| BERT, C4 MLM, retrieval fine-tuning and MTEB/LongEmbed | [NLP integration and launch guide](docs/BERT_MLM.md) |
| Assembly101 integration | [Assembly101](docs/ASSEMBLY101.md) |
| Food-101 training | [Food-101](docs/FOOD101.md) |
| Future component studies | [Research scope](docs/ABLATIONS.md) |
| Validation and previous measurements | [Tests](tests/README.md), [results](results/README.md) |

The grouped RMSNorm implementation borrows from FLA under MIT; attribution and
license text are in [NOTICE](NOTICE). PyTorch owns the matrix
factorization/solve primitives. See the CUDA contract for the pinned upstream
source and the exact reuse boundary.
