# Ridgon: From Sequential Memory Updates to Global Equilibrium

**Bidirectional associative memory through compact global ridge solves.**

Ridgon turns the associations in a complete input into a shared memory and
reads that memory with independent queries. Inspired by error-correcting memory
and test-time regression, it computes a global ridge solution with a direct
solve in feature space. At fixed rank, forward and backward costs grow linearly
with sequence length.

The repository provides a PyTorch reference, a Triton/CUDA implementation, and
vision and BERT integrations. The public package is `ridgon-operator`, imported
as `ridgon`.

## How it works

For one head, let `Q`, `K` and `V` be independent projections of the input, and
let `n` be its valid-token count. Set `Aq = Q / sqrt(n)` and
`Ak = K / sqrt(n)`. The global memory is

```text
M* = argmin_M  0.5 ||V - Ak M||²_F + 0.5 ||M||²_F
   = (I + Akᵀ Ak)⁻¹ Akᵀ V
```

All valid positions contribute to the same memory. The matrix to factor is
`r × r`, independent of sequence length. The current readout adds a learned
correction in coordinates defined by the key statistics:

```text
Rᵀ R = I + Akᵀ Ak
Pq = Aq R⁻¹,  Pk = Ak R⁻¹
T = I + Delta                         # one learned matrix per head
O = Pq T Pkᵀ V = Aq M* + Pq Delta Pkᵀ V
```

`Delta` starts at zero, so the initial readout is exactly the ridge solution
queried by `Aq`. Key statistics and memory depend on the input; `Delta` is shared
across samples. Per-token, per-head RMSNorm and an output projection complete
the mixer. RMSNorm's learned channel gain is shared across heads.

The fast path evaluates an equivalent compact form without materializing `Pq`,
`Pk` or a token-by-token interaction matrix. It uses PyTorch Cholesky/triangular
solves and selectively fused Triton kernels. See the
[core equations](docs/CORE_CONTRACT.md) and [CUDA contract](docs/CUDA_CONTRACT.md)
for the gradient, masking and precision details.

## Install and use

Use Python 3.10–3.12 with CUDA PyTorch 2.14 and its matching Triton runtime.
Local GPU validation uses PyTorch `2.14.0+cu132` on SM80/A800.

```bash
git clone https://github.com/Yang916-yy/Ridgon.git
cd Ridgon
python -m pip install -e '.[vision]'
```

Use `.[nlp]` for the BERT workflows or `.[sequence]` for genomic/LRA training.
Framework integrations and experiment entrypoints run from the source checkout.
The operator requires no custom CUDA extension, MathDx or CMake build; the
DeiT III pretraining recipe separately requires NVIDIA Apex for fused LAMB.

```python
import torch
from ridgon import Ridgon, RidgonConfig
from ridgon.ball import cuda

layer = Ridgon(RidgonConfig(dim=192, num_heads=3, rank=16)).cuda()
x = torch.randn(8, 65, 192, device="cuda", dtype=torch.bfloat16)

cuda.load(device=x.device)
y = layer(x, implementation="cuda")
y_reference = layer(x, implementation="reference")
```

CUDA supports ranks **16 / 32 / 48 / 64** and boolean validity masks.
Parameters remain FP32; the operator uses BF16 projections and FP32 statistics,
solves and normalization. Public outputs follow the input dtype. Unsupported
contracts fail explicitly; numerical boundaries are recorded in the CUDA guide.

Train with an ordinary optimizer, for example fused AdamW:

```python
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, fused=True)
# loss.backward(); optimizer.step()
```

Repository launchers place `core_delta` in the regular weight-decay group.
Under AdamW, this pulls `T` toward identity. No model-specific optimizer hook
or norm constraint is required.

## Vision models

All three models use patch size 16, 12 blocks, residual 3×3 depthwise CPE,
Pre-LayerNorm blocks, packed SwiGLU, and token-LayerNorm followed by mean pooling.
The SwiGLU branch width is `8/3` of the embedding width. There is no CLS token
or LayerScale.

| Model | Width | Heads | Rank | SwiGLU branch width | Parameters¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridgon-T | 192 | 6 | 16 | 512 | 5,279,656 |
| Ridgon-S | 384 | 6 | 32 | 1024 | 20,327,272 |
| Ridgon-B | 768 | 12 | 48 | 2048 | 83,309,032 |

¹ Including the 1,000-class ImageNet head. Training recipes specify their own
DropPath, optimizer, augmentation and schedule.

[ViT³-derived training](docs/IMAGENET_VIT3.md) provides a 300-epoch, 224px recipe
for T/S/B. [DeiT III-derived training](docs/IMAGENET_DEIT3.md) provides 400- and
800-epoch, 192px Base recipes followed by 20 epochs at 224px. These are adapted
protocols, not claims of identical upstream architectures or training settings.
The entrypoint supports DDP, gradient accumulation, persistent data workers,
CUDA Graphs and optional TorchInductor fusion.

## Documentation

| Topic | Document |
| --- | --- |
| Equations, parameterization and masks | [Core contract](docs/CORE_CONTRACT.md) |
| CUDA precision, upstream reuse and Graph execution | [CUDA contract](docs/CUDA_CONTRACT.md) |
| Code ownership | [Architecture](docs/ARCHITECTURE.md) |
| ImageNet, ViT³-derived training recipe | [ViT³-derived protocol](docs/IMAGENET_VIT3.md) |
| ImageNet, DeiT III 400/800 + 20 epochs | [DeiT III-derived protocols](docs/IMAGENET_DEIT3.md) |
| BERT, C4 MLM, retrieval fine-tuning and MTEB/LongEmbed | [NLP launch guide](docs/BERT_MLM.md) |
| GenomicBenchmarks and Long Range Arena | [Sequence experiments](docs/SEQUENCE_EXPERIMENTS.md) |
| Assembly101 integration | [Assembly101](docs/ASSEMBLY101.md) |
| Food-101 training | [Food-101](docs/FOOD101.md) |
| Future component studies | [Research scope](docs/ABLATIONS.md) |
| Validation and historical measurements | [Tests](tests/README.md), [results](results/README.md) |

## Research status and provenance

Current model checkpoint contract: **19**; CUDA algorithm contract: **17**;
source version: **0.11.0**. The current API is `Ridgon` / `RidgonConfig`.
Older model checkpoints and the former `lsso` namespace are not supported.

The [paper draft](paper/main.pdf) carries the Ridgon title, but its technical
body, figures and experimental tables still describe the earlier LSSO model.
The [archived results](results/README.md) retain their original names and source
contracts. They are not measurements or contraction guarantees for the current
Ridgon architecture. The current equations are documented in the core contract;
new task results require their own training records and provenance.

Ridgon draws on the error-correcting memory and test-time regression viewpoints
of DeltaNet and MesaNet. Global ridge regression with query readout also has
prior foundations in Intention. The current operator is not an exact
bidirectional expansion of DeltaNet's ordered recurrence.

Code is licensed under [Apache-2.0](LICENSE). The grouped RMSNorm implementation
adapts FLA code under MIT; its attribution and license are in [NOTICE](NOTICE).
The CUDA guide records the upstream sources and reuse boundaries.
