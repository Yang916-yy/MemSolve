# MemSolve: From Sequential Memory Updates to Global Equilibrium

**Full-context associative memory through compact equilibrium solves.**

This research checkout restores the no-RankRotary LSSO operator from `e769a9a`.
Its visual shell uses ordinary ViT3-style mean readout and residuals. Names remain **MemSolve**,
**MemSolve-ViT** and **MemSolve-BERT**. The independent-Q/K/V redesign is retained
in Git history, not mixed into this implementation.

## How it works

Each head projects relation B and content C, forms a soft frame P from
`B / sqrt(valid_tokens)`, and compresses content as `Z = P^T C`.
The default `equilibrium` mode learns an accretive compact generator
`J = L L^T + Omega` shared across samples. Its exact readout is

```text
S = solve(I + J, Z)
Y = eta C + P [2 S - (1 + eta) Z]
```

The compact core map is solved once per head and reused across the batch.
`complement` removes the learned compact readout, retaining `eta (C - P P^T C)`. The scalar complement eta is learned
per head. There is no independent query, Q/K convolution, internal positional
rotation, output RMSNorm or output gate.

The reference uses QR; the native path evaluates an equivalent compact form
without materializing token-size P. See [the equations](docs/CORE_CONTRACT.md),
[architecture](docs/ARCHITECTURE.md), and [CUDA contract](docs/CUDA_CONTRACT.md).

## Install and use

Use Python 3.10–3.12 with CUDA PyTorch 2.14 and its matching Triton runtime.
Local GPU validation uses PyTorch `2.14.0+cu132` on SM80/A800.

```bash
git clone https://github.com/Yang916-yy/MemSolve.git
cd MemSolve
python -m pip install -e '.[vision]'
```

Use `.[nlp]` for the BERT workflows or `.[sequence]` for genomic/LRA training.
Framework integrations and experiment entrypoints run from the source checkout.
The native backend requires the matching extension: Torch
`2.14.0+cu132`, CUDA 13.2, MathDx 26.06.1 and CMake >=3.25. For an A800/SM80,
activate the intended Python environment and build from the checkout:

```bash
PYTHON=python MEMSOLVE_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh
```

Set `CUDA_HOME` and `MATHDX_ROOT` if the toolkit locations differ from the
defaults in `tools/build_cuda.sh`. Select the target architecture listed in the
[CUDA guide](docs/CUDA_CONTRACT.md#build-and-runtime-artifacts). Native ABI 16
requires rebuilding older extensions; the model checkpoint contract remains 26.

The reference also runs on CUDA and needs no extension. LAMB training recipes
separately require NVIDIA Apex.

```python
import torch
from memsolve import MemSolve, MemSolveConfig
from memsolve.ball import cuda

layer = MemSolve(MemSolveConfig(dim=192, num_heads=3, rank=16)).cuda()
x = torch.randn(8, 65, 192, device="cuda", dtype=torch.bfloat16)

cuda.load(device=x.device)
y = layer(x, implementation="cuda")
y_reference = layer(x, implementation="reference")
```

CUDA supports ranks **16 / 32 / 48 / 64** and boolean validity masks.
Parameters remain FP32. The restored native implementation uses BF16 packed
projections with FP32 compact statistics/solves; public outputs follow the input
dtype. See the CUDA guide for exact precision and supported boundaries.

Train with an ordinary optimizer, for example fused AdamW:

```python
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, fused=True)
# loss.backward(); optimizer.step()
```

## Vision models

The default visual scaffold uses ordinary ViT3-style readout and residuals:
patch16, 12 blocks, residual 3x3 depthwise CPE in every block, Pre-LayerNorm,
GELU MLP and per-token final LayerNorm followed by mean pooling.
Like ordinary ViT3, it has no CLS token or LayerScale.
There is no RankRotary or absolute patch position table.

| Model | Width | Heads | Rank | MLP width |
| --- | ---: | ---: | ---: | ---: |
| MemSolve-ViT-T | 192 | 6 | 16 | 768 |
| MemSolve-ViT-S | 384 | 6 | 32 | 1584 |
| MemSolve-ViT-B | 768 | 12 | 48 | 3072 |

S has **19,876,336** parameters including its 1,000-class head. The operator
is restored from historical Small; mean pooling and removal of LayerScale
change that run's visual scaffold, so its measured accuracy is not inherited. The historical Base result used a different
position mechanism; it is not an exact result for this restored configuration.

Training recipes specify their own
DropPath, optimizer, augmentation and schedule.

[Default ImageNet training](docs/IMAGENET_DEIT3.md) follows RoPE-ViT:
T/S train at 224px for 400 epochs; Base trains at 192px for 400 epochs and
fine-tunes at 224px for 20 epochs. Pretraining uses fused LAMB, 3-Augment,
DeiT BCE and global batch 2048; Base fine-tuning uses fused AdamW and batch 512.
The LR schedule follows upstream: nominal 5-epoch warmup and one zero-based
`step(epoch)` after each training epoch, for both pretraining and fine-tuning.
EMA uses upstream's constant decay 0.99996 and updates after optimizer steps.
Validation records ordinary and EMA metrics, with separate `checkpoint_best.pt`
and `checkpoint_best_ema.pt` files. Fine-tuning loads the selected weight set;
resume restores both sets with optimizer state tied to ordinary weights.
[ViT³-derived training](docs/IMAGENET_VIT3.md) remains an explicit 300-epoch, 224px recipe
for T/S/B. [DeiT III-derived training](docs/IMAGENET_DEIT3.md) provides 400- and
800-epoch, 192px Base recipes followed by 20 epochs at 224px. These are adapted
protocols, not claims of identical upstream architectures or training settings.
The entrypoint supports DDP, gradient accumulation, persistent data workers,
CUDA Graphs and optional TorchInductor fusion.

From the source checkout, import the adapter to register the three models with
timm. The registered names are `memsolve_vit_tiny`, `memsolve_vit_small` and
`memsolve_vit_base` (MemSolve-ViT-T/S/B):

```python
import timm
import integrations.timm  # registers MemSolve-ViT
from memsolve.ball import cuda

cuda.load()
model = timm.create_model(
    "memsolve_vit_small", pretrained=False, img_size=224,
    implementation="cuda",
).cuda()
```

Use BF16 autocast for CUDA execution and retain FP32 model parameters. The
registered models default to the reference backend unless `implementation` is
specified. No pretrained weights for this model contract are published yet.

## Text model

**MemSolve-BERT** retains BERT's embeddings, Post-LayerNorm residuals and tied
MLM head, with MemSolve mixers and SwiGLU FFNs. It uses learned absolute positions and the restored
relation/content mixer without short convolution. The default Base geometry is
width 768, 12 layers, 12 heads and rank 32.

```python
from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM

model = MemSolveBertForMaskedLM(MemSolveBertConfig(
    memsolve_implementation="cuda",
)).cuda()
```

Importing the adapter registers `memsolve_bert` with Hugging Face `AutoConfig`,
`AutoModel` and `AutoModelForMaskedLM`. [Text workflows](docs/BERT_MLM.md) cover
C4 MLM, Sentence Transformers retrieval fine-tuning and MTEB/LongEmbed
evaluation. Both modalities call the same MemSolve operator and CUDA path.

## Documentation

| Topic | Document |
| --- | --- |
| Equations, parameterization and masks | [Core contract](docs/CORE_CONTRACT.md) |
| CUDA precision, upstream reuse and Graph execution | [CUDA contract](docs/CUDA_CONTRACT.md) |
| Code ownership | [Architecture](docs/ARCHITECTURE.md) |
| ImageNet, ViT³-derived training recipe | [ViT³-derived protocol](docs/IMAGENET_VIT3.md) |
| ImageNet, RoPE-ViT 400e (Base: +20e), optional DeiT III | [LAMB training protocols](docs/IMAGENET_DEIT3.md) |
| BERT, C4 MLM, retrieval fine-tuning and MTEB/LongEmbed | [NLP launch guide](docs/BERT_MLM.md) |
| GenomicBenchmarks and Long Range Arena | [Sequence experiments](docs/SEQUENCE_EXPERIMENTS.md) |
| Assembly101 integration | [Assembly101](docs/ASSEMBLY101.md) |
| Food-101 training | [Food-101](docs/FOOD101.md) |
| Future component studies | [Research scope](docs/ABLATIONS.md) |
| Validation and historical measurements | [Tests](tests/README.md), [results](results/README.md) |

## Research status and provenance

Current model checkpoint contract: **26**; native CUDA contract: **16**;
source version: **0.15.0**. The current API is `MemSolve` / `MemSolveConfig`.
Older model checkpoints and the former `lsso` namespace are not supported.

The [paper draft](paper/main.pdf) carries the MemSolve title, but its technical
body, figures and experimental tables still describe the earlier LSSO model.
The [archived results](results/README.md) retain their original names and source
contracts. They are not measurements or theoretical guarantees for the current
MemSolve architecture. The current equations are documented in the core contract;
new task results require their own training records and provenance.

MemSolve draws on the error-correcting memory and test-time regression viewpoints
of DeltaNet and MesaNet. Global ridge regression with query readout also has
prior foundations in Intention. The current operator is not an exact
bidirectional expansion of DeltaNet's ordered recurrence.

Code is licensed under [Apache-2.0](LICENSE). Upstream and historical
attribution notices remain in [NOTICE](NOTICE).
The CUDA guide records the upstream sources and reuse boundaries.
