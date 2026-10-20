# MemSolve: From Sequential Memory Updates to Global Equilibrium

**Full-context associative memory through compact equilibrium solves.**

**MemSolve** is the operator; **MemSolve-ViT** and **MemSolve-BERT** adapt it to
vision and text. It combines an input-dependent relation frame with a learned
compact core shared across samples. The visual shell uses ordinary ViT3-style
mean readout and residuals. Earlier LSSO/Ridgon architectures remain in Git history.

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
per head. There is no independent query, Q/K convolution, output RMSNorm or
output gate. Optional RoVE rotates content before global aggregation and
inversely rotates its correction before the output projection; the relation
frame and compact solve are unchanged.

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
Optional [ViG-style dense vision recipes](docs/DOWNSTREAM_PROTOCOLS.md) attach
ViT-Adapter to pretrained MemSolve-ViT for Mask R-CNN and UPerNet; install
`.[vision,openmmlab]` plus compiled MMCV for these tasks.
The native backend requires the matching extension: Torch
`2.14.0+cu132`, CUDA 13.2, MathDx 26.06.1 and CMake >=3.25. For an A800/SM80,
activate the intended Python environment and build from the checkout:

```bash
PYTHON=python MEMSOLVE_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh
```

Set `CUDA_HOME` and `MATHDX_ROOT` if the toolkit locations differ from the
defaults in `tools/build_cuda.sh`. Select the target architecture listed in the
[CUDA guide](docs/CUDA_CONTRACT.md#build-and-runtime-artifacts). Native ABI 17
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
Parameters remain FP32. The native implementation uses BF16 packed
projections with FP32 compact statistics/solves; public outputs follow the input
dtype. Both modes share the optimized statistics/readout/VJP path, with
chunked long-sequence statistics and reuse of temporary backward buffers.
Saved tensors remain intact for repeated backward. See the CUDA guide for
[measured memory and resource changes](docs/CUDA_CONTRACT.md#measured-memory-and-resource-changes)
and the exact precision contract.

Exact sequence sharding is available from the source checkout through
`integrations.distributed.SequenceParallel`, including unequal shards and
centered-convolution halos. See the [sequence-parallel contract](docs/CUDA_CONTRACT.md#exact-sequence-parallelism)
for global loss normalization, gradient SUM, and CUDA Graph usage.

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

S has **19,876,336** parameters including its 1,000-class head.
The completed 300-epoch ImageNet-1K runs reach **81.0%** (Small, 20M)
and **82.7%** (Base, 78M), reporting best EMA validation accuracy.
These runs use the ViT3-derived AdamW/CE recipe with repeated augmentation
and global batch 2048; they differ from the 400-epoch LAMB/BCE recipe below.
Full protocols and comparisons are in the [paper draft](paper/main.pdf).
Historical LSSO checkpoints used different scaffolds or position mechanisms;
their accuracy is not assigned to the current model.

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
MLM head, with MemSolve mixers and SwiGLU FFNs. New configurations use full-head
RoVE (base 10000, unit amplitude, frequency-only YaRN beyond 128 tokens) and a
centered residual depthwise convolution (kernel 3) before each mixer.
The default Base geometry is width 768, 14 layers, 12 heads and rank 32:
**111,123,938** parameters including the tied MLM head, approximately matching
the parameter budget of 12-layer BERT-base. Set `memsolve_short_conv_kernel=0`
explicitly for the no-convolution ablation.

RoVE is optional: set `position_embedding_type="none"` to disable it with no
position table, or `"absolute"` to use BERT's learned positions. The standalone
operator uses `MemSolveConfig(rotary=True/False)`.

```python
from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM

model = MemSolveBertForMaskedLM(MemSolveBertConfig(
    memsolve_implementation="cuda",
)).cuda()
```

Importing the adapter registers `memsolve_bert` with Hugging Face `AutoConfig`,
`AutoModel`, `AutoModelForMaskedLM` and `AutoModelForSequenceClassification`. [Text workflows](docs/BERT_MLM.md) cover
C4 MLM, LION M2-BASE GLUE fine-tuning, Sentence Transformers retrieval and MTEB/LongEmbed
evaluation. They include CPU preflight, independent fixed-mask MLM evaluation,
best-validation export, local full-corpus C4 streaming, and continued pretraining
from a checkpoint. The full
default is 70,000 updates at sequence length 128 and global batch 4096,
with Inductor fusion and CUDA Graph forward/backward capture; measure runtime
before launch. DDP communication and optimization remain in HF Trainer.
Completed MLM/GLUE runs used learned absolute positions; their saved configs
retain that architecture. RoVE removes the learned position-table limit, but
long-window quality still requires measurement. Both modalities call the same
MemSolve operator and CUDA path.

## Documentation

| Topic | Document |
| --- | --- |
| Equations, parameterization and masks | [Core contract](docs/CORE_CONTRACT.md) |
| CUDA precision, upstream reuse and Graph execution | [CUDA contract](docs/CUDA_CONTRACT.md) |
| Code ownership | [Architecture](docs/ARCHITECTURE.md) |
| ImageNet, ViT³-derived training recipe | [ViT³-derived protocol](docs/IMAGENET_VIT3.md) |
| ImageNet, RoPE-ViT 400e (Base: +20e), optional DeiT III | [LAMB training protocols](docs/IMAGENET_DEIT3.md) |
| BERT, C4 MLM, GLUE, retrieval and MTEB/LongEmbed | [NLP launch guide](docs/BERT_MLM.md) |
| GenomicBenchmarks and Long Range Arena | [Sequence experiments](docs/SEQUENCE_EXPERIMENTS.md) |
| Assembly101 integration | [Assembly101](docs/ASSEMBLY101.md) |
| Food-101 training | [Food-101](docs/FOOD101.md) |
| Mechanism studies and additional experiment priorities | [Research scope](docs/ABLATIONS.md) |
| Validation and historical measurements | [Tests](tests/README.md), [results](results/README.md) |

## Research status and provenance

Current model checkpoint contract: **26**; native CUDA contract: **17**;
source version: **0.15.0**. The current API is `MemSolve` / `MemSolveConfig`.
Older model checkpoints and the former `lsso` namespace are not supported.

The [paper draft](paper/main.pdf) describes the current shared-frame
Equilibrium / Complement operator, its compact CUDA computation, and completed
Small/Base ImageNet-1K and parameter-matched BERT/GLUE experiments.
Exploratory ImageNet-100 runs are excluded from the manuscript.
The paper remains a working draft, with two-column main text and references
and a single-column appendix. It does not yet use a conference template. The
[archived results](results/README.md) retain their original names and source
contracts and are not relabeled as measurements of the current architecture.
Editable PDF/SVG figures are generated by [paper/figures.py](paper/figures.py).
The diagrams use a shared STIX font family and 9pt labels.
The convergence plot compares ordinary-weight MemSolve-S validation with the
[DeiT authors' four-GPU Small run](https://gist.github.com/fmassa/0dbd0184a0adb904ef42277b487d8b53),
both for 300 epochs. It uses each run's own training recipe and shows every
recorded epoch without smoothing. Final ImageNet table results use EMA.

Rebuild the three diagrams without experiment data:

```bash
python paper/figures.py --schematics-only --output paper \
  --audit-dir /path/outside/repository/figure-audit
```

To also rebuild the convergence plot, replace `--schematics-only` with
`--evidence /path/to/figure-evidence.json`. The evidence file points to local
logs and records their SHA256 hashes; its schema is described in the plotting
script. Python requires Matplotlib and NumPy. Raw logs, evidence manifests,
writing notes, and preview PNGs stay outside Git.
Layout references for the original vector artwork include
[VGGT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html)
and [MambaVision](https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html),
using the [academic-diagram guide](https://github.com/plbbl/academic-diagram-skill).

Appendix B includes forward, rank-space backward, and shared-core gradient
pseudocode. Build the manuscript with `latexmk -pdf -cd paper/main.tex`;
the LaTeX dependencies include `algorithm`, `algpseudocode`, `titlesec`,
`caption`, `placeins`, and Times fonts. On Debian/Ubuntu these are provided
by `texlive-science`, `texlive-latex-extra`, and `texlive-fonts-recommended`.

MemSolve draws on the error-correcting memory and test-time regression viewpoints
of DeltaNet and MesaNet. Global ridge regression with query readout also has
prior foundations in Intention. The current operator is not an exact
bidirectional expansion of DeltaNet's ordered recurrence.

Code is licensed under [Apache-2.0](LICENSE). Upstream and historical
attribution notices remain in [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); redistributed license texts
are included in source and wheel distributions.
The CUDA guide records the upstream sources and reuse boundaries.
