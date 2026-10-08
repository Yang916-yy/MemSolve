# MemSolve: From Sequential Memory Updates to Global Equilibrium

**Full-context associative memory through compact equilibrium solves.**

MemSolve turns the associations in a complete input into a shared memory and
reads that memory with independent queries. Inspired by error-correcting memory
and test-time regression, it computes a global ridge solution with a direct
solve in feature space. At fixed rank, forward and backward costs grow linearly
with sequence length.

The repository provides a PyTorch reference, a Triton/CUDA implementation, and
vision and BERT integrations. The public package is `memsolve-operator`, imported
as `memsolve`. MemSolve names the operator; MemSolve-ViT and MemSolve-BERT
name its vision and language integrations.

## How it works

For one head, let `Q` and `K` be independently projected and depthwise-filtered
features, and `V` a tokenwise value projection. Q/K filters are centered width-3
for sequences or 3×3 on an explicit image patch grid, initialized to identity.
Vision adds a learned 2D absolute position table to patch embeddings before
the first block. The operator does not rotate Q/K.
Let `n` be its valid-token count. Set `Aq = Q / sqrt(n)` and
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
across samples. Per-token, per-head RMSNorm is followed by a low-rank sigmoid
channel gate and the output projection. The gate is `sigmoid((X Wdown) Wup + b)`;
its bottleneck width defaults to 32, independent of memory rank. RMSNorm's learned
channel gain is shared across heads; gate values are token/channel specific.

The fast path evaluates an equivalent compact form without materializing `Pq`,
`Pk` or a token-by-token interaction matrix. It uses PyTorch Cholesky/triangular
solves and selectively fused Triton kernels. See the
[core equations](docs/CORE_CONTRACT.md) and [CUDA contract](docs/CUDA_CONTRACT.md)
for the gradient, masking and precision details.

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
The operator requires no custom CUDA extension, MathDx or CMake build; the
LAMB recipes (including the T/S default) separately require NVIDIA Apex for fused LAMB.

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
Parameters remain FP32; the operator uses BF16 QKV/gate projections, FP16 readout/Wo, and FP32 statistics,
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

All three models use patch size 16, 12 blocks, independent 3×3 depthwise Q/K
filters and a learned 2D absolute patch position table,
Pre-LayerNorm blocks, packed SwiGLU, and token-LayerNorm followed by mean pooling.
The SwiGLU branch width is `8/3` of the embedding width. There is no CLS token
or LayerScale. Position embeddings reuse timm initialization and bicubic grid
interpolation; there is no per-layer residual CPE.

`output_gate_rank=32` controls the output gate bottleneck. `qk_conv_kernel_size=3`
controls the centered odd-width neighborhood (width k for 1D, k×k for 2D).
Vision adapters fix the convolution dimension to 2; sequence/BERT adapters use 1.

| Model | Width | Heads | Rank | SwiGLU branch width | Parameters¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MemSolve-ViT-T | 192 | 6 | 16 | 512 | 5,464,744 |
| MemSolve-ViT-S | 384 | 6 | 32 | 1024 | 20,697,448 |
| MemSolve-ViT-B | 768 | 12 | 48 | 2048 | 84,090,856 |

¹ At 224px, including the learned position table and 1,000-class ImageNet head.
Training recipes specify their own
DropPath, optimizer, augmentation and schedule.

[Default ImageNet training](docs/IMAGENET_DEIT3.md) follows RoPE-ViT:
T/S train at 224px for 400 epochs; Base trains at 192px for 400 epochs and
fine-tunes at 224px for 20 epochs. Pretraining uses fused LAMB, 3-Augment,
DeiT BCE and global batch 2048; Base fine-tuning uses fused AdamW and batch 512.
Pretraining uses a local 10-epoch linear warmup counted in optimizer updates;
Base fine-tuning retains 5 warmup epochs.
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
MLM head, with MemSolve mixers and SwiGLU FFNs. It uses centered 1D Q/K
convolution and learned absolute positions. The default Base geometry is
width 768, 12 layers, 12 heads and rank 32, with 103,214,394 MLM parameters.

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

Current model checkpoint contract: **24**; CUDA algorithm contract: **19**;
source version: **0.14.0**. The current API is `MemSolve` / `MemSolveConfig`.
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

Code is licensed under [Apache-2.0](LICENSE). The grouped RMSNorm implementation
adapts FLA code under MIT; its attribution and license are in [NOTICE](NOTICE).
The CUDA guide records the upstream sources and reuse boundaries.
