# MemSolve: From Sequential Memory Updates to Global Equilibrium

**Full-context associative memory through compact equilibrium solves.**

MemSolve jointly designs error correction and token readout. An input-dependent
soft frame gathers associations from all tokens into a compact memory. A learned
core determines its equilibrium response, which is read back through the same
frame. At fixed rank and width, computation is linear in sequence length.

[Paper](paper/main.pdf) · [Operator equations](docs/CORE_CONTRACT.md) ·
[CUDA implementation](docs/CUDA_CONTRACT.md) · [Model architecture](docs/ARCHITECTURE.md)

## Results

| Model | Parameters | ImageNet-1K Top-1 (EMA) |
| --- | ---: | ---: |
| MemSolve-ViT-T | 5M | 74.7% |
| MemSolve-ViT-S | 20M | 81.0% |
| MemSolve-ViT-B | 78M | 82.7% |

The visual backbone has 12 constant-width blocks, residual CPE, GELU FFNs,
and token-wise LayerNorm followed by mean pooling. All three models train
at 224px for 300 epochs with the [RA/EMA recipe](docs/IMAGENET_VIT3.md).
The parameter-matched, absolute-position MemSolve-BERT reaches **81.9** average
GLUE development score. The paper gives per-task metrics and protocols.

On one A800-80GB at 8,192 tokens, complete-mixer forward/backward execution is
**7.5× faster than FlashAttention** and **7.9× faster than BiMamba** under
BF16 autocast and CUDA Graphs. See the paper's efficiency tables for shapes,
latency, and peak memory.

## Installation

Use Python 3.10–3.12 and CUDA PyTorch 2.14 with its matching Triton runtime.
The tested native stack is Torch `2.14.0+cu132`, CUDA 13.2, MathDx 26.06.1,
and CMake >=3.25 on SM80/A800.

```bash
git clone https://github.com/Yang916-yy/MemSolve.git
cd MemSolve
python -m pip install -e '.[vision]'
PYTHON=python MEMSOLVE_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh
```

Use `.[nlp]` for language workflows or `.[sequence]` for genomic/LRA experiments.
Dense vision requires `.[vision,openmmlab]` and compiled MMCV; see the
[dense prediction guide](docs/DOWNSTREAM_PROTOCOLS.md).
Set `CUDA_HOME` and `MATHDX_ROOT` for nondefault toolkit locations.
The [CUDA guide](docs/CUDA_CONTRACT.md#build-and-runtime-artifacts) describes
architecture targets and runtime packaging. The reference backend needs no
native extension.

## Operator

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

`equilibrium` learns a shared accretive core. `complement` omits that core
while retaining input-dependent context mixing. Optional `rotary=True` enables
RoVE content transport. Both modes share the CUDA statistics, readout, and
backward implementation without storing the token-sized frame.

CUDA supports ranks **16 / 32 / 48 / 64**, FP16/BF16 inputs, FP32 parameters,
and boolean validity masks. Compact statistics and solves retain FP32 precision;
outputs follow the input dtype. Exact sequence sharding is provided by
`integrations.distributed.SequenceParallel`.

## Vision

Import `integrations.timm` to register `memsolve_vit_tiny`,
`memsolve_vit_small`, and `memsolve_vit_base`:

```python
import timm
import integrations.timm

model = timm.create_model(
    "memsolve_vit_small", pretrained=False, img_size=224,
    implementation="cuda",
).cuda()
```

Keep parameters in FP32 and use BF16 autocast. Registered models default to the
reference backend; `pretrained=True` is not provided. Load local checkpoints
through the experiment entrypoints.

The default ImageNet entrypoint uses the paper's 300-epoch recipe: fused AdamW,
cross-entropy, repeated augmentation, EMA, and effective batch 2048.
Two GPUs with physical batch 512 each accumulate two microbatches per update.

```bash
torchrun --standalone --nproc_per_node=2 experiments/train_imagenet.py \
  --tier small --data-root /datasets/imagenet-1k-wds \
  --output /runs/memsolve-small
```

[ImageNet guide](docs/IMAGENET_VIT3.md) covers data, batching, CUDA Graphs,
compilation, resume, and ImageNet-100 ablations.
[Mask R-CNN and UPerNet](docs/DOWNSTREAM_PROTOCOLS.md) attach ViT-Adapter to
the pretrained visual blocks.

## Language

MemSolve-BERT uses BERT embeddings and Post-LayerNorm residuals, SwiGLU FFNs,
and a tied MLM head. The Base configuration has width 768, 14 layers, 12 heads,
rank 32, and a centered residual depthwise convolution of width three.

```python
from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM

model = MemSolveBertForMaskedLM(MemSolveBertConfig(
    memsolve_implementation="cuda",
)).cuda()
```

The supplied configuration uses RoVE. The paper's GLUE encoder uses
`position_embedding_type="absolute"`; this is an explicit model choice,
not an interchangeable checkpoint setting. Both use the C4/128-token,
70,000-step pretraining budget. See the [NLP guide](docs/BERT_MLM.md) for
preparation, MLM, GLUE, retrieval, MTEB/LongEmbed, and position settings.
[Re-DocRED](docs/RE_DOCRED.md) provides document-level relation fine-tuning.

## Benchmarks and tests

```bash
python -m benchmarks.benchmark_ball --operator memsolve --mode train \
  --batch 4 --length 8192 --dim 768 --heads 12 --rank 32 \
  --dtype bfloat16 --graph --warmup 20 --steps 50 --repeats 7
```

Select `--operator flashattention` for timm attention with forced Flash SDPA,
or `--operator bimamba` for the official
[Vim BiMamba-v2 implementation](https://github.com/hustvl/Vim/tree/dd0358ad1e42701f22afbefa0717cc8825cf9f45/mamba-1p1p1).
BiMamba requires its upstream package and compiled kernels. Use
`--bimamba-module` for an alternate installation module, `--mode forward`
for inference, and `--profile` for kernel inspection.

The [test guide](tests/README.md) lists focused correctness checks.
[Sequence experiments](docs/SEQUENCE_EXPERIMENTS.md) retain the reusable
GenomicBenchmarks and LRA loaders and matched-encoder runner.

## Repository

| Directory | Contents |
| --- | --- |
| `memsolve/`, `csrc/` | Operator, reference math, and CUDA implementation |
| `integrations/` | timm, BERT, dense vision, and sequence-parallel adapters |
| `experiments/` | Reusable training/data entrypoints and configurations |
| `benchmarks/`, `tests/` | Mixer benchmarks and correctness tests |
| `docs/` | Current implementation and experiment guides |
| `paper/` | LaTeX, bibliography, required PDF figures, and compiled paper |

Datasets, checkpoints, logs, local launch queues, plotting scripts, and revision
records belong outside the checkout. Earlier architectures and retired experiments
remain in Git history. Model checkpoint contract: **26**; native ABI: **17**.

Build the paper with `latexmk -pdf -cd paper/main.tex`. On Debian/Ubuntu,
install `texlive-science`, `texlive-latex-extra`, and `texlive-fonts-recommended`.

Code is licensed under [Apache-2.0](LICENSE). Attribution and redistributed
licenses are in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
and `licenses/`.
