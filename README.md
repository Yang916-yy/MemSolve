# MemSolve: From Sequential Memory Updates to Global Equilibrium

**Full-context associative token mixing via compact equilibrium solves in $\mathcal{O}(N)$ time.**

[**Paper (PDF)**](paper/main.pdf) · [**Operator Math**](docs/CORE_CONTRACT.md) · [**CUDA Fast Path**](docs/CUDA_CONTRACT.md) · [**Model Architecture**](docs/ARCHITECTURE.md) · [**Experiment Logs**](results/README.md)

---

## Overview

Sequential test-time memory layers update their state one token at a time, making parallel training reliant on prefix scans and forcing bidirectional models to run separate forward and backward passes. **MemSolve** reformulates associative memory over a full token set as a simultaneous equilibrium problem.

For each head, an input-dependent soft frame $P = A R^{-1}$ (with upper Cholesky factor $R^\top R = I_r + A^\top A$) gathers token relations and content into compact rank-$r$ sufficient statistics. A learned accretive core $J = L L^\top + \Omega$ ($LL^\top \succ 0, \Omega^\top = -\Omega$) determines the global equilibrium response via a contractive Cayley map $F = (I_r + J)^{-1}(I_r - J)$, and the updated memory is read back to all tokens through the same frame:

$$
Y = \eta C + P (F - \eta I_r) P^\top C.
$$

Because $I_r + A^\top A$ and $I_r + J$ are strictly positive/accretive ($r \times r$ with $r \ll N$), the entire layer reduces to compact $r \times r$ Cholesky and LU solves—never materializing an $N \times N$ attention matrix or the $N \times r$ soft frame $P$ in GPU memory.

## Highlights

- **Vision Backbone & Dense Prediction**: With a plain non-hierarchical 12-layer ViT backbone, **MemSolve-ViT** reaches **74.7%** (Tiny, 5M), **81.0%** (Small, 20M), and **82.7%** (Base, 78M) Top-1 accuracy on ImageNet-1K. Paired with ViT-Adapter, it achieves **44.7 / 40.6** (Small) and **46.6 / 41.8** (Base) box / mask AP on COCO (Mask R-CNN 1×), and **47.2** (Small) / **48.3** (Base) mIoU on ADE20K (UPerNet 160k).
- **Bidirectional Language Pretraining**: Pretrained on C4 and fine-tuned on GLUE under the LION M2-BERT budget, **MemSolve-BERT** (112M) achieves an average GLUE score of **81.9** (**81.5** with parameter-free RoVE, 111M).
- **Long Range Arena (LRA)**: Across all 5 LRA tasks under a controlled encoder with Rotary Value Encoding (RoVE), MemSolve reaches **67.35%** mean accuracy—outperforming softmax attention (66.71%) and every tested linear, kernel, and token-routing baseline.
- **Hardware Efficiency**: On an NVIDIA A800-80GB GPU under BF16 autocast and CUDA Graphs, full forward+backward execution (`recompute_readout=False`) is **7.9×** and **8.3×** faster than FlashAttention-2 and fused BiMamba-v2 at 8,192 tokens, and **29.0×** and **9.2×** faster at 32,768 tokens—outperforming FlashAttention-2 across all tested lengths from 128 to 32,768 tokens while using **11%** and **57%** less peak memory at 32,768 tokens (optional `recompute_readout="auto"` reduces 32,768-token peak memory by **17%** and **60%** at **28.2×** speedup).

## Installation

Requires Python 3.10–3.12 and PyTorch 2.14. Building the native CUDA C++/cuBLASLt extension requires a matching CUDA toolkit ($\ge 12.0$) and CMake $\ge 3.25$, with no external SDK required by default (`MEMSOLVE_USE_MATHDX=ON` optionally builds with NVIDIA MathDx 26.06.1 on CUDA 13.2; the pure-PyTorch reference implementation works out of the box without compiling):

```bash
git clone https://github.com/Yang916-yy/MemSolve.git
cd MemSolve
python -m pip install -e '.[vision]'
PYTHON=python bash tools/build_cuda.sh
```

By default, `bash tools/build_cuda.sh` compiles a self-contained multi-arch `80;90a` fatbinary (Ampere `SM80` + Hopper `SM90a` + embedded PTX) on both `x86_64` and `aarch64` (`MEMSOLVE_CUDA_ARCHITECTURES` overrides the target SMs).

Optional extras:
- `pip install -e '.[nlp]'` for C4 MLM pretraining and GLUE fine-tuning.
- `pip install -e '.[sequence]'` for LRA and sequence benchmarks.
- `pip install -e '.[vision,openmmlab]'` plus `bash tools/build_mmcv.sh` for COCO detection and ADE20K segmentation (see [docs/DOWNSTREAM_PROTOCOLS.md](docs/DOWNSTREAM_PROTOCOLS.md)).

## Quick Start

### Standalone MemSolve Layer

```python
import torch
from memsolve import MemSolve, MemSolveConfig
from memsolve.ball import cuda

layer = MemSolve(MemSolveConfig(dim=384, num_heads=6, rank=32)).cuda()
x = torch.randn(4, 196, 384, device="cuda", dtype=torch.bfloat16)

cuda.load(device=x.device)
y = layer(x, implementation="cuda")
y_ref = layer(x, implementation="reference")
```

- `core_mode="equilibrium"` (default) enables the learned accretive core $J$; `core_mode="complement"` sets $F = 0$ while keeping input-dependent context mixing.
- `rotary=True` enables Rotary Value Encoding (RoVE), which rotates content streams before aggregation and applies the inverse rotation at readout so relative positions are preserved without altering the Gram matrix $A^\top A$.
- Exact multi-GPU sequence parallelism with $\mathcal{O}(B H (r^2 + r d))$ communication volume is provided by `integrations.distributed.SequenceParallel`.

### Vision (`timm`)

```python
import timm
import integrations.timm

model = timm.create_model(
    "memsolve_vit_small",
    pretrained=False,
    img_size=224,
    implementation="cuda",
).cuda()
```

Registered `timm` entrypoints cover `memsolve_vit_tiny`, `memsolve_vit_small`, `memsolve_vit_base`, and `memsolve_vit_large` (also available via `integrations.timm.create_memsolve_vit_variant`). To train on ImageNet-1K (see [docs/IMAGENET_VIT3.md](docs/IMAGENET_VIT3.md)):

```bash
torchrun --standalone --nproc_per_node=2 experiments/train_imagenet.py \
  --tier small --data-root /datasets/imagenet-1k-wds \
  --output /runs/memsolve-small
```

### Language (Hugging Face `transformers`)

```python
from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM

model = MemSolveBertForMaskedLM(MemSolveBertConfig(
    position_embedding_type="absolute",
    memsolve_implementation="cuda",
)).cuda()
```

See [docs/BERT_MLM.md](docs/BERT_MLM.md) for C4 data preparation, 70k-step MLM pretraining, and GLUE fine-tuning commands.

### Benchmarks & Tests

Benchmark end-to-end forward/backward latency and peak memory against FlashAttention-2 or BiMamba:

```bash
python -m benchmarks.benchmark_ball --operator memsolve --mode train \
  --batch 4 --length 8192 --dim 768 --heads 12 --rank 32 \
  --dtype bfloat16 --graph --warmup 20 --steps 50 --repeats 7
```

`benchmark_ball` defaults to `--recompute-readout false` to match uncheckpointed FlashAttention-2 and BiMamba-v2; pass `--recompute-readout auto` (the `MemSolveConfig` default for $N > 512$) or `true` to benchmark readout recomputation.

Run the test suite (see [tests/README.md](tests/README.md)):

```bash
python -m pytest -q
```

## Repository Structure

| Path | Description |
| --- | --- |
| [`memsolve/`](memsolve/), [`csrc/`](csrc/) | Core `MemSolve` module, exact PyTorch reference, and Triton + native CUDA C++ (SM80/SM90a, cuBLASLt, optional MathDx) fast path |
| [`integrations/`](integrations/) | `timm` vision backbones, Hugging Face `MemSolveBert`, OpenMMLab ViT-Adapter, and `SequenceParallel` |
| [`experiments/`](experiments/) | Training & evaluation entrypoints for ImageNet, COCO/ADE20K, C4 MLM, GLUE, and LRA |
| [`benchmarks/`](benchmarks/), [`tests/`](tests/) | Operator latency/memory benchmark harness and unit/numerical test suite |
| [`docs/`](docs/) | Detailed guides for operator math, CUDA kernels, and each experimental benchmark |
| [`results/`](results/README.md) | Complete training logs, evaluation metrics, and resolved configs for all paper tables |
| [`paper/`](paper/) | LaTeX source (`main.tex`), figures, bibliography, and compiled manuscript (`main.pdf`) |

## License

Released under the [Apache-2.0 License](LICENSE). Third-party attributions and license notices are documented in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [`licenses/`](licenses/).
