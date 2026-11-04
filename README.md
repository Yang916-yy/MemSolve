# MemSolve: From Sequential Memory Updates to Global Equilibrium

**Full-context associative token mixing via compact equilibrium solves in $\mathcal{O}(N)$ time.**

[**Paper (PDF)**](paper/main.pdf) · [**Operator Math**](docs/CORE_CONTRACT.md) · [**CUDA Fast Path**](docs/CUDA_CONTRACT.md) · [**Model Architecture**](docs/ARCHITECTURE.md) · [**Experiment Logs**](results/README.md)

---

## Overview

Sequential test-time memory layers update their state one token at a time, making parallel training reliant on prefix scans and forcing bidirectional models to run separate forward and backward passes. **MemSolve** reformulates associative memory over a full token set as a simultaneous equilibrium problem.

For each head, an input-dependent soft frame $P = A(I + A^\top A)^{-1/2}$ gathers token relations and content into a compact rank-$r$ sufficient statistic. A learned accretive core $J = I + L L^\top + \Omega \succeq I$ determines the global equilibrium response via a bounded Cayley map $F = (I + J)^{-1}(I - J)$, and the updated memory is read back to all tokens through the same frame:

$$
Y = \eta C + P (F - \eta I) P^\top C.
$$

Because $I + A^\top A$ and $I + J$ are strictly positive/accretive ($r \times r$ with $r \ll N$), the entire layer reduces to compact $r \times r$ Cholesky and LU solves—never materializing an $N \times N$ attention matrix or the $N \times r$ orthonormal frame $P$ in GPU memory.

## Highlights

- **Vision Backbone & Dense Prediction**: With a plain non-hierarchical 12-layer ViT backbone, **MemSolve-ViT** reaches **74.7%** (Tiny, 5M), **81.0%** (Small, 20M), and **82.7%** (Base, 78M) Top-1 accuracy on ImageNet-1K. Paired with ViT-Adapter, it achieves **46.9 box AP / 41.8 mask AP** on COCO (Mask R-CNN 1×) and **48.5 mIoU** on ADE20K (UPerNet 160k).
- **Bidirectional Language Pretraining**: Pretrained on C4 and fine-tuned on GLUE under the LION M2-BERT budget, **MemSolve-BERT** (110M) achieves an average GLUE score of **81.9**.
- **Long Range Arena (LRA)**: Across all 5 LRA tasks under a controlled encoder with Rotary Value Encoding (RoVE), MemSolve reaches **74.9%** mean accuracy—outperforming softmax attention (72.1%) and every tested linear, kernel, and token-routing baseline.
- **Hardware Efficiency**: On an NVIDIA A800-80GB GPU under BF16 autocast and CUDA Graphs, full forward+backward execution at 8,192 tokens is **7.8× faster than FlashAttention-2** and **8.1× faster than BiMamba** (and **28.9×** / **9.2×** faster at 32,768 tokens).

## Installation

Requires Python 3.10–3.12 and PyTorch 2.14 with CUDA 13.2, NVIDIA MathDx 26.06.1, and CMake $\ge 3.25$ to build the native cuSOLVERDx/cuBLASLt extension (the pure-PyTorch reference implementation works out of the box without compiling):

```bash
git clone https://github.com/Yang916-yy/MemSolve.git
cd MemSolve
python -m pip install -e '.[vision]'
PYTHON=python MEMSOLVE_CUDA_ARCHITECTURES=80 bash tools/build_cuda.sh
```

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

To train on ImageNet-1K (see [docs/IMAGENET_VIT3.md](docs/IMAGENET_VIT3.md)):

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

Run the test suite (see [tests/README.md](tests/README.md)):

```bash
python -m pytest -q
```

## Repository Structure

| Path | Description |
| --- | --- |
| [`memsolve/`](memsolve/), [`csrc/`](csrc/) | Core `MemSolve` module, exact PyTorch reference, and Triton/cuSOLVERDx/cuBLASLt CUDA fast path |
| [`integrations/`](integrations/) | `timm` vision backbones, Hugging Face `MemSolveBert`, OpenMMLab ViT-Adapter, and `SequenceParallel` |
| [`experiments/`](experiments/) | Training & evaluation entrypoints for ImageNet, COCO/ADE20K, C4 MLM, GLUE, and LRA |
| [`benchmarks/`](benchmarks/), [`tests/`](tests/) | Operator latency/memory benchmark harness and unit/numerical test suite |
| [`docs/`](docs/) | Detailed guides for operator math, CUDA kernels, and each experimental benchmark |
| [`results/`](results/README.md) | Complete training logs, evaluation metrics, and resolved configs for all paper tables |
| [`paper/`](paper/) | LaTeX source (`main.tex`), figures, bibliography, and compiled manuscript (`main.pdf`) |

## License

Released under the [Apache-2.0 License](LICENSE). Third-party attributions and license notices are documented in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [`licenses/`](licenses/).
