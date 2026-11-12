# Long Range Arena (LRA) & Sequence Experiments

This guide covers the Long Range Arena (LRA) benchmarks in the paper, including our controlled 10-mixer comparison under matched RoVE position encodings (`experiments/train_transformers.py`), the one-command reproduction script (`experiments/reproduce_lra.py`), and the standalone Nyströmformer-protocol runner (`experiments/train_lra.py`).

## Reproducing the Paper's Controlled LRA Benchmark

The paper evaluates **MemSolve** alongside 9 attention and efficient-attention mixers (**MHA**, **Local**, **Linear**, **BigBird**, **Performer**, **Nyströmformer**, **Agent Attention**, **MiTA**, and **PolaFormer**) inside an identical 6-layer encoder with **Rotary Value Encoding (RoVE)**.

All 50 published run logs, resolved configurations, and test summaries are in [`results/lra/`](../results/lra/). You can reproduce any recorded `<model>/<task>` pair with a single command:

```bash
python -m experiments.reproduce_lra memsolve/image \
  --data-root /datasets/lra --cache-root /tmp/lra-cache \
  --output /runs/lra/memsolve-image
```

Pass `--prepare-only` to inspect the resolved hyperparameters and data splits without starting training.

### Shared Encoder & RoVE Setup (`experiments/configs/lra_rove.toml`)

All 10 mixers share the exact same 6-layer pre-LayerNorm backbone, FFN width (ratio 2), pooling, token masks, and optimizer schedule:
- **Position Encoding**: Symbolic tasks (`listops`, `text`, `retrieval`) use 1D RoVE (`--position-encoding rove_1d`); pixel tasks (`image`, `pathfinder`) use 2D axial RoVE (`--position-encoding rove_2d`), rotating horizontal and vertical channel halves with base 10,000. For attention baselines, RoVE rotates values $V$ before attention and applies the inverse rotation at readout before the output projection, so every mixer receives the exact same content-position transport.
- **Task Schedule**:

| Task | Sequence Length | Width / Heads | Epochs | Effective Batch | Peak LR | RoVE |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| ListOps | 2048 | 128 / 4 | 40 | 64 | 1e-3 | 1D |
| Text (IMDb bytes) | 4096 | 128 / 4 | 32 | 32 | 5e-4 | 1D |
| Retrieval (AAN bytes) | 4096 per doc | 128 / 4 | 20 | 64 pairs | 1e-3 | 1D |
| Image (CIFAR-10 gray) | 1024 | 128 / 4 | 200 | 128 | 5e-4 | 2D |
| Pathfinder-32 (hard) | 1024 | 128 / 4 | 200 | 128 | 2e-4 | 2D |

### Running Custom LRA Jobs

You can also launch any task and mixer directly through `experiments/train_transformers.py`:

```bash
python -m experiments.train_transformers \
  --config experiments/configs/lra_rove.toml --task listops \
  --mixer memsolve --position-encoding rove_1d --execution compile-graph \
  --epochs 40 --batch-size 64 --eval-batch-size 64 --lr 0.001 \
  --data-root /datasets/lra --cache-root /tmp/lra-cache \
  --output /runs/lra/listops-memsolve
```

Supported `--mixer` choices:
- `memsolve`: Equilibrium MemSolve (default rank 16 in `lra_rove.toml`, or set `--rank`).
- `mha`: Exact multi-head attention (supports `--attention-backend flex` for compiled PyTorch FlexAttention).
- `local`, `linear`, `bigbird`: Windowed (256-token), ELU+1 linear, and sparse block BigBird attention.
- `performer`: 128 positive orthogonal random features per head.
- `nystromformer`: 128 masked contiguous landmarks with 6 iterative pinv steps.
- `agent`: 128 masked pooled agent tokens.
- `mita`: 128 routers, top-128 selected keys, 1 router per query (uses variable-length FlashAttention; pair with `--execution compile` rather than `compile-graph`).
- `polaformer`: Polarity-aware linear attention with learned power ($\alpha = 4$).

Passing `--fail-on-nonfinite` halts immediately with exit code `86` and writes `failure.json` if losses, gradients, or weights become `NaN`/`Inf`. `--execution compile` and `--execution compile-graph` compile full encoder blocks end-to-end through Inductor (including registered `MemSolve` custom ops without Dynamo graph breaks).

## Learned Absolute Position Recipe (`experiments/configs/lra.toml`)

For experiments with learned 1D/2D absolute position embeddings (4 layers, width 128, 4 heads, FFN width 512, rank 32):

```bash
python -m experiments.train_transformers \
  --config experiments/configs/lra.toml --task listops \
  --mixer memsolve --seed 0 \
  --data-root /datasets/lra --cache-root /tmp/lra-cache \
  --output /runs/lra/listops-abs-seed0
```

## Standalone Nyströmformer-Protocol Runner (`experiments/train_lra.py`)

`experiments/train_lra.py` implements the smaller 2-layer, 64-width, fixed-step protocol from the [Nyströmformer repository](https://github.com/mlpen/Nystromformer/tree/56893131bf3fa99b5a2d3ab452a591dca722529a/LRA):

```bash
python -m experiments.train_lra \
  --config experiments/configs/lra_nystromformer.toml --task listops \
  --data-root /datasets/lra --cache-root /tmp/lra-cache \
  --output /runs/lra/listops-nystrom-protocol
```
