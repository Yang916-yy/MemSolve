# ImageNet with plain ViT³ training

`experiments/train_imagenet.py` uses `experiments/configs/imagenet_vit3.toml`.
The recipe follows the plain **ViT³-T/S/B** models, not H-ViT³ or the MESA variants:
[paper](https://arxiv.org/abs/2512.01643) and
[official source, e3477587d099e6b9e83e9e7c80b1b999e0989a20](https://github.com/LeapLabTHU/ViTTT/tree/e3477587d099e6b9e83e9e7c80b1b999e0989a20/vittt).
The source files are `cfgs/vittt_{t,s,b}.yaml`, `config.py`, `main.py`,
`lr_scheduler.py`, and `data/build.py`.

## Recipe

| Setting | T / S / B |
| --- | --- |
| Epochs / input / patch | 300 / 224 × 224 / 16 |
| Global effective batch / learning rate | 1024 / 0.001 |
| Optimizer | PyTorch fused AdamW, β=(0.9, 0.999), ε=1e-8, weight decay 0.05 |
| Scheduler | Cosine per optimizer update, 20 warmup epochs, no warmup prefix |
| Warmup / minimum LR | 1e-6 / 1e-5 |
| Gradient clipping | Global norm 5, after gradient accumulation |
| Augmentation | RandAugment `rand-m9-mstd0.5-inc1`, color jitter 0.4 |
| Random erasing | Probability 0.25, pixel mode, count 1 |
| Mixup / CutMix | 0.8 / 1.0; probability 1, switch probability 0.5, batch mode |
| Labels / loss | Smoothing 0.1; soft-target cross entropy with Mixup |
| Validation | Bicubic resize to 256, center crop 224, ImageNet mean/std |
| AMP | BF16 autocast, FP32 parameters/solver, no gradient scaling |
| EMA / MESA / repeated augmentation | Disabled |

The paper specifies batch 4096. This run selects **1024** for two GPUs, matching
the global batch of the upstream README launch. Upstream scales its base LR
5e-4, warmup LR 5e-7 and minimum LR 5e-6 from batch 512, giving respectively
0.001, 1e-6 and 1e-5. Thus this is not the paper's batch-4096 setting.

| Tier | Width | Depth | Heads | LSSO rank | Maximum DropPath |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 192 | 12 | 6 | 16 | 0.0 |
| small | 384 | 12 | 6 | 32 | 0.1 |
| base | 768 | 12 | 12 | 48 | 0.4 |

The small model uses MLP ratio 4.125 (1584 hidden channels, aligned to 16)
to target 20M parameters: 20,033,392 including the 1000-class head.
Tiny and base retain ratio 4.0. This width adjustment is an LSSO architecture
choice, not a change to the upstream training recipe. Earlier S/r32 speed
measurements used ratio 4.0 and do not measure this adjusted model.

DropPath increases linearly from zero to the listed maximum across blocks,
as in plain ViT³. LSSO ranks are our model choices. The LSSO encoder uses residual 3×3 depthwise CPE before each block, replacing
the learned absolute position table. CPE uses PyTorch/cuDNN, excludes CLS, and
keeps masked patch features out of neighboring valid updates. It retains CLS
pooling, LayerScale 1e-4 and
LayerNorm ε=1e-6; the mixer has no RankRotary. This is a **ViT³-derived training
protocol for LSSO**, not a reproduction of the ViT³ architecture.

## Batching and data

`batch_size` is physical per GPU. Accumulation is derived so that
`world_size × batch_size × grad_accum = 1024`; LR is not rescaled again.
On two GPUs, batch 512 per GPU gives accumulation 1. There are 1,251 optimizer
updates per epoch (1,281,024 scheduled images), dropping the incomplete update.
`--grad-accum` optionally asserts the derived value.

Mixup/CutMix uses independent virtual groups of 128, the upstream code's
per-GPU batch default. Physical batches must be multiples of 128. These virtual
groups and gradient accumulation are local adaptations, not a claim about the
paper's physical GPU arrangement. Changing physical batches preserves these
group sizes and the effective update, but does not promise identical RNG draws.

The existing `timm/imagenet-1k-wds` data contract is retained: `_info.json`,
1,024 training tar shards and 64 validation tar shards, with a pinned manifest
SHA-256 and structural preflight. No ImageFolder extraction is needed.
Training uses one transformed view per source record, shuffled shards and a
bounded 8,192-record shuffle buffer (initial fill 2,048). This streaming order
is not the upstream ImageFolder/DistributedSampler permutation. Every rank
currently reads the full 50,000-image validation set; reductions preserve the
metric, at the cost of redundant validation work. Training and validation
worker counts are independent (defaults 10 and 4).

## Training execution

The default `execution = "graph"` captures fixed-shape model forward and
backward, including DDP gradient synchronization. Optimizer updates, gradient
clipping, augmentation and scheduling remain outside the CUDA Graph.
Capture warmup does not update parameters and restores RNG state and model
buffers before the first training replay. The graph is reused across epochs
and released before distributed shutdown.

Use `--execution compile-graph` to additionally compile the surrounding vision
blocks with TorchInductor. The native LSSO boundary remains outside compilation,
and one outer CUDA Graph captures the compiled segments and LSSO calls.
Inductor's own CUDA Graphs are disabled for this mode.

Both Graph modes require CUDA, fixed input shapes and `grad_accum = 1`.
Use `--execution eager` when a smaller physical batch requires gradient
accumulation. The default two-GPU configuration uses batch 512 per GPU and
therefore satisfies the Graph requirement.

## Launch and resume

Use the current source checkout with its matching compiled native runtime;
install vision dependencies with `python -m pip install -e '.[vision]'`.
Apex is not required. The notebook
[`imagenet_launcher.ipynb`](../notebooks/imagenet_launcher.ipynb) validates an
existing environment and launches the same entrypoint.

```bash
torchrun --standalone --nproc_per_node=2 experiments/train_imagenet.py \
  --tier small --data-root /datasets/imagenet-1k-wds \
  --output runs/imagenet/vit3_small --batch-size 512 --grad-accum 1
```

Select `--tier tiny` or `--tier base` for the other sizes. All tiers train at
224px in one stage; there is no 192px pretraining or resolution fine-tuning.
`--epochs 30` is a diagnostic override and is recorded as `explicitly-modified`.
No MESA teacher or EMA model is allocated, updated, evaluated or checkpointed.

Resume with the same command and `--resume PATH/checkpoint_last.pt`. Outputs
include `metadata.json`, `metrics.jsonl`, `checkpoint_last.pt`, and
`checkpoint_best.pt`, selected by ordinary-model validation top-1 accuracy.
Checkpoint writes are atomic. Resume validates model, recipe, data, batching,
world size and RNG contracts; it restores optimizer LR and scheduler state
without an extra epoch-level scheduler step. Scheduler updates follow the
upstream zero-based optimizer-update index, including the first warmup update.
Workers are recreated each epoch to support epoch-boundary RNG replay.

The ImageNet checkpoint envelope is now **7**, LSSO model contract **13**, and
native ABI **11**. Old ImageNet training states cannot resume under this recipe.
Historical classification results remain labeled with their original DeiT III
training protocol and source version; this migration produces no new accuracy
results and does not relabel those measurements.

The BF16 autocast setting is a local precision adaptation from the upstream FP16 AMP entrypoint. LSSO keeps its BF16/FP32 internal mixed-precision contract; model parameters and AdamW states remain FP32. No GradScaler is created or checkpointed.
