# ImageNet training

`experiments/train_imagenet.py` trains MemSolve-ViT-T/S/B. Its default
configuration, `experiments/configs/imagenet300.toml`, contains the 300-epoch
RA/EMA recipe used by the paper's ImageNet-1K models.

The architecture and optimizer schedule derive from plain
[ViT³](https://github.com/LeapLabTHU/ViTTT/tree/e3477587d099e6b9e83e9e7c80b1b999e0989a20/vittt).
MemSolve supplies the mixer and residual CPE. See [architecture](ARCHITECTURE.md)
for the block definition and timm registration.

## Model and recipe

| Setting | Tiny | Small | Base |
| --- | ---: | ---: | ---: |
| Width / depth | 192 / 12 | 384 / 12 | 768 / 12 |
| Heads / rank | 6 / 16 | 6 / 32 | 12 / 48 |
| GELU FFN width | 768 | 1584 | 3072 |
| Parameters, 1000 classes | 5,053,936 | 19,876,336 | 77,980,792 |
| Maximum DropPath, linear by depth | 0.0 | 0.1 | 0.4 |

| Training setting | Value |
| --- | --- |
| Epochs / input / patch | 300 / 224 × 224 / 16 |
| Effective batch / peak LR | 2048 / 0.002 |
| Optimizer | Fused AdamW, betas (0.9, 0.999), epsilon 1e-8 |
| Weight decay / gradient clipping | 0.05 / global norm 5 |
| Schedule | Per-update cosine; 20 warmup epochs within the 300-epoch budget |
| Warmup / minimum LR | 2e-6 / 2e-5 |
| Augmentation | RandAugment `rand-m9-mstd0.5-inc1`, three-repeat RA |
| Mixup / CutMix | 0.8 / 1.0; batch mode, probability 1, switch probability 0.5 |
| Random erasing | Probability 0.25, pixel mode |
| Loss / label smoothing | Soft-target CE / 0.1 |
| EMA | Fixed decay 0.9999, updated after each optimizer step |
| Validation | Bicubic resize to 256, center crop 224, ImageNet mean/std |
| Execution | BF16 autocast, FP32 parameters, compile + CUDA Graph |

The configuration retains `color_jitter=0.4`; timm's selected RandAugment policy
handles color augmentation without a separate jitter transform. There is no
MESA teacher. Ordinary and EMA validation are recorded separately, with separate
best checkpoints. The paper reports best EMA Top-1.

## Data and batching

Use the `timm/imagenet-1k-wds` layout: `_info.json`, 1024 training tar shards,
and 64 validation tar shards. The loader checks the pinned manifest and shard
structure. No ImageFolder extraction is needed. Data stays outside the checkout.

`world_size × batch_size × grad_accum = effective_batch`.
The default two-GPU launch uses 512 images per GPU and accumulation 2;
625 complete updates form an epoch. `--batch-size` changes the physical batch,
and accumulation is derived. `--grad-accum` can assert the expected value.
Learning rate comes from the recipe and is not scaled again.

RA uses timm's `RepeatAugSampler` and WIDS indexed tar access. Repetition occurs
before rank distribution, while the epoch budget stays fixed. Mixup/CutMix uses
the complete physical batch; this recipe has no virtual augmentation groups.
Changing physical batch changes mixing pairs even at the same effective batch.

Train/validation workers default to 12/4 per rank and remain alive across epochs.
Workers use `spawn`, prefetch one batch, and reseed from seed/epoch/rank/worker.
Each rank evaluates all 50,000 validation images; reductions preserve the metric.

## Launch, execution, and resume

```bash
torchrun --standalone --nproc_per_node=2 experiments/train_imagenet.py \
  --config experiments/configs/imagenet300.toml --tier small \
  --data-root /datasets/imagenet-1k-wds --output /runs/memsolve-small \
  --batch-size 512 --grad-accum 2
```

Use `--tier tiny` or `--tier base` for the other sizes. All three train at 224px
in one stage. Install `.[vision]` and build the native extension first.

`--execution compile-graph` compiles surrounding vision blocks while leaving
the native mixer outside Dynamo, then captures a complete accumulated
forward/backward update. DDP synchronizes on the final microbatch. Optimizer,
clipping, augmentation, and scheduling remain outside the graph. Warmup restores
RNG state and buffers without updating weights. `graph` omits compilation;
`eager` omits capture. Graph modes require fixed microbatch shapes.

Resume with the same recipe and `--resume /runs/memsolve-small/checkpoint_last.pt`.
Resume restores model, EMA, optimizer, scheduler, and RNG state and validates
the saved model/data/batching contract. It requires the original run's recipe;
specify `--config` when resuming an alternative configuration.

Outputs include `metadata.json`, `metrics.jsonl`, `checkpoint_last.pt`,
`checkpoint_best.pt`, and `checkpoint_best_ema.pt`. Checkpoints store both
ordinary and EMA weights and identify the selected set. `--eval --resume PATH`
evaluates a checkpoint without training. Run artifacts stay outside Git.

## ImageNet-100 ablations

The paper compares Equilibrium, Complement, no skew coupling, and rank 16
using Small for 100 epochs. The fixed subset has 126,689 training and 5,000
validation images. The recipe uses effective batch 512, LR 5e-4, five warmup
epochs, the same augmentation policy, and no RA or EMA. The paper records the
selection rule and results. Use the same class list and initialization across comparisons. The subset
manifest and run-specific preparation are stored with the experiment artifacts. Trial queues,
source snapshots, and logs are stored with the runs, not in this repository.
