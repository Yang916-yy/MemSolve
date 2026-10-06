# MemSolve-ViT with LAMB and 3-Augment

## Default: RoPE-ViT-derived 400-epoch training

`experiments/configs/imagenet_ropevit_400.toml` is the default for all T/S/B
tiers. Training duration, global batch, learning rate, decay, DropPath,
loss and augmentations follow the Small and Base commands in the
[official RoPE-ViT README](https://github.com/naver-ai/rope-vit/blob/main/deit/README.md).
Tiny borrows the Small recipe; upstream does not list a Tiny training command.

| Setting | T/S pretraining | Base pretraining | Base resolution fine-tuning |
| --- | --- | --- | --- |
| Resolution / epochs | 224 × 224 / 400 | 192 × 192 / 400 | 224 × 224 / 20 |
| Global effective batch | 2048 | 2048 | 512 |
| Physical batch per GPU, two GPUs | 512 | 512 | 256 |
| Gradient accumulation | 2 | 2 | 1 |
| Optimizer | Apex fused LAMB | Apex fused LAMB | PyTorch fused AdamW |
| Peak LR | 0.004 | 0.003 | 1e-5 |
| Minimum LR | 1e-5 | 1e-5 | 1e-5 |
| Warmup | 5 epochs, start 1e-6 | 5 epochs, start 1e-6 | 5 epochs, start 1e-6 |
| Weight decay | 0.03 | 0.03 | 0.1 |
| Uniform DropPath | 0 | 0.1 | 0.2 |
| Loss / smoothing | DeiT BCE / 0 | DeiT BCE / 0 | Soft-target CE / 0.1 |
| Augmentation | 3-Augment | 3-Augment | RandAugment m9 |
| Repeated augmentation | Enabled | Enabled | Disabled |
| Color jitter argument | 0.3 | 0.3 | 0.3 |
| Mixup / CutMix | 0.8 / 1.0 | 0.8 / 1.0 | 0.8 / 1.0 |
| Virtual augmentation group | 256 | 256 | 64 |
| Gradient clipping | LAMB internal norm 1 | LAMB internal norm 1 | Disabled |
| Random erasing / eval crop ratio | 0 / 1.0 | 0 / 1.0 | 0 / 1.0 |
| EMA decay, no warmup | 0.99996 | 0.99996 | 0.99996 |

All optimizers use β=(0.9, 0.999), ε=1e-8. The LR is the official value for
the effective batch; no additional LR scaling is applied. A pretraining
optimizer update consumes `2 GPUs × 512 images × 2 microbatches = 2048`.
Fine-tuning consumes `2 × 256 = 512`. Base therefore uses 420 epochs in total.
Its phase-specific DropPath is passed to the actual backbone, stored in the
model contract and checked on resume.

EMA follows the upstream [main.py](https://github.com/naver-ai/rope-vit/blob/main/deit/main.py)
and [engine.py](https://github.com/naver-ai/rope-vit/blob/main/deit/engine.py):
initialize a shadow from the starting model, update it after each optimizer step
with constant decay 0.99996, and save it as `model_ema`. Unlike the upstream entrypoint,
our validation evaluates both weight sets on the same decoded batches. Metrics
`val_loss/acc1/acc5` describe ordinary weights; `ema_val_loss/acc1/acc5` describe EMA.
`checkpoint_best.pt` and `checkpoint_best_ema.pt` are selected independently by their
respective top-1 accuracy; `checkpoint_last.pt` follows the save interval.
All files contain ordinary weights, EMA weights, optimizer state and both best scores.
EMA-best files declare `selected_weights = "model_ema"`; the other files select `model`.
Resume restores both weight sets and their best scores, always keeping optimizer
state paired with ordinary weights. Resolution fine-tuning loads the file's selected
weights and initializes a fresh EMA shadow. `--eval --resume ...` reports both sets.
The implementation uses timm ModelEmaV3's foreach parameter updates, without EMA
warmup. It copies buffers instead of averaging them; the MemSolve-ViT backbone
has no running-statistics buffers. Extra-state architecture metadata is preserved.
EMA stays outside the CUDA Graph and updates once per accumulated optimizer step.
The shadow is copied before compile installs bound forward wrappers and synchronized
after DDP initialization, so its evaluation reads its own parameters on every rank.

The MemSolve backbone, BF16 AMP with FP32 parameters, per-update cosine scheduling
and indexed WebDataset, plus dual ordinary/EMA validation, are local adaptations.
Upstream steps its scheduler per
epoch. These differences make this a **RoPE-ViT-derived protocol**.

The virtual groups use the repeat-before-rank-split sampler described below,
with eight independent Mixup/CutMix draws per effective update. The model
receives complete physical batches. Pretraining schedules 1,280,000 transformed
images and 625 optimizer updates per epoch; repeated augmentation does not
triple its length. Fine-tuning schedules 1,281,024 images and 2,502 updates.
Train/validation workers remain persistent (12/4 per GPU); `compile-graph`
retains the outer CUDA Graph and PyTorch fusion. Language recipes are unchanged.

```bash
# T/S: use --tier tiny or --tier small; Base pretraining is shown here.
torchrun --standalone --nproc_per_node=2 experiments/train_imagenet.py \
  --tier base --phase pretrain --data-root /datasets/imagenet-1k-wds \
  --output runs/imagenet/memsolve_b_ropevit400

# A separate launch after the 400-epoch pretraining run.
torchrun --standalone --nproc_per_node=2 experiments/train_imagenet.py \
  --tier base --phase finetune --data-root /datasets/imagenet-1k-wds \
  --finetune runs/imagenet/memsolve_b_ropevit400/checkpoint_best.pt \
  --output runs/imagenet/memsolve_b_ropevit224
```

The notebook selects the same default configuration and exposes `PHASE` and
`FINETUNE_CHECKPOINT` for the Base transition. The selected TOML and adaptations
are recorded in checkpoint metadata as `ropevit-derived`. Fine-tuning loads
compatible weights with a fresh optimizer and schedule. Use `--resume` only to
continue the same phase and contract; an 800-epoch checkpoint cannot be resumed
under the new 400-epoch schedule.

## Optional explicit DeiT III Base configurations

`experiments/configs/imagenet_deit3_400.toml` adapts the ImageNet-1K recipe in
[DeiT III](https://arxiv.org/abs/2204.07118), tables 1, 6 and 13. The backbone
retains MemSolve-ViT-B/r48, packed SwiGLU, Q/K convolution followed by axial 2D RoPE, token-LN mean
pooling, no CLS and no LayerScale. BF16 AMP, no EMA, local indexed WebDataset and
physical-batch gradient accumulation are local adaptations. This is a
**DeiT III-derived training protocol**, not the original ViT architecture or
an identical ImageFolder sampling protocol.

`experiments/configs/imagenet_deit3_800.toml` implements the longer Base recipe
from the [official training and fine-tuning commands](https://github.com/facebookresearch/deit/blob/main/README_revenge.md).
It trains at 192px for 800 epochs with weight decay 0.05 and uniform DropPath
0.2, followed by 20 epochs at 224px with weight decay 0.1 and DropPath 0.2.
It defaults to `compile-graph`: PyTorch Inductor fuses the surrounding vision
operations while the outer CUDA Graph also captures the MemSolve kernels and
accumulated DDP backward. The operator mathematics and precision contract
remain the same. Select `--execution graph` to measure the uncompiled graph.

The table below describes the 400-epoch recipe. The 800-epoch recipe changes
pretraining duration, pretraining decay, DropPath in both phases, and execution;
batching, augmentation, optimizers and learning-rate scales are shared.

| Setting | Pretraining | Resolution fine-tuning |
| --- | --- | --- |
| Resolution / epochs | 192 × 192 / 400 | 224 × 224 / 20 |
| Physical batch per GPU, two GPUs | 512 | 256 |
| Accumulation / global effective batch | 2 / 2048 | 1 / 512 |
| Virtual augmentation group / draws per update | 256 / 8 | 64 / 8 |
| Optimizer | NVIDIA Apex fused LAMB | PyTorch fused AdamW |
| LR / minimum LR | 0.003 / 1e-5 | 1e-5 / 1e-5 |
| Warmup | 5 epochs, start at 1e-6 | 5 epochs, start at 1e-6 |
| Weight decay | 0.02 | 0.1 |
| DropPath | 0.1, uniform across layers | 0.1, uniform across layers |
| Loss / smoothing | DeiT multi-label BCE / 0 | Soft-target CE / 0.1 |
| Augmentation | 3-Augment, color jitter 0.3 | RandAugment m9, color jitter 0 |
| Repeated augmentation | Three views, fixed epoch quota | Disabled |
| Mixup / CutMix | 0.8 / 1.0, batch mode | 0.8 / 1.0, batch mode |
| Random erasing | Disabled | Disabled |
| Gradient clipping | LAMB internal global norm 1 | Disabled |
| Evaluation crop ratio | 1.0 | 1.0 |

The 400-epoch default uses decay 0.02 and DropPath 0.1. The longer 800-epoch
recipe raises regularization; reducing its epoch count alone would not select
the published 400-epoch settings. Pretraining schedules 1,280,000 transformed
images and 625 optimizer updates per epoch. Repeated augmentation changes
source reuse and independently transforms each view; it does not triple the
epoch length. Fine-tuning schedules 1,281,024 images and 2,502 updates per epoch.

The restored virtual-device schedule globally shuffles the source indices,
forms unique 256-source groups, repeats each whole group three times, then
splits the group stream by rank. Each repeated index receives a fresh transform.
The same source's views enter the same or adjacent optimizer updates; worker
count does not change the source schedule. The epoch consumes 426,752 distinct
sources, with the last 256-source group contributing two views at the quota
boundary. Epoch length and update count stay fixed.

This reuses the repository's earlier virtual-group schedule (`22c89c0`). It
follows the repeat-before-rank-split ordering of the
[official RASampler](https://github.com/facebookresearch/deit/blob/main/samplers.py),
but repeats whole groups rather than individual adjacent indices. Consequently
the per-update grouping is an adaptation, not a claim of identical sample order.
Mixup/CutMix draws independently within each virtual group. Two groups are
concatenated into each 512-image pretraining forward; four 64-image groups
form each 256-image fine-tuning forward. The network and CUDA Graph always see
the complete physical batch.

The training reader uses [WIDS](https://github.com/webdataset/wids) 0.1.11's
indexed mmap tar access to honor the global schedule without depending on
worker partitioning. It reads the existing tar files directly, without
extraction, duplicate downloads or decoded-image caches. Workers retain shard
indexes and mmap handles to avoid repeated header scans; only each loader
process raises its own file-descriptor soft limit. Install the `vision` extra
to include WIDS. Validation retains the streaming reader. Persistent workers
reseed Python, NumPy and Torch on their first sample of each shared epoch, so
epoch-boundary restarts replay the same augmentations with the same worker count.

3-Augment follows
[the official augmentation implementation](https://github.com/facebookresearch/deit/blob/main/augment.py):
random resized crop, horizontal flip, a uniformly chosen grayscale/solarize/PIL
Gaussian blur, color jitter, and ImageNet normalization. Mixup/CutMix runs on
the virtual groups before graph replay. BCE uses positive membership
after mixing (`targets > 0`), as in the official DeiT training loop.

## NVIDIA Apex

The pretraining optimizer imports `apex.optimizers.FusedLAMB`; it never silently
substitutes AdamW or an unfused implementation. Apex requires CUDA extensions.
The following revision was built with local PyTorch 2.14.0+cu132 and CUDA 13.2:
`575968bc1f9127ccd61003a681472a83af4ff1a1`.

```bash
git clone https://github.com/NVIDIA/apex.git /path/to/apex
git -C /path/to/apex checkout 575968bc1f9127ccd61003a681472a83af4ff1a1
cd /path/to/apex
CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH \
  TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
  /cywang/vision-venv/bin/python -m pip install --no-build-isolation --no-deps .
```

Set the toolkit path and architecture for the target machine. BF16 refers to
autocast activations; parameters, gradients and optimizer moments remain FP32,
which the fused LAMB implementation supports. LAMB performs its own clipping
after accumulation and gradient synchronization. It applies layerwise trust
ratios and therefore has a different update from AdamW; Delta remains in the
regular decay group, without a model-specific optimizer hook.

## Launch the two phases

```bash
/cywang/vision-venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module experiments.train_imagenet --config experiments/configs/imagenet_deit3_400.toml \
  --tier base --phase pretrain --data-root /tmp/imagenet-1k-wds \
  --output /cywang/runs/imagenet/ridgon_b_deit3_400_seed0

/cywang/vision-venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module experiments.train_imagenet --config experiments/configs/imagenet_deit3_400.toml \
  --tier base --phase finetune --data-root /tmp/imagenet-1k-wds \
  --finetune /cywang/runs/imagenet/ridgon_b_deit3_400_seed0/checkpoint_best.pt \
  --output /cywang/runs/imagenet/ridgon_b_deit3_224_seed0
```

Fine-tuning strictly loads compatible model weights into a fresh 224px model.
RoPE tables are regenerated for the actual patch grid; no learned position table is interpolated. The
optimizer, schedule, epoch counter, best metric and random streams start anew;
the source checkpoint is recorded in `metrics.jsonl`. Use `--resume` for an
interrupted run within the same phase. `--resume` and `--finetune` are mutually
exclusive. Fresh runs require an empty output directory. For resume, use the
original run directory: checkpoint compatibility is checked, but the loader
does not yet verify that a nonempty destination belongs to that checkpoint.

For the 800-epoch protocol, use `imagenet_deit3_800.toml` in both commands and
separate output directories. Initialize pretraining from scratch rather than
resuming a 400-epoch checkpoint: the cosine schedule, decay and DropPath differ.

## Accumulated CUDA Graph

Graph mode captures one complete effective-batch update. Two 512-image
microbatches run sequentially, with loss divided by two before each backward.
DDP `no_sync` suppresses the first reduction; the final backward synchronizes
the accumulated gradient. Both micros use equal shapes. Activations can be
reused within the graph; input buffers and small outputs cover both micros.
The optimizer, clipping, augmentation and update-based scheduler stay outside
the graph and run once per effective batch.

The first DDP prewarm uses synchronized backward before enabling `no_sync`,
following [PyTorch issue 143580](https://github.com/pytorch/pytorch/issues/143580).
Warmup updates no weights, and capture restores RNG state and model buffers.
The graph is reused across epochs and reset before distributed shutdown. Both
`graph` and `compile-graph` support accumulation; the latter also compiles the
surrounding vision blocks while retaining the MemSolve CUDA boundary.

Compiled execution preserves eager BF16 rounding through Inductor's
[`emulate_precision_casts`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)
option. Fused kernels keep the cast boundaries without materializing every
intermediate tensor. AOTAutograd uses `backward_pass_autocast="off"` because
the entrypoint runs backward outside forward autocast, following the
[PyTorch AMP guidance](https://github.com/pytorch/pytorch/blob/main/docs/source/amp.md).
These fixed implementation settings are recorded in runtime metadata.
