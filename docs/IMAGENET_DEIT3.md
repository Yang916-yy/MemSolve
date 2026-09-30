# Ridgon-B with the DeiT III training recipes

`experiments/configs/imagenet_deit3_400.toml` adapts the ImageNet-1K recipe in
[DeiT III](https://arxiv.org/abs/2204.07118), tables 1, 6 and 13. The backbone
retains Ridgon-B/r48, packed SwiGLU, per-layer residual CPE, token-LN mean
pooling, no CLS and no LayerScale. BF16 AMP, no EMA, local indexed WebDataset and
physical-batch gradient accumulation are local adaptations. This is a
**DeiT III-derived training protocol**, not the original ViT architecture or
an identical ImageFolder sampling protocol.

`experiments/configs/imagenet_deit3_800.toml` implements the longer Base recipe
from the [official training and fine-tuning commands](https://github.com/facebookresearch/deit/blob/main/README_revenge.md).
It trains at 192px for 800 epochs with weight decay 0.05 and uniform DropPath
0.2, followed by 20 epochs at 224px with weight decay 0.1 and DropPath 0.2.
It defaults to `compile-graph`: PyTorch Inductor fuses the surrounding vision
operations while the outer CUDA Graph also captures the Ridgon kernels and
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
CPE has no resolution-dependent learned position table to interpolate. The
optimizer, schedule, epoch counter, best metric and random streams start anew;
the source checkpoint is recorded in `metrics.jsonl`. Use `--resume` for an
interrupted run within the same phase. `--resume` and `--finetune` are mutually
exclusive, and output directories cannot overwrite an existing experiment.

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
surrounding vision blocks while retaining the Ridgon CUDA boundary.

Compiled execution preserves eager BF16 rounding through Inductor's
[`emulate_precision_casts`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py)
option. Fused kernels keep the cast boundaries without materializing every
intermediate tensor. AOTAutograd uses `backward_pass_autocast="off"` because
the entrypoint runs backward outside forward autocast, following the
[PyTorch AMP guidance](https://github.com/pytorch/pytorch/blob/main/docs/source/amp.md).
These fixed implementation settings are recorded in runtime metadata.
