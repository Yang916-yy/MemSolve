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

| Tier | Width | Depth | Heads | Ridgon rank | Maximum DropPath |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 192 | 12 | 6 | 16 | 0.0 |
| small | 384 | 12 | 6 | 32 | 0.1 |
| base | 768 | 12 | 12 | 48 | 0.4 |

The small model uses **SwiGLU with gate width 1024**, a packed input
projection of width 2048, and 20,327,272 parameters including the 1000-class
head. `mlp_ratio=4.0` denotes the equivalent two-projection MLP weight budget:
the gate width is `ceil_to_16(floor(2 * int(width * mlp_ratio) / 3))`.
All three tiers use budget ratio 4.0, so the gate width is exactly `(8/3) * width`:
512 / 1024 / 2048 for T / S / B. Their parameter counts including the
1000-class head are 5,279,656 / 20,327,272 / 83,309,032.
This replaces the original ViT³ GELU FFN; it is a local architecture choice,
not part of the upstream recipe. Historical results and speed measurements
belong to their recorded model versions.

The packed forward reuses [timm GluMlp](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py)
with SiLU on the first half and multiplication by the second half. The
parameter-budget adjustment follows the same two-thirds principle used by
[DINOv2 SwiGLU](https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py).
Both input branches and the output projection use timm's linear initialization
(truncated normal, std 0.02; zero biases). We bypass GluMlp's near-constant
second-half initialization, which targets the value branch with this packing.


DropPath increases linearly from zero to the listed maximum across blocks,
as in plain ViT³. Ridgon ranks are our model choices. The canonical
`integrations.timm.create_ridgon_vit` encoder uses residual 3×3 depthwise CPE
before each Pre-LN block, with no absolute position table, CLS, or LayerScale.
CPE uses PyTorch/cuDNN and keeps masked patch features out of neighboring valid
updates. At 224/16, all 196 tokens are image patches. The final readout is
per-token LayerNorm (ε=1e-6), then patch mean, then the classifier. timm's
post-pooling `fc_norm` is explicitly disabled; moving LN after the mean would
change the architecture. Masked pooling averages only valid patches.

The mixer retains independent Q/K/V, a shared learned core, direct readout and
per-head RMSNorm with a shared channel gain initialized to one and epsilon
`1e-6`, following the FLA Gated DeltaNet model configuration (see [core contract](CORE_CONTRACT.md));
it has no projection-local convolution or rotation. This is
**Ridgon in a plain ViT³-style visual scaffold with its derived training recipe**.
It does not use the upstream TTT mixer, adaptive convolution, or MESA variant.

## Batching and data

`batch_size` is physical per GPU. Accumulation is derived so that
`world_size × batch_size × grad_accum = 1024`; LR is not rescaled again.
On two GPUs, batch 512 per GPU gives accumulation 1. There are 1,251 optimizer
updates per epoch (1,281,024 scheduled images), dropping the incomplete update.
`--grad-accum` optionally asserts the derived value.

Mixup/CutMix is applied once to the complete physical per-GPU batch, as in
[the official ViT³ training loop](https://github.com/LeapLabTHU/ViTTT/blob/e3477587d099e6b9e83e9e7c80b1b999e0989a20/vittt/main.py).
With two GPUs, each call receives 512 samples. There is no virtual augmentation
group size or divisibility-by-128 requirement. The upstream default batch size
of 128 is configurable; it is not an additional augmentation grouping rule.
Changing physical batch size changes the batch-mode mixing correlations and
pairing pool, even when gradient accumulation preserves the effective batch.
Worker quotas and optional repeated augmentation use complete physical batches.

The existing `timm/imagenet-1k-wds` data contract is retained: `_info.json`,
1,024 training tar shards and 64 validation tar shards, with a pinned manifest
SHA-256 and structural preflight. No ImageFolder extraction is needed.
Training uses one transformed view per source record, shuffled shards and a
bounded 8,192-record shuffle buffer (initial fill 2,048). This streaming order
is not the upstream ImageFolder/DistributedSampler permutation. Every rank
currently reads the full 50,000-image validation set; reductions preserve the
metric, at the cost of redundant validation work. Training and validation
worker counts are independent (defaults 12 and 4).

## Training execution

The default `execution = "graph"` captures fixed-shape model forward and
backward, including DDP gradient synchronization. Optimizer updates, gradient
clipping, augmentation and scheduling remain outside the CUDA Graph.
Capture warmup does not update parameters and restores RNG state and model
buffers before the first training replay. The graph is reused across epochs
and released before distributed shutdown.

Use `--execution compile-graph` to additionally compile the surrounding vision
blocks with TorchInductor. The Ridgon CUDA boundary remains outside compilation,
and one outer CUDA Graph captures the compiled segments and Ridgon calls.
Inductor's own CUDA Graphs are disabled for this mode.
The compiler variant budget accounts for each block's distinct DropPath
probability and train/eval mode, avoiding the default eight-variant limit.

Both Graph modes require CUDA, fixed input shapes and `grad_accum = 1`.
Use `--execution eager` when a smaller physical batch requires gradient
accumulation. The default two-GPU configuration uses batch 512 per GPU and
therefore satisfies the Graph requirement.

## Launch and resume

Use the current source checkout with CUDA PyTorch and Triton;
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
Train and validation workers stay alive across epochs (`persistent_workers=True`
when workers are enabled). A shared epoch counter updates their dataset replicas;
each worker reseeds Python, NumPy and PyTorch from the run seed, epoch, rank and
worker ID at iterator creation, so fresh workers after resume reproduce that
epoch. Workers use `spawn` and prefetch one physical batch each. Defaults are
12 train / 4 validation workers per rank. This follows the lifecycle and replica
semantics in the [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/2.14/data.html).

The ImageNet checkpoint envelope is now **12**, vision scaffold contract **2**,
Ridgon model contract **19**, and CUDA contract **17**. The model metadata records
`vit3_cpe_mean_swiglu_v2`, SwiGLU, no CLS/LayerScale and token-LN mean pooling.
Previous GELU or CLS/LayerScale ImageNet states cannot resume as this model.
Historical classification results retain their original training protocol and
source version, including the previous shared-A ViT³-derived run. This refactor
produces no new accuracy results and does not relabel those measurements.

The BF16 autocast setting is a local precision adaptation from the upstream FP16 AMP entrypoint. Ridgon keeps its BF16/FP32 internal mixed-precision contract; model parameters and AdamW states remain FP32. No GradScaler is created or checkpointed.

The optimizer uses ordinary fused AdamW for all parameters. The
[identity-centered core](CORE_CONTRACT.md#identity-centered-parameterization-and-training)
stores Delta with zero initialization and forms T = I + Delta. Delta receives
the regular 0.05 weight decay, pulling T toward I; there are no optimizer hooks.

K projections use the Ridgon-owned fan-in normal initializer (variance `1/embed_dim`,
zero bias), applied after timm initializes the child projections. Q/V keep timm
std=0.02. This source-0.8.1 initialization change is a local Ridgon choice, not a
claim about the upstream ViT³ initializer; loaded checkpoint weights are restored.
