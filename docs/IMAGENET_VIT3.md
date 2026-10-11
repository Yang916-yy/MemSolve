> Structure update: MemSolve now uses the restored no-RankRotary accretive
> operator. Vision uses CPE/GELU and token-LN mean pooling without LayerScale,
> following ordinary ViT3; BERT retains its own SwiGLU shell. Historical results and recipe
> discussions below are not measurements of a new run of this restoration.
> See [the current architecture](ARCHITECTURE.md) and [core contract](CORE_CONTRACT.md).

# ImageNet with plain ViT³ training

The visual model family is **MemSolve-ViT-T/S/B**; ViT³ names the source of this
training recipe. From the source checkout, `import integrations.timm` registers
`memsolve_vit_tiny`, `memsolve_vit_small` and `memsolve_vit_base` with
`timm.create_model`. For example:

```python
import timm
import integrations.timm
from memsolve.ball import cuda

cuda.load()
model = timm.create_model("memsolve_vit_small", img_size=224,
                          implementation="cuda", pretrained=False).cuda()
```

Registered models and ImageNet training share `MemSolveViT`. `img_size` selects
the input resolution; the T/S/B geometries below stay the same. The source
factory also accepts the explicit `image_size` argument. The timm entrypoints
default to `implementation="reference"`; select CUDA explicitly and use BF16
autocast with FP32 parameters. `pretrained=True` is rejected until weights for
the current contract are available. Use `forward_intermediates` for feature maps.

Select this recipe explicitly with `--config experiments/configs/imagenet_vit3.toml`.
All tiers now default to the [400-epoch RoPE-ViT-derived adaptation](IMAGENET_DEIT3.md),
with a separate 20-epoch resolution fine-tuning phase for Base.
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

| Tier | Width | Depth | Heads | MemSolve rank | Maximum DropPath |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 192 | 12 | 6 | 16 | 0.0 |
| small | 384 | 12 | 6 | 32 | 0.1 |
| base | 768 | 12 | 12 | 48 | 0.4 |

The restored S model uses a GELU MLP with width1584 (`mlp_ratio=4.125`),
matching the archived Small experiment. T/B use ratio4.0 (width768/3072).
T/S/B parameter counts including the 1,000-class head are
5,053,936 / 19,876,336 / 77,980,792 in equilibrium mode.

The shared visual adapter uses residual depthwise 3x3 CPE before each block,
Pre-LayerNorm, GELU MLP and no LayerScale or CLS token.
DropPath increases linearly to the tier's maximum. At224/16 there are196 patch tokens. CPE operates only on patches; no learned absolute patch table is used.
The final readout is per-token LayerNorm, valid-patch mean and the classifier.
The operator remains the historical no-RankRotary core; the readout and
LayerScale choices now follow ordinary ViT3. The whole model is not upstream ViT3.

The operator defaults to equilibrium; complement omits the learned core.
There is no RankRotary, independent Q/K projection, Q/K convolution, output
RMSNorm or channel gate. See [the mathematical contract](CORE_CONTRACT.md).

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
Worker quotas use complete physical batches. The separate
[DeiT III recipe](IMAGENET_DEIT3.md) explicitly enables virtual augmentation
groups and repeated augmentation; it does not change the ViT³ recipe above.

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
blocks with TorchInductor. The MemSolve CUDA boundary remains outside compilation,
and one outer CUDA Graph captures the compiled segments and MemSolve calls.
Inductor's own CUDA Graphs are disabled for this mode.
The compiler variant budget accounts for each block's distinct DropPath
probability and train/eval mode, avoiding the default eight-variant limit.

Both Graph modes require CUDA and fixed microbatch shapes. They capture an
entire accumulated update, divide each microbatch loss by `grad_accum`, and
synchronize DDP gradients only on the last microbatch. Optimizer and scheduler
updates remain outside the graph. The default two-GPU configuration uses batch
512 per GPU and accumulation 1. See [DeiT III](IMAGENET_DEIT3.md) for the
two-microbatch, global-batch-2048 configuration.

## Launch and resume

Use the current source checkout with CUDA PyTorch and Triton;
install vision dependencies with `python -m pip install -e '.[vision]'`.
Apex is not required. The notebook
[`imagenet_launcher.ipynb`](../notebooks/imagenet_launcher.ipynb) validates an
existing environment and launches the same entrypoint.

```bash
torchrun --standalone --nproc_per_node=2 experiments/train_imagenet.py \
  --config experiments/configs/imagenet_vit3.toml \
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

The ImageNet checkpoint envelope remains12. The restored vision scaffold
contract is7 and the operator contract26; native ABI is12. Metadata records
`vit3_cpe_accretive_mean_gelu_v7`, GELU, token-LN mean pooling, no LayerScale and CPE. Previous Q/K/V
checkpoints cannot resume as this model. Historical Small weights were checked
against the archived source after explicitly translating contract metadata;
this does not add an automatic checkpoint migration API.

BF16 autocast remains a local precision choice. The restored native backend uses
BF16 packed projections and FP32 compact solves; parameters remain FP32.
The current training utilities retain persistent workers and later fixes;
structural restoration is not a claim to reproduce all historical sampling or
augmentation settings. No new training run is started by this change.

The optimizer uses ordinary fused AdamW. The learned shared core coordinates
start at zero (J=I) and receive the configured weight decay; there are no
constraint-optimizer hooks. Projection initialization follows the restored
relation/content model. See [the core contract](CORE_CONTRACT.md).
