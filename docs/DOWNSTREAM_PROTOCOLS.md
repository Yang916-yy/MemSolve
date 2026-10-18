# Dense vision: ViG-style ViT-Adapter

MemSolve uses its pretrained **CPE/GELU, no-CLS, no-LayerScale** encoder inside
a full ViT-Adapter. The four interaction stages cover blocks 0–2, 3–5, 6–8,
9–11. A convolutional spatial prior supplies stride 4/8/16/32 maps; deformable
attention injects spatial features into the tokens and extracts updated spatial
features after each block group. The last stage has two additional extractors.
The resulting maps feed standard OpenMMLab Mask R-CNN/FPN or UPerNet heads.
This is not a hierarchy added inside the MemSolve operator.

## Sources and scope

Here **ViG means Gated Linear Attention** ([AAAI 2025 paper, Appendix C](https://arxiv.org/html/2405.18425v2)),
not Vision GNN. The paper explicitly uses ViT-Adapter following Vision-RWKV.
Its [official repository](https://github.com/hustvl/ViG) does not distribute the
dense downstream code. Consequently these are **ViG-derived recipes**, with
implementation details taken from [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV/tree/0e0e38a6fd4475ae3817e14420ab85fab08d9097)
and [ViT-Adapter](https://github.com/czczup/ViT-Adapter/tree/fd64a5190f9f4530e8e5961cc87b2a3ad591190b).
They are not claimed to be an exact reproduction of an unpublished ViG config.
Code attribution and Apache-2.0 notices are in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).
No downstream accuracy results have been produced yet.

| Scale | Width / depth / heads | Rank | MLP ratio | Adapter heads / value ratio | DropPath |
| --- | --- | ---: | ---: | --- | ---: |
| Tiny | 192 / 12 / 6 | 16 | 4 | 6 / 1 | 0.1 |
| Small | 384 / 12 / 6 | 32 | 4.125 | 6 / 1 | 0.2 |
| Base | 768 / 12 / 12 | 48 | 4 | 12 / 0.5 | 0.3 |

All scales use spatial-prior width 64, four sampling points, CFFN ratio 0.25,
SyncBN spatial features and zero-initialized injector scaling. Adapter scaling
is separate from the encoder, which still has no LayerScale. New adapter
parameters follow upstream initialization; pretrained encoder parameters are
not reinitialized. Dense model parameter counts include adapter and task heads
and must not be reported as classifier backbone counts.

## Recipes

| Setting | COCO 2017 | ADE20K |
| --- | --- | --- |
| Head | Mask R-CNN + FPN | UPerNet + auxiliary FCN |
| Training length | 12 epochs (1x) | 160,000 optimizer steps |
| Global batch | 16 | 16 |
| AdamW LR | 1e-4, all scales | 1.2e-4 Tiny; 6e-5 Small/Base |
| Weight decay | 0.05 | 0.01 |
| Layer LR decay | 0.85 | 0.85 Tiny/Small; 0.8 Base |
| Warmup | 500 steps, factor 0.001 | 1,500 steps, factor 1e-6 |
| LR decay | ×0.1 at epochs 8, 11 | linear polynomial to zero |
| Train images | resize to fit 1333×800, flip | random scale 0.5–2, 512² crop, flip, photometric distortion |
| Evaluation | single-scale 1333×800 | single-scale, 512² sliding windows, stride 341 |

The learning rates, weight decay and budgets follow ViG Appendix C. In
particular, ViG's stated COCO LR/WD are used for Tiny too, although the referenced
Vision-RWKV Tiny config differs. Adapter dimensions, layer decay and augmentation
details follow Vision-RWKV. UPerNet working width is 512 for all scales.

Defaults target **two GPUs, eight images per GPU**, with four persistent loader
workers per GPU. This preserves the upstream global batch of 16 (originally
eight GPUs × two images). `with_cp=True` checkpoints encoder blocks and
injector/extractor branches if memory requires it; spatial-prior BatchNorm is
not recomputed. No training has been launched to establish throughput yet.

FP32 weights and **BF16 autocast** are used instead of the upstream FP16 runner.
MemSolve keeps its own mixed-precision numerical contract. Deformable sampling
uses upstream's FP32 precision island with MMCV's compiled CUDA operator; its
projections remain under autocast. The detection wrappers likewise use FP32
RoIAlign and cast final RPN/box predictions to FP32 before box decoding/NMS:
MMCV 2.1 lacks BF16 RoIAlign and requires matching NMS dtypes. Head convolutions
and linear layers still run under BF16 autocast. BF16 gradient scaling is
disabled. The CPU grid-sample implementation is the
upstream reference, not a silent CUDA fallback.

## Checkpoints and padding

`--backbone-checkpoint` loads the repository's ImageNet checkpoint envelope.
Tier, geometry and operator mode must match. Small keeps the actual pretrained
MLP ratio 4.125. The classifier and final classification LayerNorm are removed,
as in ViT-Adapter; per-block CPE and block norms are retained. Downstream
DropPath may differ from pretraining. `checkpoint_weights=auto` honors
`selected_weights`, so `checkpoint_best_ema.pt` initializes from **model_ema**.
Set `--cfg-options model.backbone.checkpoint_weights=model` to explicitly select
raw weights from a checkpoint containing both. Model optimizer state is not
transferred between classification and dense tasks.

CPE now accepts an explicit rectangular grid without changing classification
weights. Inputs are padded to multiples of 32 for the spatial prior. Image
metadata determines the valid pixels and tokens; padded pixel values are zeroed
before either branch, and the global MemSolve solve excludes invalid tokens.
Sliding inference slices the same validity mask for each crop and leaves the
caller's metadata unchanged. Convolution/SyncBN padding behavior otherwise
follows the adapter: this is not a claim of invariance to changing batch canvas
size or BatchNorm statistics.

## Environment

Use a dedicated environment so OpenMMLab does not change an active training stack:

```bash
pip install -e '.[vision,openmmlab]'
```

The tested Python interface versions are MMEngine 0.10.7, MMDetection 3.3.0,
MMSegmentation 1.2.2, and MMCV 2.1.0. Install **compiled MMCV**, matching the local
Torch/CUDA build; `mmcv-lite` cannot run Mask R-CNN or CUDA deformable attention.
See [MMCV installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
Torch 2.14 requires C++20 when rebuilding MMCV 2.1.0, whose setup defaults to
C++17. The build helper pins upstream commit `57c4e25` and selects C++20 when
required by Torch headers:

```bash
PYTHON=/path/to/dense-venv/bin/python TORCH_CUDA_ARCH_LIST=8.0 bash tools/build_mmcv.sh
```

Run this in the dedicated downstream environment. It replaces `mmcv-lite`
there with the compiled build. Do not substitute an old prebuilt CUDA wheel
from a different Torch ABI.

## Data and launch

Datasets stay outside the repository. COCO root contains `train2017/`,
`val2017/`, `annotations/instances_train2017.json` and
`annotations/instances_val2017.json`. ADE root is `ADEChallengeData2016/`, with
`images/{training,validation}` and `annotations/{training,validation}`.
ADE's original label PNGs are retained; the loader handles reduce-zero-label.

```bash
torchrun --standalone --nproc_per_node=2 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/coco_mask_rcnn_memsolve_adapter_small_1x.py \
  --launcher pytorch --data-root /datasets/coco \
  --backbone-checkpoint /checkpoints/small/checkpoint_best.pt \
  --work-dir /runs/coco/memsolve_small_adapter

torchrun --standalone --nproc_per_node=2 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/ade20k_upernet_memsolve_adapter_small_160k.py \
  --launcher pytorch --data-root /datasets/ADEChallengeData2016 \
  --backbone-checkpoint /checkpoints/small/checkpoint_best.pt \
  --work-dir /runs/ade20k/memsolve_small_adapter
```

Tiny and Base leaf configs are alongside Small. For a different GPU count,
choose per-GPU batch so global batch remains 16, or explicitly request
`--auto-scale-lr`. Use `--resume` for the latest downstream checkpoint,
`--resume /path/checkpoint.pth` for a specific checkpoint, and
`--test /path/checkpoint.pth` for evaluation; omit `--backbone-checkpoint` for
resume/test. `--cfg-options model.backbone.with_cp=True` enables checkpointing.

## Validation (2026-10-17)

The dedicated local environment is `/cywang/memsolve-dense-venv` (Torch
2.14.0+cu132 / MMCV 2.1.0 rebuilt with C++20). Verification covers:

- 31 CPU tests across the dense adapter and timm integration, including real
  MMDetection/MMSegmentation data preprocessors, task heads, loss backward,
  and prediction under FP32 and BF16 autocast.
- Rectangular CPE, padded pixel invariance, sliding-window mask handling,
  EMA selection, layer LR decay and activation-checkpoint gradient/BN parity.
- The actual pretrained Base best-EMA checkpoint loads successfully. Backbone
  plus adapter has 90,890,864 parameters before task heads.
- A native Tiny CUDA adapter BF16 forward/backward, including MemSolve's fast
  path and MMCV deformable sampling, with finite gradients for all parameters.
  The configured MMEngine BF16 optimizer wrapper also completes an AdamW step.
- MMCV CUDA sampling forward/gradients against the upstream CPU oracle.

These are synthetic correctness checks, not full-resolution throughput,
distributed training, COCO AP or ADE20K mIoU measurements. Formal downstream
training and dataset downloads have not been started.
