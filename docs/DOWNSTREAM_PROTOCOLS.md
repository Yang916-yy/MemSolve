# Dense Vision: COCO Detection & ADE20K Segmentation

For dense prediction, we wrap the pretrained non-hierarchical **MemSolve-ViT** backbone (CPE + GELU, no CLS token, no LayerScale) in a standard **ViT-Adapter** and attach OpenMMLab Mask R-CNN + FPN (for COCO 2017) or UPerNet (for ADE20K).

The encoder blocks are divided into four equal interaction stages (`[0–2, 3–5, 6–8, 9–11]` for 12-layer Tiny/Small/Base, or 6 blocks per stage for 24-layer Large). A convolutional spatial prior extracts stride 4/8/16/32 feature maps, and multi-scale deformable attention injects spatial features into the token stream and extracts multi-scale pyramids after each stage.

## Architecture & Hyperparameters

Our setup follows the ViT-Adapter recipes used by [ViG (AAAI 2025, Appendix C)](https://arxiv.org/html/2405.18425v2) and [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV/tree/0e0e38a6fd4475ae3817e14420ab85fab08d9097):

| Scale | Width / depth / heads | Rank | MLP ratio | Adapter heads / value ratio | DropPath |
| --- | --- | ---: | ---: | --- | ---: |
| Tiny | 192 / 12 / 6 | 16 | 4.0 | 6 / 1.0 | 0.1 |
| Small | 384 / 12 / 6 | 32 | 4.125 | 6 / 1.0 | 0.2 |
| Base | 768 / 12 / 12 | 48 | 4.0 | 12 / 0.5 | 0.3 |

All three paper tiers use a 64-channel spatial prior, 4 deformable sampling points, CFFN ratio 0.25, `SyncBN`, and zero-initialized injector scaling (`MemSolveViTBackbone` also supports `variant="large"` with width 1024, depth 24, 16 heads, rank 64, and default `deform_num_heads=16`).

| Setting | COCO 2017 (Mask R-CNN 1×) | ADE20K (UPerNet 160k) |
| --- | --- | --- |
| Task head | Mask R-CNN + FPN | UPerNet (width 512) + auxiliary FCN |
| Schedule | 12 epochs | 160,000 steps |
| Global batch size | 16 (e.g., 2 GPUs × 8 images) | 16 (e.g., 2 GPUs × 8 images) |
| AdamW peak LR | 1e-4 (all scales) | 1.2e-4 (Tiny); 6e-5 (Small / Base) |
| Weight decay | 0.05 | 0.01 |
| Layer-wise LR decay | 0.85 | 0.85 (Tiny / Small); 0.80 (Base) |
| Warmup | 500 steps (linear, factor $10^{-3}$) | 1,500 steps (linear, factor $10^{-6}$) |
| LR schedule | Step decay ($\times 0.1$ at epochs 8, 11) | Polynomial decay to 0 |
| Training resolution | Resize within $1333 \times 800$, random flip | Random scale $0.5\text{–}2.0\times$, $512 \times 512$ crop, photometric distortion |
| Evaluation | Single-scale $1333 \times 800$ | Single-scale $512 \times 512$ sliding window (stride 341) |

### Mixed Precision & Rectangular Grids

- **Precision**: Models keep FP32 parameters and run under **BF16 autocast** (without loss scaling) during both training and evaluation. Deformable attention sampling, RoIAlign, and box decoding/NMS run in FP32 to match MMCV 2.1's compiled kernel types.
- **Arbitrary Image Sizes & Padding**: The backbone's depthwise CPE operates on the dynamic $(H_p, W_p)$ patch grid of each padded batch canvas (padded to multiples of 32 for the spatial prior). Padded pixels are zeroed before entering either branch, and the MemSolve layer masks out padded patch tokens so they do not contribute to the global equilibrium state.
- **Checkpoints**: Passing `--backbone-checkpoint` loads an ImageNet checkpoint (`checkpoint_best_ema.pt` automatically uses `model_ema` weights when `checkpoint_weights="auto"`), stripping the classification head and final norm.

## Environment Setup

Install the OpenMMLab dependencies in a dedicated virtual environment and compile MMCV against your local PyTorch/CUDA installation (`mmcv-lite` does not include the CUDA ops needed by Mask R-CNN and deformable attention):

```bash
pip install -e '.[vision,openmmlab]'
PYTHON=/path/to/venv/bin/python TORCH_CUDA_ARCH_LIST=8.0 bash tools/build_mmcv.sh
```

Tested with MMEngine 0.10.7, MMDetection 3.3.0, MMSegmentation 1.2.2, and MMCV 2.1.0.

## Running Training & Evaluation

Prepare standard COCO (`train2017/`, `val2017/`, `annotations/`) and ADE20K (`ADEChallengeData2016/images/`, `annotations/`) directories, then launch with `torchrun`:

```bash
# COCO 2017 object detection & instance segmentation (1x schedule)
torchrun --standalone --nproc_per_node=2 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/coco_mask_rcnn_memsolve_adapter_small_1x.py \
  --launcher pytorch --data-root /datasets/coco \
  --backbone-checkpoint /checkpoints/small/checkpoint_best_ema.pt \
  --work-dir /runs/coco/memsolve_small_adapter

# ADE20K semantic segmentation (160k steps)
torchrun --standalone --nproc_per_node=2 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/ade20k_upernet_memsolve_adapter_small_160k.py \
  --launcher pytorch --data-root /datasets/ADEChallengeData2016 \
  --backbone-checkpoint /checkpoints/small/checkpoint_best_ema.pt \
  --work-dir /runs/ade20k/memsolve_small_adapter
```

- Switch `small` to `tiny` or `base` in the config filename for the other model sizes.
- Add `--cfg-options model.backbone.with_cp=True` to enable activation checkpointing if GPU memory is tight.
- Add `--cfg-options model.backbone.compile_blocks=True` to dynamically `torch.compile` full `MemSolveViTBackbone` encoder blocks (including registered `MemSolve` custom ops with no Dynamo graph breaks; CUDA Graphs remain disabled for variable detection/segmentation canvas shapes).
- Use `--resume` (or `--resume /path/to/checkpoint.pth`) to resume training, or `--test /path/to/checkpoint.pth` to run evaluation only.
- Published training logs and resolved configs for our Small and Base runs are in [`results/coco/`](../results/coco/) and [`results/ade20k/`](../results/ade20k/).
