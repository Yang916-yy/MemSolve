> Structure update: MemSolve now uses the restored no-RankRotary accretive
> operator. Vision uses CPE/GELU and token-LN mean pooling without LayerScale,
> following ordinary ViT3; BERT retains its own SwiGLU shell. Historical results and recipe
> discussions below are not measurements of a new run of this restoration.
> See [the current architecture](ARCHITECTURE.md) and [core contract](CORE_CONTRACT.md).

# Optional dense downstream adapters

Detection and segmentation are outside the current experimental scope. The
adapters below are retained for development; no current MemSolve downstream
accuracy or complete detector/segmenter validation is claimed.

> Current source uses model contract 26 and native CUDA ABI 16. External position
> embeddings belong to the surrounding model. Historical measurements retain
> their recorded source versions and are not new-source results.


The optional dense adapter uses the same restored operator with a historical
GELU/mean-pooling encoder without LayerScale and learned patch positions for dynamic padded grids.
It deliberately retains that adapter's learned-position path; the ImageNet
CPE default is not a compatible pretrained backbone without an explicit study.
These adapters are outside the current experimental scope.

| Scale | Width | Depth | Heads | MemSolve rank | Feature taps |
| --- | ---: | ---: | ---: | ---: | --- |
| Small | 384 | 12 | 6 | 32 | `(3, 5, 7, 11)` |
| Base | 768 | 12 | 12 | 48 | `(3, 5, 7, 11)` |
| Large | 1024 | 24 | 16 | 64 | `(7, 11, 15, 23)` |

The four plain-ViT maps are converted to spatial strides `4, 8, 16, 32` with
the same simple feature pyramid convention used by XCiT: two `2x` transpose
convolutions, one transpose convolution, identity, and one `2x` max pool.
The external FPN or UperNet head then consumes those four maps.

The OpenMMLab wrappers derive an image-validity mask from each sample's
`img_shape`, zero padded pixels before patch embedding, and pass the resulting
token mask to every global MemSolve mix. This prevents another image's batch
padding, including padding that partially overlaps an edge patch, from affecting
valid outputs.

## Protocol provenance

The historical ImageNet results used DeiT III training. The current classifier
supports both [ViT³-derived](IMAGENET_VIT3.md) and
[DeiT III-derived](IMAGENET_DEIT3.md) training. The dense S/B/L configurations
below retain their original geometry and are not validated downstream results
for either current recipe.

COCO is a **downstream standard protocol**, not an official DeiT III detection
recipe. It is the public [XCiT Mask R-CNN + FPN 3x
protocol](https://github.com/facebookresearch/xcit/tree/82f5291f412604970c39a912586e008ec009cdca/detection):
COCO 2017, batch size 2 per GPU, XCiT multi-scale crop augmentation, AdamW
`1e-4` with weight decay `0.05`, 36 epochs, 500-iteration linear warmup, and
milestones at epochs 27 and 33.

ADE20K follows the DeiT III paper's UperNet evaluation and the public
[XCiT UperNet 160k
protocol](https://github.com/facebookresearch/xcit/tree/82f5291f412604970c39a912586e008ec009cdca/semantic_segmentation):
ADE20K 150 classes, batch size 2 per GPU, 512px crop, AdamW `6e-5` with
weight decay `0.01`, 1,500-iteration warmup, and 160,000 iterations. The
small decoder uses 384 working channels, while base and large use 512.

## Environment

Install the vision and OpenMMLab Python packages first:

```bash
pip install -e '.[vision,openmmlab]'
```

Then install a **compiled** `mmcv==2.1.*` build matched to the active PyTorch
and CUDA stack using the [OpenMMLab installation
guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
`mmcv-lite` does not provide the CUDA/C++ operators required by Mask R-CNN.
These downstream launchers use AdamW. ImageNet
[DeiT III pretraining](IMAGENET_DEIT3.md#nvidia-apex) separately requires Apex.

## Dataset layout

COCO 2017 provides both detection boxes and instance masks for Mask R-CNN.
Download `train2017.zip`, `val2017.zip`, and
`annotations_trainval2017.zip` from the [official COCO download page](https://cocodataset.org/#download).
The current configs use the train and validation splits; panoptic/stuff
annotations are not used by this Mask R-CNN recipe.

For semantic segmentation, use `ADEChallengeData2016.zip`, the scene-parsing
benchmark package derived from ADE20K, from [MIT Scene Parsing](https://sceneparsing.csail.mit.edu/).
The `2016` in the archive name is expected. Keep its original label PNGs;
the configured loader handles the label-zero convention.

The root passed to `--data-root` must have the following contents:

```text
/datasets/coco/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json

/datasets/ADEChallengeData2016/
  images/training/
  images/validation/
  annotations/training/
  annotations/validation/
```

Keep datasets outside the Git checkout. COCO's root points directly to the
folder containing `train2017`; ADE20K's root points to `ADEChallengeData2016`,
not its parent folder. Match Small/Base/Large configuration to the pretrained
backbone; the launch examples below use Base.

## Pretrained checkpoint compatibility

Loading validates the current ImageNet envelope and exact operator/vision
contracts. A Q/K/V checkpoint cannot be loaded into the restored operator.
The CPE ImageNet default is also distinct from this learned-position adapter;
no new downstream results or checkpoint compatibility are claimed.

The current compatibility check still expects the ViT³-style linear DropPath
schedule. It rejects the constant-DropPath DeiT III checkpoints even when their
weights have compatible shapes; that adapter limitation remains to be fixed
before using those checkpoints downstream.

For new downstream training, `--backbone-checkpoint` initializes the encoder,
drops the ImageNet classifier, and allows the new pyramid/task heads to start
from their own initialization. ImageNet optimizer state is not a downstream
resume. `--resume` applies to an existing checkpoint of the downstream task.

Current operator validation targets SM80/A800; see the
[CUDA contract](CUDA_CONTRACT.md). Adapter CPU checks do not validate a complete
MMDetection detector or MMSegmentation segmenter. A matching compiled MMCV stack must be validated with the required
Torch/CUDA version before launching a full run. Operator timings are not
end-to-end throughput or evidence of COCO AP/ADE20K mIoU.

## Launch

The launcher loads the matching native CUDA extension before model construction.
The backbone verifies the checkpoint digest, tier, operator and vision geometry.
Learned position tables are interpolated for its dynamic grids. Config filenames
containing `deit3` retain their historical paths; the registered class remains
MemSolveViTBackbone.

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/coco_mask_rcnn_memsolve_deit3_base_3x.py \
  --data-root /datasets/coco \
  --backbone-checkpoint runs/imagenet/deit3_base_224/checkpoint_best.pt \
  --work-dir runs/coco/memsolve_deit3_base_3x --launcher pytorch
```

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/ade20k_upernet_memsolve_deit3_base_160k.py \
  --data-root /datasets/ADEChallengeData2016 \
  --backbone-checkpoint runs/imagenet/deit3_base_224/checkpoint_best.pt \
  --work-dir runs/ade20k/memsolve_deit3_base_160k --launcher pytorch
```

Use `--resume` for a downstream checkpoint, `--resume auto` for the latest
checkpoint in the work directory, and `--test CHECKPOINT` for evaluation.
The six leaf configs are named by task and scale under
`experiments/openmmlab/configs/`.
