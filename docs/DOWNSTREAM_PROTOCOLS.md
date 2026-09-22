# Dense Downstream Protocols

> Current source uses model contract 13 and native ABI 11. External position
> embeddings belong to the surrounding model. Historical measurements retain
> their recorded source versions and are not new-source results.


The dense experiments share the DeiT III LSSO backbone used by ImageNet. They
use a learned two-dimensional patch position table with no learned CLS
position. The learned spatial position table belongs to the backbone; the
mixer consumes token features and the validity mask.

| Scale | Width | Depth | Heads | LSSO rank | Feature taps |
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
token mask to every global LSSO mix. This prevents another image's batch
padding, including padding that partially overlaps an edge patch, from affecting
valid outputs.

## Protocol provenance

The historical ImageNet results used DeiT III training. The current classifier
uses plain ViT³-derived T/S/B training, documented in `docs/IMAGENET_VIT3.md`.
The dense S/B/L configurations below retain their original geometry and are
not newly validated downstream results under that training protocol.

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
The current ImageNet training entrypoint uses PyTorch AdamW and needs no Apex.

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

Current loading validates both ImageNet envelope format **5** and each LSSO
layer's model contract **12**. The native extension separately requires ABI
**8**. A valid envelope digest does not bypass a mismatched model contract.

A v0.6.3 ImageNet checkpoint may contain model contract 11 and therefore cannot
be loaded directly into current source. The repository does not yet provide an
automatic migration command. Preserve the original file; check tensor names,
shapes, geometry and numerical semantics before producing a migrated copy,
then validate loading and forward/backward behavior. Do not simply overwrite
`_extra_state` or use `strict=False` as a conversion procedure.

For new downstream training, `--backbone-checkpoint` initializes the encoder,
drops the ImageNet classifier, and allows the new pyramid/task heads to start
from their own initialization. ImageNet optimizer state is not a downstream
resume. `--resume` applies to an existing checkpoint of the downstream task.

The latest operator optimization was tested on SM120 with biased LSSO shapes.
The local verification environment did not contain timm, compiled MMCV,
MMDetection or MMSegmentation; it did not execute a complete detector or
segmenter. A matching compiled MMCV stack must be validated with the required
Torch/CUDA version before launching a full run. Operator timings are not
end-to-end throughput or evidence of COCO AP/ADE20K mIoU.

## Launch

The CUDA extension must have been built for every participating GPU
architecture with `tools/build_cuda.sh`. The launcher loads the strict artifact
before MMEngine constructs the model. New downstream runs require an explicit
ImageNet checkpoint; they never silently train a paper result from scratch.
The backbone verifies the checkpoint's current ImageNet contract and canonical
digest, then checks its tier, LSSO operator, and shared DeiT III geometry before
accepting any pretrained tensor. When a valid ImageNet checkpoint and the
downstream backbone use different learned 2D patch grids, the backbone applies
the same bicubic position-table interpolation used by ImageNet fine-tuning.

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/coco_mask_rcnn_lsso_deit3_base_3x.py \
  --data-root /datasets/coco \
  --backbone-checkpoint runs/imagenet/deit3_base_224/checkpoint_best.pt \
  --work-dir runs/coco/lsso_deit3_base_3x --launcher pytorch
```

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_openmmlab.py \
  experiments/openmmlab/configs/ade20k_upernet_lsso_deit3_base_160k.py \
  --data-root /datasets/ADEChallengeData2016 \
  --backbone-checkpoint runs/imagenet/deit3_base_224/checkpoint_best.pt \
  --work-dir runs/ade20k/lsso_deit3_base_160k --launcher pytorch
```

Use `--resume` for a downstream checkpoint, `--resume auto` for the latest
checkpoint in the work directory, and `--test CHECKPOINT` for evaluation.
The six leaf configs are named by task and scale under
`experiments/openmmlab/configs/`.
