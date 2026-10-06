"""COCO 2017 Mask R-CNN + FPN 3x, MemSolve DeiT III Base."""

_base_ = "./_base_/coco_mask_rcnn_fpn_3x.py"

custom_imports = dict(
    imports=["integrations.openmmlab"],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        type="MemSolveViTBackbone",
        variant="base",
        rank=48,
        out_indices=(3, 5, 7, 11),
        implementation="cuda",
    ),
    neck=dict(in_channels=[768, 768, 768, 768]),
)
