"""COCO 2017 Mask R-CNN + FPN 3x, MemSolve DeiT III Large."""

_base_ = "./_base_/coco_mask_rcnn_fpn_3x.py"

custom_imports = dict(
    imports=["integrations.openmmlab"],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        type="MemSolveViTBackbone",
        variant="large",
        rank=64,
        out_indices=(7, 11, 15, 23),
        implementation="cuda",
    ),
    neck=dict(in_channels=[1024, 1024, 1024, 1024]),
)
