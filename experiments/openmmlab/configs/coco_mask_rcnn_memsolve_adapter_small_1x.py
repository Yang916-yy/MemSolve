"""MemSolve-small: ViG-derived COCO 1x with full ViT-Adapter."""
_base_ = "./_base_/coco_mask_rcnn_fpn_1x.py"
custom_imports = dict(imports=["integrations.openmmlab"], allow_failed_imports=False)
model = dict(
    backbone=dict(type="MemSolveViTBackbone", variant="small", rank=32,
                  core_mode="equilibrium", implementation="cuda",
                  drop_path_rate=0.2, deform_num_heads=6, deform_ratio=1.0,
                  conv_inplane=64, cffn_ratio=0.25, n_points=4, with_cp=False),
    neck=dict(in_channels=[384] * 4),
)
