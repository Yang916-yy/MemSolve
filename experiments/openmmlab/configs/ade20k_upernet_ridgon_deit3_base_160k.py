"""ADE20K UperNet 160k, Ridgon DeiT III Base."""

_base_ = "./_base_/ade20k_upernet_160k.py"

custom_imports = dict(
    imports=["integrations.openmmlab"],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        type="RidgonViTBackbone",
        variant="base",
        rank=48,
        out_indices=(3, 5, 7, 11),
        implementation="cuda",
    ),
    decode_head=dict(in_channels=[768, 768, 768, 768], channels=512),
    auxiliary_head=dict(in_channels=768),
)
