"""ADE20K UperNet 160k, LSSO DeiT III Large."""

_base_ = "./_base_/ade20k_upernet_160k.py"

custom_imports = dict(
    imports=["integrations.openmmlab"],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        type="LSSODeiT3Backbone",
        variant="large",
        rank=64,
        out_indices=(7, 11, 15, 23),
        implementation="cuda",
        core_mode="dynamic",
    ),
    decode_head=dict(in_channels=[1024, 1024, 1024, 1024], channels=512),
    auxiliary_head=dict(in_channels=1024),
)
