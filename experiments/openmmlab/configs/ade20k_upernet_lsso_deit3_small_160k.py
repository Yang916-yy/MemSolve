"""ADE20K UperNet 160k, LSSO DeiT III Small."""

_base_ = "./_base_/ade20k_upernet_160k.py"

custom_imports = dict(
    imports=["integrations.openmmlab"],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        type="LSSODeiT3Backbone",
        variant="small",
        rank=32,
        out_indices=(3, 5, 7, 11),
        implementation="cuda",
        core_mode="dynamic",
    ),
    decode_head=dict(
        in_channels=[384, 384, 384, 384],
        channels=384,
    ),
    auxiliary_head=dict(in_channels=384),
)
