"""MemSolve-small: ViG-derived ADE20K 160k with full ViT-Adapter."""
_base_ = "./_base_/ade20k_upernet_160k.py"
custom_imports = dict(imports=["integrations.openmmlab"], allow_failed_imports=False)
model = dict(
    backbone=dict(type="MemSolveViTBackbone", variant="small", rank=32,
                  core_mode="equilibrium", implementation="cuda",
                  drop_path_rate=0.2, deform_num_heads=6, deform_ratio=1.0,
                  conv_inplane=64, cffn_ratio=0.25, n_points=4, with_cp=False),
    decode_head=dict(in_channels=[384] * 4, channels=512),
    auxiliary_head=dict(in_channels=384),
)
optim_wrapper = dict(optimizer=dict(lr=6e-05),
                     paramwise_cfg=dict(layer_decay_rate=0.85))
