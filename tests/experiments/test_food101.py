import torch
from experiments.food101 import build_model, learning_rate


def test_food_model_uses_current_qkv_operator_and_existing_grid():
    torch.set_num_threads(2)
    model = build_model()
    assert model.encoder.pos_embed is None
    assert model.encoder.cls_token is None
    assert isinstance(model.encoder.fc_norm, torch.nn.Identity)
    assert len(model.encoder.blocks) == 12
    for block in model.encoder.blocks:
        assert isinstance(block.ls1, torch.nn.Identity)
        assert block.cpe.kernel_size == (3, 3)
        mixer = block.attn.mixer
        assert mixer.w_qkv.out_features == 2 * 6 * 32 + 384
        assert block.attn.implementation == "reference"


def test_schedule_has_three_epoch_warmup_and_fixed_final_floor():
    assert learning_rate(299, 100, 30, 5e-4) == 5e-4
    assert learning_rate(300, 100, 30, 5e-4) == 5e-4
    assert learning_rate(2999, 100, 30, 5e-4) == 1e-6
