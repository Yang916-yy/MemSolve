import torch
from experiments.food101 import build_model, learning_rate


def test_food_model_uses_restored_operator_and_mean_scaffold():
    torch.set_num_threads(2)
    model = build_model()
    assert model.encoder.pos_embed is None
    assert model.encoder.cls_token is None
    assert isinstance(model.encoder.fc_norm, torch.nn.Identity)
    assert len(model.encoder.blocks) == 12
    for block in model.encoder.blocks:
        assert isinstance(block.ls1, torch.nn.Identity)
        assert block.cpe.groups == model.num_features
        mixer = block.attn.mixer
        assert not hasattr(mixer, "qk_conv")
        assert not hasattr(mixer.config, "rank_rotary")
        assert mixer.w_bc.out_features == 6 * 32 + 384
        assert block.attn.implementation == "reference"


def test_schedule_has_three_epoch_warmup_and_fixed_final_floor():
    assert learning_rate(299, 100, 30, 5e-4) == 5e-4
    assert learning_rate(300, 100, 30, 5e-4) == 5e-4
    assert learning_rate(2999, 100, 30, 5e-4) == 1e-6
