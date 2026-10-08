import torch
from experiments.food101 import build_model, learning_rate


def test_food_model_uses_current_qkv_operator_and_existing_grid():
    torch.set_num_threads(2)
    model = build_model()
    assert model.encoder.pos_embed.shape == (1, model.patch_embed.num_patches, model.num_features)
    assert model.encoder.cls_token is None
    assert isinstance(model.encoder.fc_norm, torch.nn.Identity)
    assert len(model.encoder.blocks) == 12
    for block in model.encoder.blocks:
        assert isinstance(block.ls1, torch.nn.Identity)
        assert not hasattr(block, 'cpe')
        mixer = block.attn.mixer
        assert mixer.qk_conv.kernel_size == (3, 3)
        assert mixer.get_extra_state()['position_encoding'] == 'external'
        assert mixer.w_qkv.out_features == 2 * 6 * 32 + 384
        assert block.attn.implementation == "reference"


def test_schedule_has_three_epoch_warmup_and_fixed_final_floor():
    assert learning_rate(299, 100, 30, 5e-4) == 5e-4
    assert learning_rate(300, 100, 30, 5e-4) == 5e-4
    assert learning_rate(2999, 100, 30, 5e-4) == 1e-6
