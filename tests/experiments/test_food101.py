"""Contracts that make the short vision ablations comparable."""
import torch

from experiments.food101 import VARIANTS, build_model, learning_rate


def test_variants_share_initialization_and_grid_but_change_requested_core():
    torch.set_num_threads(2)
    dynamic = build_model("dynamic")
    reference = dict(dynamic.named_parameters())
    assert dynamic.encoder.pos_embed.shape == (1, 14 * 14, 384)
    assert dynamic.encoder.no_embed_class
    assert len(dynamic.encoder.blocks) == 12
    for variant in VARIANTS[1:]:
        candidate = build_model(variant)
        for name, parameter in candidate.named_parameters():
            torch.testing.assert_close(parameter, reference[name], rtol=0, atol=0)
        for block in candidate.encoder.blocks:
            cfg = block.attn.mixer.config
            assert cfg.core_mode.value == variant
            assert block.attn.implementation == "reference"


def test_schedule_has_three_epoch_warmup_and_fixed_final_floor():
    assert learning_rate(299, 100, 30, 5e-4) == 5e-4
    assert learning_rate(300, 100, 30, 5e-4) == 5e-4
    assert learning_rate(2999, 100, 30, 5e-4) == 1e-6
