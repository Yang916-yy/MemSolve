from __future__ import annotations

import copy
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 is supported by the package.
    import tomli as tomllib
from pathlib import Path

import pytest
import torch

from integrations.timm import (
    _TimmMemSolveMixer,
    _require_timm_attention_mask_api,
    create_memsolve_vit,
)
from memsolve import CoreMode, MemSolve
from memsolve.ball import cuda


pytestmark = pytest.mark.integration


def test_vision_extra_requires_the_attn_mask_capable_timm_release() -> None:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    assert "timm>=1.0.16,<2" in project["project"]["optional-dependencies"]["vision"]


def test_timm_adapter_rejects_a_vision_transformer_without_token_layout_api() -> None:
    class LegacyVisionTransformer(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def forward_intermediates(self, x: torch.Tensor) -> list[torch.Tensor]:
            return [x]

    with pytest.raises(RuntimeError, match="timm>=1.0.16"):
        _require_timm_attention_mask_api(LegacyVisionTransformer)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual64 = actual.detach().to(dtype=torch.float64)
    expected64 = expected.detach().to(dtype=torch.float64)
    return float(
        torch.linalg.vector_norm(actual64 - expected64)
        / torch.linalg.vector_norm(expected64).clamp_min(1e-12)
    )


def test_timm_factory_rejects_empty_depth_before_framework_import() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        create_memsolve_vit(
            image_size=32,
            patch_size=4,
            num_classes=100,
            embed_dim=16,
            depth=0,
            num_heads=2,
            rank=4,
            mlp_ratio=2.0,
            core_mode=CoreMode.EQUILIBRIUM,

            bias=True,
        )


def test_timm_adapter_forwards_the_requested_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TimmMemSolveMixer(
        16,
        2,
        rank=4,
        core_mode=CoreMode.EQUILIBRIUM,

        implementation="cuda",
        qkv_bias=True,
        qk_norm=False,
        scale_norm=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=None,
    )
    captured: dict[str, object] = {}

    def fake_forward(
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        implementation: str,
    ) -> torch.Tensor:
        captured["valid_mask"] = valid_mask
        captured["implementation"] = implementation
        return x

    monkeypatch.setattr(adapter.mixer, "forward", fake_forward)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(adapter(x), x)
    assert captured["valid_mask"] is None
    assert captured["implementation"] == "cuda"


def test_timm_factory_rejects_unknown_implementation_before_framework_import() -> None:
    with pytest.raises(ValueError, match="implementation"):
        create_memsolve_vit(
            image_size=32,
            patch_size=4,
            num_classes=10,
            embed_dim=16,
            depth=1,
            num_heads=2,
            rank=4,
            mlp_ratio=2.0,
            core_mode=CoreMode.EQUILIBRIUM,

            bias=True,
            implementation="automatic",
        )


def test_timm_cuda_adapter_does_not_fall_back_to_reference() -> None:
    adapter = _TimmMemSolveMixer(
        32,
        2,
        rank=16,
        core_mode=CoreMode.EQUILIBRIUM,

        implementation="cuda",
        qkv_bias=False,
        qk_norm=False,
        scale_norm=False,
        proj_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=None,
    )
    with pytest.raises(ValueError, match="requires x to be a CUDA tensor"):
        adapter(torch.randn(1, 5, 32))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_timm_cuda_adapter_matches_reference_outputs_and_gradients() -> None:
    pytest.importorskip("timm")
    try:
        cuda.load()
    except RuntimeError as error:
        pytest.skip(f"native CUDA extension is unavailable: {error}")

    torch.manual_seed(53)
    kwargs = {
        "image_size": 32,
        "patch_size": 4,
        "num_classes": 10,
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 2,
        "rank": 16,
        "mlp_ratio": 2.0,
        "core_mode": CoreMode.EQUILIBRIUM,
        "bias": True,
        "drop_path_rate": 0.0,
    }
    fast = create_memsolve_vit(**kwargs, implementation="cuda").cuda().eval()
    reference = create_memsolve_vit(
        **kwargs,
        implementation="reference",
    ).cuda().eval()
    reference.load_state_dict(copy.deepcopy(fast.state_dict()))

    fast_x = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
    reference_x = fast_x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fast_output = fast(fast_x)
        reference_output = reference(reference_x)
    upstream = torch.randn_like(fast_output)
    fast_gradients = torch.autograd.grad(
        (fast_output * upstream).sum(),
        (fast_x, *fast.parameters()),
    )
    reference_gradients = torch.autograd.grad(
        (reference_output * upstream).sum(),
        (reference_x, *reference.parameters()),
    )

    assert _relative_l2(fast_output, reference_output) <= 5e-3
    for fast_gradient, reference_gradient in zip(
        fast_gradients,
        reference_gradients,
    ):
        assert torch.isfinite(fast_gradient).all()
        assert _relative_l2(fast_gradient, reference_gradient) <= 1e-2


@pytest.mark.parametrize("schedule,expected", [("linear", [0.0, 0.2, 0.4]), ("constant", [0.4] * 3)])
def test_shared_scaffold_applies_the_selected_drop_path_schedule(schedule, expected):
    pytest.importorskip("timm")
    from integrations.timm import create_memsolve_vit

    model = create_memsolve_vit(
        image_size=32, patch_size=16, num_classes=10, embed_dim=32,
        depth=3, num_heads=2, rank=16, mlp_ratio=2, core_mode="equilibrium",
        bias=True, drop_path_rate=0.4, drop_path_schedule=schedule,
    )
    actual = [getattr(block.drop_path1, "drop_prob", 0.0) for block in model.blocks]
    assert actual == pytest.approx(expected)


def test_cpe_replaces_absolute_positions_and_matches_patch_grid_convolution():
    pytest.importorskip("timm")
    from integrations.timm import create_memsolve_vit, VisionTokenLayout
    torch.manual_seed(137)
    model = create_memsolve_vit(
        image_size=32, patch_size=16, num_classes=10, embed_dim=32,
        depth=1, num_heads=2, rank=16, mlp_ratio=2, core_mode="equilibrium",
        bias=True, position_encoding="cpe",
    )
    assert model.encoder.pos_embed is None
    block = model.blocks[0]
    assert block.cpe.groups == 32 and block.cpe.kernel_size == (3, 3)
    # Observe exactly the tensor entering the first normalization.
    seen = []
    handle = block.norm1.register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0]))
    x = torch.randn(2, 4, 32, requires_grad=True)
    block(x)
    grid = x.reshape(2, 2, 2, 32).permute(0, 3, 1, 2).contiguous()
    expected = x + torch.nn.functional.conv2d(
        grid, block.cpe.weight, block.cpe.bias, padding=1, groups=32,
    ).flatten(2).transpose(1, 2)
    torch.testing.assert_close(seen[-1], expected)
    seen.clear()
    valid = torch.ones(2, 4, dtype=torch.bool);valid[:, 2] = False
    first = block(x, attn_mask=VisionTokenLayout(valid))
    altered = x.detach().clone();altered[:, 2] = 1000
    second = block(altered, attn_mask=VisionTokenLayout(valid))
    torch.testing.assert_close(first[valid], second[valid])
    first[valid].square().sum().backward()
    assert block.cpe.weight.grad is not None
    assert torch.isfinite(block.cpe.weight.grad).all()
    assert torch.count_nonzero(x.grad[:, 2]) == 0
    handle.remove()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_adamw_matches_unfused_fp32_updates():
    torch.manual_seed(77)
    first = torch.nn.Linear(17, 7, device='cuda')
    second = copy.deepcopy(first)
    fused = torch.optim.AdamW(first.parameters(), lr=.001, weight_decay=.05, fused=True)
    unfused = torch.optim.AdamW(second.parameters(), lr=.001, weight_decay=.05, fused=False, foreach=False)
    for _ in range(3):
        for a, b in zip(first.parameters(), second.parameters()):
            a.grad = torch.randn_like(a); b.grad = a.grad.clone()
        fused.step(); unfused.step()
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


def test_vit3_mean_readout_excludes_padding_and_has_no_layerscale():
    from integrations.timm import VisionTokenLayout
    torch.manual_seed(159)
    model = create_memsolve_vit(
        image_size=32, patch_size=16, num_classes=3, embed_dim=32,
        depth=2, num_heads=2, rank=16, mlp_ratio=2, bias=True,
    ).eval()
    assert model.encoder.cls_token is None
    assert model.encoder.num_prefix_tokens == 0
    assert isinstance(model.encoder.norm, torch.nn.LayerNorm)
    assert isinstance(model.encoder.fc_norm, torch.nn.Identity)
    assert all(isinstance(block.ls1, torch.nn.Identity) and
               isinstance(block.ls2, torch.nn.Identity) for block in model.blocks)
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    features = model.encoder.forward_features(images)
    torch.testing.assert_close(model(images), model.encoder.head(features.mean(1)))
    valid = torch.tensor([[True, True, True, False], [False, False, False, False]])
    masked_features = model.encoder.forward_features(images, attn_mask=VisionTokenLayout(valid))
    expected = model.encoder.head(torch.stack((masked_features[0, :3].mean(0),
                                               torch.zeros_like(masked_features[1, 0]))))
    actual = model(images, valid_mask=valid)
    torch.testing.assert_close(actual, expected)
    altered = images.detach().clone()
    altered[0, :, 16:, 16:] = 1000
    altered[1] = 1000
    torch.testing.assert_close(model(altered, valid_mask=valid), actual)
    actual.sum().backward()
    assert torch.count_nonzero(images.grad[0, :, 16:, 16:]) == 0
    assert torch.count_nonzero(images.grad[1]) == 0
    state = copy.deepcopy(model.state_dict())
    state['_extra_state'].update(version=6, pooling='cls', class_token=True, layer_scale=True)
    with pytest.raises(RuntimeError, match='vision scaffold'):
        model.load_state_dict(state)
