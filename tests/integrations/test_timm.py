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
    _TimmLSSOMixer,
    _require_timm_attention_mask_api,
    create_lsso_vit,
)
from lsso import CoreMode, LSSO
from lsso.ball import cuda


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
        create_lsso_vit(
            image_size=32,
            patch_size=4,
            num_classes=100,
            embed_dim=16,
            depth=0,
            num_heads=2,
            rank=4,
            mlp_ratio=2.0,
            core_mode=CoreMode.DYNAMIC,

            bias=True,
        )


def test_timm_adapter_forwards_the_requested_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TimmLSSOMixer(
        16,
        2,
        rank=4,
        core_mode=CoreMode.DYNAMIC,

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
        create_lsso_vit(
            image_size=32,
            patch_size=4,
            num_classes=10,
            embed_dim=16,
            depth=1,
            num_heads=2,
            rank=4,
            mlp_ratio=2.0,
            core_mode=CoreMode.DYNAMIC,

            bias=True,
            implementation="automatic",
        )


def test_timm_cuda_adapter_does_not_fall_back_to_reference() -> None:
    adapter = _TimmLSSOMixer(
        32,
        2,
        rank=16,
        core_mode=CoreMode.DYNAMIC,

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
        "core_mode": CoreMode.DYNAMIC,
        "bias": True,
        "drop_path_rate": 0.0,
    }
    fast = create_lsso_vit(**kwargs, implementation="cuda").cuda().eval()
    reference = create_lsso_vit(
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
