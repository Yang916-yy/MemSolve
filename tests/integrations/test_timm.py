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
from memsolve import MemSolve
from memsolve.ball import cuda


pytestmark = pytest.mark.integration


@pytest.mark.parametrize("variant,width,heads,rank,count", [
    ("tiny", 192, 6, 16, 5_427_112),
    ("small", 384, 6, 32, 20_622_184),
    ("base", 768, 12, 48, 83_940_328),
])
def test_registered_vision_models(variant, width, heads, rank, count):
    timm = pytest.importorskip("timm")
    name = f"memsolve_vit_{variant}"
    assert name in timm.list_models("memsolve_vit_*")
    assert name not in timm.list_models("memsolve_vit_*", pretrained=True)
    model = timm.create_model(name, img_size=32).eval()
    assert len(model.blocks) == 12
    assert model.num_features == width and model.num_classes == 1000
    assert sum(p.numel() for p in model.parameters()) == count
    for block in model.blocks:
        config = block.attn.mixer.config
        assert (config.num_heads, config.rank, config.qk_conv_dim) == (heads, rank, 2)
        assert config.output_gate_rank == 32
    assert model.pretrained_cfg["architecture"] == name
    assert model.get_classifier() is model.encoder.head
    model.reset_classifier(7)
    assert model.num_classes == 7
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 7) and torch.isfinite(output).all()


def test_registered_vision_checkpoint_and_pooling_contract(tmp_path):
    timm = pytest.importorskip("timm")
    kwargs = dict(img_size=32, embed_dim=32, depth=1, num_heads=2, num_classes=7)
    model = timm.create_model("memsolve_vit_tiny", **kwargs).eval()
    images = torch.randn(2, 3, 32, 32)
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    restored = timm.create_model("memsolve_vit_tiny", checkpoint_path=checkpoint, **kwargs).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(images), model(images), rtol=0, atol=0)
    restored.reset_classifier(0)
    assert restored(images).shape == (2, 32)
    with pytest.raises(ValueError, match="mean pooling"):
        restored.reset_classifier(7, global_pool="token")
    with pytest.raises(ValueError, match="mean pooling"):
        timm.create_model("memsolve_vit_tiny", global_pool="token")
    with pytest.raises(ValueError, match="No pretrained"):
        timm.create_model("memsolve_vit_tiny", pretrained=True)


@pytest.mark.parametrize("factory", [create_memsolve_vit])
@pytest.mark.parametrize("kernel_size,gate_rank", [(3, 32), (5, 16)])
def test_vision_initialization_preserves_key_fan_in_and_timm_qv_rules(factory, kernel_size, gate_rank):
    pytest.importorskip("timm")
    torch.manual_seed(41)
    model = factory(
        image_size=32, patch_size=4, num_classes=10, embed_dim=192,
        depth=2, num_heads=6, rank=16, mlp_ratio=2, bias=True,
        qk_conv_kernel_size=kernel_size, output_gate_rank=gate_rank,
    )
    encoder = model.encoder if hasattr(model, "encoder") else model
    for reinitialize in (False, True):
        if reinitialize:
            encoder.init_weights()
        layers = [module for module in model.modules() if isinstance(module, MemSolve)]
        assert len(layers) == 2
        for layer in layers:
            q, k, v = layer.w_qkv.weight.split((96, 96, 192))
            assert abs(k.std().item() * 192**0.5 - 1) < 0.05
            for weight in (q, v):
                assert abs(weight.std().item() / 0.02 - 1) < 0.05
            assert torch.count_nonzero(layer.w_qkv.bias) == 0
            assert layer.config.qk_conv_dim == 2
            assert layer.config.output_gate_rank == gate_rank
            assert layer.gate_down.weight.shape == (gate_rank, 192)
            assert torch.count_nonzero(layer.gate_up.bias) == 0
            for weight in (layer.gate_down.weight, layer.gate_up.weight):
                assert abs(weight.std().item() / .02 - 1) < .05
            expected = torch.zeros_like(layer.qk_conv.weight)
            expected[:, 0, kernel_size // 2, kernel_size // 2] = 1
            torch.testing.assert_close(layer.qk_conv.weight, expected, rtol=0, atol=0)
        for block in model.blocks:
            for weight in block.mlp.fc1.weight.chunk(2):
                assert abs(weight.std().item() / 0.02 - 1) < 0.05
            assert torch.count_nonzero(block.mlp.fc1.bias) == 0


def test_small_swiglu_width_and_packed_forward_backward():
    pytest.importorskip("timm")
    torch.manual_seed(71)
    model = create_memsolve_vit(
        image_size=32, patch_size=16, num_classes=10, embed_dim=384,
        depth=1, num_heads=6, rank=32, mlp_ratio=4.0, bias=True,
    )
    mlp = model.blocks[0].mlp
    assert mlp.fc1.out_features == 2048
    assert mlp.fc2.in_features == 1024
    x = torch.randn(2, 4, 384, requires_grad=True)
    wg, wv = mlp.fc1.weight.chunk(2)
    bg, bv = mlp.fc1.bias.chunk(2)
    linear = torch.nn.functional.linear
    expected = linear(torch.nn.functional.silu(linear(x, wg, bg)) * linear(x, wv, bv),
                      mlp.fc2.weight, mlp.fc2.bias)
    actual = mlp(x)
    torch.testing.assert_close(actual, expected)
    inputs = (x, *mlp.parameters())
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)
    old_state = model.get_extra_state() | {"version": 1}
    with pytest.raises(RuntimeError, match="incompatible"):
        model.set_extra_state(old_state)


def test_constant_droppath_and_resolution_weight_transfer():
    kwargs = dict(patch_size=16, num_classes=10, embed_dim=192, depth=2,
                  num_heads=6, rank=16, mlp_ratio=4, bias=True,
                  drop_path_rate=.1, drop_path_schedule='constant')
    source = create_memsolve_vit(image_size=192, **kwargs)
    target = create_memsolve_vit(image_size=224, **kwargs)
    for model in (source, target):
        for block in model.blocks:
            assert block.drop_path1.drop_prob == block.drop_path2.drop_prob == .1
    target.load_state_dict(source.state_dict(), strict=True)
    target.eval()
    with torch.no_grad():
        out = target(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 10) and torch.isfinite(out).all()


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
            bias=True,
        )


def test_timm_adapter_forwards_the_requested_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _TimmMemSolveMixer(
        16,
        2,
        rank=4,
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
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        captured["valid_mask"] = valid_mask
        captured["implementation"] = implementation
        captured["spatial_shape"] = spatial_shape
        return x

    monkeypatch.setattr(adapter.mixer, "forward", fake_forward)
    from integrations.timm import VisionTokenLayout
    x = torch.randn(2, 6, 16)
    torch.testing.assert_close(adapter(x, attn_mask=VisionTokenLayout(None, (2, 3))), x)
    assert captured["valid_mask"] is None
    assert captured["implementation"] == "cuda"
    assert captured["spatial_shape"] == (2, 3)


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
            bias=True,
            implementation="automatic",
        )


def test_timm_cuda_adapter_does_not_fall_back_to_reference() -> None:
    adapter = _TimmMemSolveMixer(
        32,
        2,
        rank=16,
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


def test_shared_scaffold_applies_linear_drop_path_without_layerscale():
    pytest.importorskip("timm")
    from integrations.timm import create_memsolve_vit

    model = create_memsolve_vit(
        image_size=32, patch_size=16, num_classes=10, embed_dim=32,
        depth=3, num_heads=2, rank=16, mlp_ratio=2,
        bias=True, drop_path_rate=0.4,
    )
    actual = [getattr(block.drop_path1, "drop_prob", 0.0) for block in model.blocks]
    assert actual == pytest.approx([0.0, 0.2, 0.4])
    assert all(isinstance(block.ls1, torch.nn.Identity) and isinstance(block.ls2, torch.nn.Identity) for block in model.blocks)


def test_rope_scaffold_has_no_cpe_and_masks_local_qk_input():
    pytest.importorskip("timm")
    from integrations.timm import create_memsolve_vit, VisionTokenLayout
    torch.manual_seed(137)
    model = create_memsolve_vit(
        image_size=32, patch_size=16, num_classes=10, embed_dim=32,
        depth=1, num_heads=2, rank=16, mlp_ratio=2,
        bias=True,
    )
    assert model.encoder.pos_embed is None
    block = model.blocks[0]
    assert not hasattr(block, 'cpe')
    assert block.attn.mixer.get_extra_state()['position_encoding'] == 'rope_2d_axial_theta100_after_qk_conv'
    with torch.no_grad():
        block.attn.mixer.qk_conv.weight.add_(torch.randn_like(block.attn.mixer.qk_conv.weight) * .1)
    # Observe exactly the tensor entering the first normalization.
    seen = []
    handle = block.norm1.register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0]))
    x = torch.randn(2, 4, 32, requires_grad=True)
    block(x)
    torch.testing.assert_close(seen[-1], x)
    seen.clear()
    valid = torch.ones(2, 4, dtype=torch.bool);valid[:, 2] = False
    first = block(x, attn_mask=VisionTokenLayout(valid))
    altered = x.detach().clone();altered[:, 2] = 1000
    second = block(altered, attn_mask=VisionTokenLayout(valid))
    torch.testing.assert_close(first[valid], second[valid])
    first[valid].square().sum().backward()
    assert block.attn.mixer.qk_conv.weight.grad is not None
    assert torch.isfinite(block.attn.mixer.qk_conv.weight.grad).all()
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


def test_mean_pooling_uses_token_norm_and_excludes_invalid_patches():
    pytest.importorskip("timm")
    torch.manual_seed(211)
    model = create_memsolve_vit(
        image_size=32, patch_size=16, num_classes=10, embed_dim=32,
        depth=2, num_heads=2, rank=16, mlp_ratio=2, bias=True,
    ).eval()
    encoder = model.encoder
    assert encoder.cls_token is None and encoder.num_prefix_tokens == 0
    assert encoder.pos_embed is None
    assert isinstance(encoder.norm, torch.nn.LayerNorm)
    assert isinstance(encoder.fc_norm, torch.nn.Identity)
    images = torch.randn(2, 3, 32, 32)
    features = encoder.forward_features(images)
    assert features.shape == (2, 4, 32)
    torch.testing.assert_close(model(images), encoder.head(features.mean(1)))
    # The top-right patch is excluded from both memory and classification.
    mask = torch.tensor([[True, False, True, True], [False, False, False, False]])
    changed = images.clone(); changed[:, :, :16, 16:] = 10000
    first, second = model(images, valid_mask=mask), model(changed, valid_mask=mask)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[1], encoder.head.bias)
    with pytest.raises(ValueError, match="without CLS"):
        model(images, valid_mask=torch.ones(2, 5, dtype=torch.bool))
    state = model.state_dict()
    state["_extra_state"] = {"architecture": "cls_layerscale"}
    with pytest.raises(RuntimeError, match="vision scaffold"):
        model.load_state_dict(state)


def test_rectangular_images_forward_actual_grid_to_qk_convolution():
    model = create_memsolve_vit(
        image_size=32, patch_size=16, num_classes=10, embed_dim=32,
        depth=1, num_heads=2, rank=16, mlp_ratio=2, bias=True,
        dynamic_img_size=True,
    ).eval()
    core = model.blocks[0].attn.mixer
    with torch.no_grad():
        core.qk_conv.weight.add_(torch.randn_like(core.qk_conv.weight) * .1)
    layouts = []
    handle = core.register_forward_pre_hook(
        lambda _module, _args, kwargs: layouts.append(kwargs['spatial_shape']), with_kwargs=True,
    )
    try:
        for height, width in ((32, 48), (48, 32)):
            images = torch.randn(2, 3, height, width, requires_grad=True)
            logits = model(images)
            assert logits.shape == (2, 10)
            logits.square().sum().backward()
            assert torch.isfinite(images.grad).all()
        assert layouts == [(2, 3), (3, 2)]
        assert core.qk_conv.weight.grad.abs().sum() > 0
    finally:
        handle.remove()
