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
    create_memsolve_vit_variant,
    vision_default_mlp_ratio,
    vision_default_rank,
    vision_spec,
    vision_variant_kwargs,
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
    pytest.importorskip("timm")
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
@pytest.mark.parametrize("norm_type,rotary_2d", [("layernorm", False), ("rmsnorm", True)])
def test_timm_cuda_adapter_matches_reference_outputs_and_gradients(norm_type, rotary_2d) -> None:
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
    kwargs.update(norm_type=norm_type, rotary_2d=rotary_2d)
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
    pytest.importorskip("timm")
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


def test_rmsnorm_axial_rove_layout_and_checkpoint_contract():
    kwargs = dict(image_size=32, patch_size=4, num_classes=10, embed_dim=32,
                  depth=1, num_heads=2, rank=16, mlp_ratio=2., bias=True,
                  dynamic_img_size=True)
    model = create_memsolve_vit(**kwargs, norm_type="rmsnorm", rotary_2d=True)
    assert isinstance(model.encoder.norm, torch.nn.RMSNorm)
    x = torch.randn(2, 3, 32, 48)
    mask = torch.ones(2, 96, dtype=torch.bool)
    mask[:, -5:] = False
    layout = model._token_layout(x, mask)
    cos, sin = layout.rotary_factors
    assert cos.shape == (1, 96, 8)
    torch.testing.assert_close(cos.square() + sin.square(), torch.ones_like(cos))
    # Moving right leaves the row-axis rotations unchanged.
    torch.testing.assert_close(cos[:, 0, :4], cos[:, 1, :4])
    assert not torch.equal(cos[:, 0, 4:], cos[:, 1, 4:])
    output = model(x, valid_mask=mask)
    assert output.shape == (2, 10) and torch.isfinite(output).all()
    restored = create_memsolve_vit(**kwargs, norm_type="rmsnorm", rotary_2d=True)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x, valid_mask=mask), output)
    with pytest.raises(RuntimeError, match="incompatible"):
        create_memsolve_vit(**kwargs).load_state_dict(model.state_dict())


def test_gelu_scaffold_rejects_swiglu_checkpoint():
    model = create_memsolve_vit(image_size=32, patch_size=4, num_classes=10,
        embed_dim=32, depth=1, num_heads=2, rank=16, mlp_ratio=2.,
        bias=True, norm_type="rmsnorm", rotary_2d=True)
    state = copy.deepcopy(model.state_dict())
    state["_extra_state"]["ffn"] = "swiglu"
    with pytest.raises(RuntimeError, match="incompatible MemSolve vision scaffold"):
        model.load_state_dict(state)


def test_vision_variant_kwargs_and_factory_forward_all_scales_and_options() -> None:
    pytest.importorskip("timm")
    expected_scales = {
        "tiny": (192, 12, 6, 16, 4.0, 0.0),
        "small": (384, 12, 6, 32, 4.125, 0.1),
        "base": (768, 12, 12, 48, 4.0, 0.4),
        "large": (1024, 24, 16, 64, 4.0, 0.45),
    }
    for variant, (dim, depth, heads, rank, mlp_ratio, drop_path) in expected_scales.items():
        spec = vision_spec(variant)
        assert (spec.embed_dim, spec.depth, spec.num_heads, spec.drop_path_rate) == (
            dim,
            depth,
            heads,
            drop_path,
        )
        assert vision_default_rank(variant) == rank
        assert vision_default_mlp_ratio(variant) == mlp_ratio
        assert vision_variant_kwargs(variant) == {
            "embed_dim": dim,
            "depth": depth,
            "num_heads": heads,
            "rank": rank,
            "mlp_ratio": mlp_ratio,
            "drop_path_rate": drop_path,
        }

    large = create_memsolve_vit_variant("large", image_size=32, num_classes=7).eval()
    assert len(large.blocks) == 24
    assert large.encoder.embed_dim == 1024
    assert large.blocks[0].attn.mixer.config.num_heads == 16
    assert large.blocks[0].attn.mixer.config.rank == 64
    assert large.encoder.head.out_features == 7
    with torch.inference_mode():
        logits = large(torch.randn(1, 3, 32, 32))
    assert logits.shape == (1, 7) and torch.isfinite(logits).all()

    customized = create_memsolve_vit_variant(
        "tiny",
        image_size=32,
        patch_size=8,
        num_classes=5,
        rank=24,
        mlp_ratio=2.5,
        core_mode=CoreMode.COMPLEMENT,
        drop_path_rate=0.25,
        dynamic_img_size=True,
        norm_type="rmsnorm",
        rotary_2d=True,
    ).eval()
    assert isinstance(customized.encoder.norm, torch.nn.RMSNorm)
    assert customized.rotary_2d is True
    assert customized.blocks[0].attn.mixer.config.rank == 24
    assert customized.blocks[0].attn.mixer.config.core_mode == CoreMode.COMPLEMENT
    assert customized.blocks[0].mlp.fc1.out_features == int(192 * 2.5)
    with torch.inference_mode():
        out = customized(torch.randn(1, 3, 24, 32))
    assert out.shape == (1, 5) and torch.isfinite(out).all()


def test_timm_create_model_registers_memsolve_vit_large() -> None:
    timm = pytest.importorskip("timm")
    model = timm.create_model("memsolve_vit_large", pretrained=False)
    assert len(model.blocks) == 24
    assert model.encoder.embed_dim == 1024
    assert model.blocks[0].attn.mixer.config.num_heads == 16
    assert model.blocks[0].attn.mixer.config.rank == 64
    assert model.default_cfg["input_size"] == (3, 224, 224)

    compact_large = timm.create_model(
        "memsolve_vit_large",
        pretrained=False,
        img_size=32,
        num_classes=11,
        norm_type="rmsnorm",
        rotary_2d=True,
    ).eval()
    assert isinstance(compact_large.encoder.norm, torch.nn.RMSNorm)
    with torch.inference_mode():
        logits = compact_large(torch.randn(1, 3, 32, 32))
    assert logits.shape == (1, 11) and torch.isfinite(logits).all()

    with pytest.raises(ValueError, match="No pretrained MemSolve-ViT weights"):
        timm.create_model("memsolve_vit_large", pretrained=True)


def test_openmmlab_backbone_supports_large_variant() -> None:
    pytest.importorskip("timm")
    from integrations.openmmlab import MemSolveViTBackbone

    backbone = MemSolveViTBackbone(variant="large", implementation="reference").eval()
    assert backbone.out_channels == (1024, 1024, 1024, 1024)
    assert len(backbone.blocks) == 24
    assert backbone.interaction_indexes == [(0, 6), (6, 12), (12, 18), (18, 24)]
    assert backbone.interactions[0].injector.attn.n_heads == 16
    assert backbone._pretrained_geometry["embed_dim"] == 1024
    assert backbone._pretrained_geometry["rank"] == 64
    with torch.inference_mode():
        features = backbone(torch.randn(1, 3, 64, 64))
    assert tuple(f.shape for f in features) == (
        (1, 1024, 16, 16),
        (1, 1024, 8, 8),
        (1, 1024, 4, 4),
        (1, 1024, 2, 2),
    )
    assert all(torch.isfinite(f).all() for f in features)


def test_cpe_compile_preserves_static_channels_last_and_materializes_dynamic_shapes() -> None:
    pytest.importorskip("timm")
    model = create_memsolve_vit(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=32,
        depth=1,
        num_heads=2,
        rank=16,
        mlp_ratio=2.0,
        core_mode=CoreMode.EQUILIBRIUM,
        bias=True,
        dynamic_img_size=True,
        position_encoding="cpe",
        implementation="reference",
    ).eval()

    graphs: list[torch.fx.GraphModule] = []

    def capture_backend(gm: torch.fx.GraphModule, _example_inputs: list[torch.Tensor]):
        graphs.append(gm)
        return gm.forward

    def node_tensor(node: torch.fx.Node) -> torch.Tensor:
        return node.meta.get("val", node.meta["example_value"])

    def find_cpe_spatial_input(gm: torch.fx.GraphModule) -> torch.fx.Node:
        matches = [
            node.args[0]
            for node in gm.graph.nodes
            if (
                (node.op == "call_module" and str(node.target).endswith("cpe"))
                or (node.op == "call_function" and "conv2d" in str(node.target))
            )
            and isinstance(node.args[0], torch.fx.Node)
            and node_tensor(node.args[0]).shape[1] == 32
        ]
        assert len(matches) == 1
        return matches[0]

    torch._dynamo.reset()
    try:
        static_fn = torch.compile(model, backend=capture_backend, fullgraph=True, dynamic=False)
        static_img = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(static_fn(static_img), model(static_img))
        static_spatial = find_cpe_spatial_input(graphs[-1])
        assert static_spatial.target == "permute"
        static_val = node_tensor(static_spatial)
        assert static_val.is_contiguous(memory_format=torch.channels_last)
        assert not static_val.is_contiguous()

        graphs.clear()
        dynamic_fn = torch.compile(model, backend=capture_backend, fullgraph=True, dynamic=True)
        dynamic_img = torch.randn(2, 3, 32, 48)
        with torch.no_grad():
            torch.testing.assert_close(dynamic_fn(dynamic_img), model(dynamic_img))
        dynamic_spatial = find_cpe_spatial_input(graphs[-1])
        assert dynamic_spatial.target == "contiguous"
        assert node_tensor(dynamic_spatial).is_contiguous()
    finally:
        torch._dynamo.reset()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_cpe_vit_matches_eager_static_and_dynamic_cuda() -> None:
    pytest.importorskip("timm")
    try:
        cuda.load()
    except RuntimeError as error:
        pytest.skip(f"native CUDA extension is unavailable: {error}")

    torch.manual_seed(71)
    eager = create_memsolve_vit(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=32,
        depth=2,
        num_heads=2,
        rank=16,
        mlp_ratio=2.0,
        core_mode=CoreMode.EQUILIBRIUM,
        bias=True,
        drop_path_rate=0.0,
        dynamic_img_size=True,
        position_encoding="cpe",
        implementation="cuda",
    ).cuda().eval()

    torch._dynamo.reset()
    try:
        compiled_static = torch.compile(eager, dynamic=False)
        x_eager = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
        x_comp = x_eager.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_eager = eager(x_eager)
            out_comp = compiled_static(x_comp)
        upstream = torch.randn_like(out_eager)
        grads_eager = torch.autograd.grad((out_eager * upstream).sum(), (x_eager, *eager.parameters()))
        grads_comp = torch.autograd.grad((out_comp * upstream).sum(), (x_comp, *eager.parameters()))
        assert _relative_l2(out_comp, out_eager) <= 5e-3
        for grad_comp, grad_eager in zip(grads_comp, grads_eager):
            assert torch.isfinite(grad_comp).all()
            assert _relative_l2(grad_comp, grad_eager) <= 1e-2

        torch._dynamo.reset()
        compiled_dynamic = torch.compile(eager, dynamic=True)
        for height, width in ((32, 32), (32, 48)):
            dx_eager = torch.randn(2, 3, height, width, device="cuda", requires_grad=True)
            dx_comp = dx_eager.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dout_eager = eager(dx_eager)
                dout_comp = compiled_dynamic(dx_comp)
            dupstream = torch.randn_like(dout_eager)
            dgrads_eager = torch.autograd.grad(
                (dout_eager * dupstream).sum(), (dx_eager, *eager.parameters())
            )
            dgrads_comp = torch.autograd.grad(
                (dout_comp * dupstream).sum(), (dx_comp, *eager.parameters())
            )
            assert _relative_l2(dout_comp, dout_eager) <= 5e-3
            for grad_comp, grad_eager in zip(dgrads_comp, dgrads_eager):
                assert torch.isfinite(grad_comp).all()
                assert _relative_l2(grad_comp, grad_eager) <= 1e-2
    finally:
        torch._dynamo.reset()
