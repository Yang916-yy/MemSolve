from copy import deepcopy

import pytest
import torch

from memsolve import MemSolve, MemSolveConfig

pytestmark = pytest.mark.core


@pytest.mark.parametrize("dim,bias", [(128, False), (384, True)])
def test_key_fan_in_initialization_and_isolated_reinitialization(dim, bias):
    torch.manual_seed(42)
    layer = MemSolve(MemSolveConfig(dim, 4, 16, bias=bias))
    offset = 64
    key = layer.w_qkv.weight[offset : 2 * offset]
    assert abs(key.std().item() * dim**0.5 - 1) < 0.05
    if bias:
        assert torch.count_nonzero(layer.w_qkv.bias[offset : 2 * offset]) == 0

    # Check the statistic the initialization is intended to control.
    x = torch.nn.functional.layer_norm(torch.randn(1024, dim), (dim,))
    projected_keys = x @ key.mT
    assert abs(projected_keys.square().mean().item() - 1) < 0.05

    before = deepcopy(layer.state_dict())
    layer.init_weights()
    assert not torch.equal(key, before["w_qkv.weight"][offset : 2 * offset])
    for name, parameter in layer.named_parameters():
        if name in ("w_qkv.weight", "w_qkv.bias"):
            torch.testing.assert_close(parameter[:offset], before[name][:offset], rtol=0, atol=0)
            torch.testing.assert_close(parameter[2 * offset:], before[name][2 * offset:], rtol=0, atol=0)
        else:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_forward_backward_and_dtype(dtype):
    torch.manual_seed(16)
    layer = MemSolve(MemSolveConfig(12, 2, 4, bias=True))
    if dtype == torch.float64:
        layer = layer.double()
    x = torch.randn(2, 7, 12, dtype=dtype, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape and y.dtype == dtype
    (y.double() * torch.randn_like(y).double()).sum().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_only_independent_qkv_static_core_and_output_parameters():
    layer = MemSolve(MemSolveConfig(12, 2, 4))
    assert set(dict(layer.named_parameters())) == {
        "w_qkv.weight",
        "qk_conv.weight",
        "core_delta",
        "head_norm_weight",
        "gate_down.weight",
        "gate_up.weight",
        "gate_up.bias",
        "w_o.weight",
    }
    assert layer.w_qkv.out_features == 2 * 2 * 4 + 12
    torch.testing.assert_close(layer.head_norm_weight, torch.ones(6))
    x = torch.randn(2, 7, 12, requires_grad=True)
    layer(x).square().sum().backward()
    q, k, v = layer.w_qkv.weight.grad.split((8, 8, 12))
    assert all(g.norm() > 0 for g in (q, k, v))
    assert not torch.equal(q, k)


@pytest.mark.parametrize("delta_scale", [0.0, 0.3])
@pytest.mark.parametrize("conv_dim", [1, 2])
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_identity_plus_delta_matches_explicit_map_and_all_gradients(delta_scale, conv_dim, kernel_size):
    from memsolve.ball.reference import ridge_query_readout, head_rms_norm, split_qkv

    torch.manual_seed(11)
    layer = MemSolve(MemSolveConfig(12, 2, 4, bias=True, qk_conv_dim=conv_dim,
                                qk_conv_kernel_size=kernel_size, output_gate_rank=3)).double()
    assert torch.count_nonzero(layer.core_delta) == 0
    with torch.no_grad():
        layer.core_delta.normal_(std=delta_scale)
        layer.qk_conv.weight.add_(torch.randn_like(layer.qk_conv.weight) * .2)
    x = torch.randn(2, 6, 12, dtype=torch.float64, requires_grad=True)
    actual = layer(x, spatial_shape=(2, 3) if conv_dim == 2 else None)
    explicit_map = (torch.eye(4).double() + layer.core_delta.detach()).requires_grad_()
    projected = torch.nn.functional.linear(x, layer.w_qkv.weight, layer.w_qkv.bias)
    qk, v = projected.split((16, 12), dim=-1)
    if conv_dim == 1:
        qk = torch.nn.functional.conv1d(qk.transpose(1, 2), layer.qk_conv.weight,
                                      padding=kernel_size // 2, groups=16).transpose(1, 2)
    else:
        qk = torch.nn.functional.conv2d(
            qk.transpose(1, 2).reshape(2, 16, 2, 3), layer.qk_conv.weight,
            padding=kernel_size // 2, groups=16,
        ).flatten(2).transpose(1, 2)
        # Independent complex-pair RoPE oracle after the convolution. With
        # rank=4 there is one unit-frequency pair for x and one for y.
        coordinates = torch.tensor([[col, row] for row in range(2) for col in range(3)], dtype=x.dtype)
        pairs = torch.view_as_complex(qk.reshape(2, 6, 4, 2, 2).contiguous())
        phases = torch.polar(torch.ones_like(coordinates), coordinates)[None, :, None, :]
        qk = torch.view_as_real(pairs * phases).reshape_as(qk)
    projected = torch.cat((qk, v), dim=-1)
    q, k, v = split_qkv(projected, 2, 4)
    raw = ridge_query_readout(q, k, v, explicit_map).transpose(1, 2)
    gate = torch.sigmoid(
        (x @ layer.gate_down.weight.T) @ layer.gate_up.weight.T + layer.gate_up.bias
    )
    expected = torch.nn.functional.linear(
        head_rms_norm(raw, layer.head_norm_weight).reshape_as(x) * gate,
        layer.w_o.weight, layer.w_o.bias,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    cotangent = torch.randn_like(actual)
    actual_inputs = (x, *layer.parameters())
    expected_inputs = tuple(explicit_map if p is layer.core_delta else p for p in actual_inputs)
    actual_grads = torch.autograd.grad(actual, actual_inputs, cotangent)
    expected_grads = torch.autograd.grad(expected, expected_inputs, cotangent)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_core_delta_standard_adamw_resume_and_unconstrained_scale():
    layer = MemSolve(MemSolveConfig(12, 2, 4)).double()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.03, weight_decay=0.05)
    identity = torch.eye(4).double().expand(2, 4, 4)

    def update(model, opt):
        opt.zero_grad(set_to_none=True)
        # A radial direction is now learnable, without projection or retraction.
        loss = (model.core_delta - identity).square().sum()
        loss.backward()
        opt.step()

    for _ in range(5):
        update(layer, optimizer)
    assert torch.all((identity + layer.core_delta).norm(dim=(-2, -1)) > 2)
    resumed = deepcopy(layer)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.03, weight_decay=0.05)
    resumed_opt.load_state_dict(deepcopy(optimizer.state_dict()))
    for _ in range(3):
        update(layer, optimizer)
        update(resumed, resumed_opt)
    torch.testing.assert_close(layer.core_delta, resumed.core_delta, rtol=0, atol=0)


@pytest.mark.parametrize("delta_scale", [0.0, 0.3])
def test_adamw_weight_decay_pulls_effective_core_toward_identity(delta_scale):
    torch.manual_seed(19)
    layer = MemSolve(MemSolveConfig(12, 2, 4)).double()
    with torch.no_grad():
        layer.core_delta.normal_(std=delta_scale)
    before = layer.core_delta.detach().clone()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.1, weight_decay=0.2)
    layer.core_delta.grad = torch.zeros_like(layer.core_delta)
    optimizer.step()
    # With zero gradient and fresh moments, only decoupled decay acts.
    torch.testing.assert_close(layer.core_delta, 0.98 * before, rtol=0, atol=0)
    identity = torch.eye(4).double()
    torch.testing.assert_close(
        identity + layer.core_delta, identity + 0.98 * before, rtol=0, atol=0,
    )


def test_trailing_nan_padding_matches_cropped_in_outputs_and_gradients():
    torch.manual_seed(4)
    layer = MemSolve(MemSolveConfig(12, 2, 4, bias=True)).double()
    with torch.no_grad():
        layer.qk_conv.weight.add_(torch.randn_like(layer.qk_conv.weight) * .2)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    x = torch.randn(1, 7, 12, dtype=torch.float64)
    x[~mask] = float("nan")
    x.requires_grad_()
    cropped = x.detach()[:, mask[0]].clone().requires_grad_()
    actual, expected = layer(x, mask), layer(cropped)
    torch.testing.assert_close(actual[:, mask[0]], expected)
    assert torch.count_nonzero(actual[~mask]) == 0
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad[:, mask[0]], cropped.grad)
    assert torch.count_nonzero(x.grad[~mask]) == 0


def test_empty_mask_is_zero_even_with_projection_bias():
    layer = MemSolve(MemSolveConfig(12, 2, 4, bias=True))
    x = torch.full((2, 7, 12), float("nan"), requires_grad=True)
    y = layer(x, torch.zeros(2, 7, dtype=torch.bool))
    assert torch.count_nonzero(y) == 0
    y.sum().backward()
    assert torch.count_nonzero(x.grad) == 0
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize("conv_dim", [1, 2])
def test_qk_convolution_identity_and_masked_neighbors(conv_dim):
    from memsolve.ball.reference import convolve_qk

    torch.manual_seed(62)
    layer = MemSolve(MemSolveConfig(12, 2, 4, bias=True, qk_conv_dim=conv_dim)).double()
    shape = (2, 3) if conv_dim == 2 else None
    packed = torch.randn(2, 6, 28, dtype=torch.float64)
    torch.testing.assert_close(convolve_qk(packed, layer.qk_conv.weight, shape),
                               packed, rtol=0, atol=0)
    with torch.no_grad():
        layer.qk_conv.weight.add_(torch.randn_like(layer.qk_conv.weight) * .2)
    filtered = convolve_qk(packed, layer.qk_conv.weight, shape)
    torch.testing.assert_close(filtered[..., 16:], packed[..., 16:], rtol=0, atol=0)
    mask = torch.tensor([[1, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    x = torch.randn(2, 6, 12, dtype=torch.float64, requires_grad=True)
    altered = x.detach().clone()
    altered[~mask] = float("nan")
    altered.requires_grad_()
    first = layer(x, mask, spatial_shape=shape)
    second = layer(altered, mask, spatial_shape=shape)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.count_nonzero(second[~mask]) == 0
    inputs = (x, *layer.parameters())
    altered_inputs = (altered, *layer.parameters())
    for a, b in zip(torch.autograd.grad(first.sum(), inputs),
                    torch.autograd.grad(second.sum(), altered_inputs)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_qk_convolution_requires_explicit_valid_spatial_layout():
    x = torch.randn(1, 6, 12)
    spatial = MemSolve(MemSolveConfig(12, 2, 4, qk_conv_dim=2))
    for shape in (None, (2, 2), (0, 6), (True, 6), (2.0, 3), [2, 3]):
        with pytest.raises(ValueError, match="spatial_shape"):
            spatial(x, spatial_shape=shape)
    with pytest.raises(ValueError, match="spatial_shape"):
        MemSolve(MemSolveConfig(12, 2, 4))(x, spatial_shape=(2, 3))
    with pytest.raises(RuntimeError, match="contract"):
        MemSolve(MemSolveConfig(12, 2, 4)).load_state_dict(spatial.state_dict())


def test_rope_cache_tracks_both_grid_axes_dtype_and_inference_to_training():
    model = MemSolve(MemSolveConfig(12, 2, 4, qk_conv_dim=2))
    x = torch.randn(1, 6, 12)
    with torch.inference_mode():
        model(x, spatial_shape=(2, 3))
    cos, sin = model._rotary_tables(x, (2, 3))
    assert cos.dtype == sin.dtype == torch.float32
    assert not torch.is_inference(cos)
    assert not any('rope' in key for key in model.state_dict())
    model(x.requires_grad_(), spatial_shape=(2, 3)).sum().backward()
    assert model._rotary_tables(x, (2, 3))[0] is cos
    assert torch.isfinite(x.grad).all()
    model(x, spatial_shape=(3, 2))
    assert not torch.equal(model._rotary_tables(x, (3, 2))[0], cos)
    assert model._rotary_tables(x, (2, 3))[0] is cos
    model.double()(x.detach().double(), spatial_shape=(3, 2))
    assert model._rotary_tables(x.double(), (3, 2))[0].dtype == torch.float64


def test_batch_independence_and_learned_convolution_uses_token_order():
    torch.manual_seed(1)
    layer = MemSolve(MemSolveConfig(12, 2, 4)).double()
    with torch.no_grad():
        layer.qk_conv.weight.add_(torch.randn_like(layer.qk_conv.weight) * .2)
    x = torch.randn(3, 9, 12, dtype=torch.float64)
    y = layer(x)
    torch.testing.assert_close(y[:1], layer(x[:1]))
    order = torch.randperm(9)
    assert not torch.allclose(layer(x[:, order]), y[:, order])


def test_checkpoint_contract_roundtrip_and_rejects_missing_or_old_semantics():
    layer = MemSolve(MemSolveConfig(12, 2, 4))
    state = deepcopy(layer.state_dict())
    layer.load_state_dict(state)
    del state["_extra_state"]
    with pytest.raises(RuntimeError, match="contract"):
        layer.load_state_dict(state, strict=False)
    state = deepcopy(layer.state_dict())
    state["_extra_state"]["version"] = 15
    with pytest.raises(RuntimeError, match="contract"):
        layer.load_state_dict(state, strict=False)


@pytest.mark.parametrize(
    "kwargs", [{"rank": 0}, {"num_heads": 0}, {"dim": 0}, {"num_heads": 5},
               {"qk_conv_dim": 3}, {"qk_conv_dim": 2, "rank": 6},
               {"qk_conv_kernel_size": 0}, {"qk_conv_kernel_size": 2},
               {"output_gate_rank": 0}]
)
def test_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        MemSolveConfig(**({"dim": 12, "num_heads": 2, "rank": 4} | kwargs))


@pytest.mark.parametrize("name", ["dim", "num_heads", "rank", "qk_conv_dim",
                                  "qk_conv_kernel_size", "output_gate_rank"])
def test_integer_dimensions(name):
    with pytest.raises(TypeError):
        MemSolveConfig(**({"dim": 12, "num_heads": 2, "rank": 4} | {name: True}))


def test_rejected_configuration_and_input():
    for option in ["core_mode", "skew_coupling", "scalar_complement"]:
        with pytest.raises(TypeError):
            MemSolveConfig(12, 2, **{option: True})
    layer = MemSolve(MemSolveConfig(12, 2))
    with pytest.raises(ValueError):
        layer(torch.empty(0, 3, 12))
    with pytest.raises(ValueError):
        layer(torch.empty(1, 0, 12))
    with pytest.raises(TypeError):
        layer(torch.ones(1, 3, 12, dtype=torch.int64))
    with pytest.raises(TypeError):
        layer(torch.ones(1, 3, 12), torch.ones(1, 3))


def test_sigmoid_gate_can_suppress_normalized_readout_and_contract_is_strict():
    layer = MemSolve(MemSolveConfig(12, 2, 4, output_gate_rank=3)).double()
    x = torch.randn(2, 7, 12, dtype=torch.float64)
    with torch.no_grad():
        layer.gate_up.weight.zero_()
        layer.gate_up.bias.zero_()
        half = layer(x)
        layer.gate_up.bias.fill_(-20)
        closed = layer(x)
    torch.testing.assert_close(closed, half * (2 * torch.sigmoid(torch.tensor(-20.0)).item()),
                               rtol=1e-7, atol=1e-15)
    for kwargs in ({"output_gate_rank": 4}, {"qk_conv_kernel_size": 5}):
        other = MemSolve(MemSolveConfig(12, 2, 4, **({"output_gate_rank": 3} | kwargs)))
        with pytest.raises(RuntimeError, match="incompatible MemSolve checkpoint contract"):
            other.set_extra_state(layer.get_extra_state())
