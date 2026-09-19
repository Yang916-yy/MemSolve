from __future__ import annotations

import copy

import pytest
import torch

from lsso import CoreMode, LSSO, LSSOConfig
from lsso.ball.reference import (
    accretive_equilibrium_mix,
    accretive_generator,
    bounded_complement,
    qr_soft_frame,
)


pytestmark = pytest.mark.core


def _reference_unmasked_output(layer: LSSO, x: torch.Tensor) -> torch.Tensor:
    """Evaluate the public unmasked contract via reference.py."""

    config = layer.config
    batch, length, _dim = x.shape
    projected = layer.w_bc(x)
    relation, content = projected.split(
        (config.num_heads * config.rank, config.dim),
        dim=-1,
    )
    relation = relation.view(
        batch,
        length,
        config.num_heads,
        config.rank,
    ).transpose(1, 2)
    content = content.view(
        batch,
        length,
        config.num_heads,
        config.head_dim,
    ).transpose(1, 2)

    valid_count = x.new_full((batch,), float(length))
    frame = qr_soft_frame(
        relation / valid_count.sqrt().view(batch, 1, 1, 1)
    )
    compact_state = frame.mT @ content
    if config.core_mode is CoreMode.DYNAMIC:
        assert layer.core_base_raw is not None
        assert layer.core_drive_weight is not None
        coordinates = layer.core_base_raw.to(dtype=x.dtype).unsqueeze(0)
        coordinates = coordinates + torch.matmul(
            compact_state,
            layer.core_drive_weight.to(dtype=x.dtype),
        ) / valid_count.sqrt().view(batch, 1, 1, 1)
        generator = accretive_generator(coordinates)
    elif config.core_mode is CoreMode.STATIC:
        assert layer.core_base_raw is not None
        generator = accretive_generator(layer.core_base_raw.to(dtype=x.dtype))
    else:
        generator = None

    output = accretive_equilibrium_mix(
        frame,
        generator,
        compact_state,
        content,
        bounded_complement(layer.eta_raw.to(dtype=x.dtype)),
    )
    output = output.transpose(1, 2).contiguous().view(
        batch,
        length,
        config.dim,
    )
    return layer.w_o(output.to(dtype=projected.dtype))


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual64 = actual.detach().to(dtype=torch.float64)
    expected64 = expected.detach().to(dtype=torch.float64)
    return float(
        torch.linalg.vector_norm(actual64 - expected64)
        / torch.linalg.vector_norm(expected64).clamp_min(1e-12)
    )


@pytest.mark.parametrize("core_mode", list(CoreMode))
def test_forward_backward(
    core_mode: CoreMode,
) -> None:
    torch.manual_seed(10)
    layer = LSSO(
        LSSOConfig(
            dim=24,
            num_heads=3,
            rank=6,
            core_mode=core_mode,

            bias=True,
        )
    ).double()
    x = torch.randn(2, 11, 24, dtype=torch.float64, requires_grad=True)
    output = layer(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


def test_default_is_the_complete_dynamic_variant() -> None:
    config = LSSOConfig(dim=24, num_heads=3, rank=6)
    assert config.core_mode is CoreMode.DYNAMIC


def test_model_diagnostics_follow_masked_reference_problem() -> None:
    torch.manual_seed(71)
    layer = LSSO(
        LSSOConfig(dim=24, num_heads=3, rank=6, bias=True)
    ).double().eval()
    with torch.no_grad():
        assert layer.core_base_raw is not None
        assert layer.core_drive_weight is not None
        layer.core_base_raw.normal_(std=0.1)
        layer.core_drive_weight.normal_(std=0.1)
        layer.eta_raw.normal_(std=0.3)
    x = torch.randn(2, 11, 24, dtype=torch.float64)
    mask = torch.tensor(
        [[True] * 9 + [False] * 2, [True] * 7 + [False] * 4]
    )
    rhs = torch.randn(2, 3, 6, 8, dtype=torch.float64)

    diagnostics = layer.diagnostics(x, valid_mask=mask, adjoint_rhs=rhs)

    assert set(diagnostics) == {
        "q",
        "contraction_slack",
        "mu",
        "state_ratio",
        "adjoint_ratio",
        "state_bound_usage",
        "adjoint_bound_usage",
    }
    assert all(value.shape == (2, 3) for value in diagnostics.values())
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    assert torch.all(diagnostics["q"] < 1.0)
    assert torch.all(diagnostics["state_bound_usage"] <= 1.0 + 2e-12)
    assert torch.all(diagnostics["adjoint_bound_usage"] <= 1.0 + 2e-12)


@pytest.mark.parametrize(
    ("mode", "present", "absent"),
    [
        (
            CoreMode.DYNAMIC,
            {"core_base_raw", "core_drive_weight"},
            set(),
        ),
        (
            CoreMode.STATIC,
            {"core_base_raw"},
            {"core_drive_weight"},
        ),
        (
            CoreMode.ZERO,
            set(),
            {"core_base_raw", "core_drive_weight"},
        ),
    ],
)
def test_core_modes_create_only_owned_parameters(
    mode: CoreMode,
    present: set[str],
    absent: set[str],
) -> None:
    names = set(
        dict(LSSO(LSSOConfig(16, 2, rank=4, core_mode=mode)).named_parameters())
    )
    assert {"eta_raw", "w_bc.weight", "w_o.weight"} <= names
    assert present <= names
    assert names.isdisjoint(absent)


@pytest.mark.parametrize("core_mode", list(CoreMode))
def test_public_forward_matches_direct_equilibrium_reference(
    core_mode: CoreMode,
) -> None:
    torch.manual_seed(11)
    layer = LSSO(
        LSSOConfig(
            dim=16,
            num_heads=2,
            rank=4,
            core_mode=core_mode,

            bias=True,
        )
    ).double()
    with torch.no_grad():
        layer.w_bc.weight.normal_(std=0.2)
        assert layer.w_bc.bias is not None
        layer.w_bc.bias.normal_(std=0.05)
        layer.w_o.weight.normal_(std=0.2)
        assert layer.w_o.bias is not None
        layer.w_o.bias.normal_(std=0.05)
        layer.eta_raw.copy_(torch.tensor([-0.4, 0.6], dtype=torch.float64))
        if layer.core_base_raw is not None:
            layer.core_base_raw.normal_(std=0.15)
        if layer.core_drive_weight is not None:
            layer.core_drive_weight.normal_(std=0.15)

    x = torch.randn(3, 7, 16, dtype=torch.float64)
    expected = _reference_unmasked_output(layer, x)
    torch.testing.assert_close(layer(x), expected, rtol=2e-11, atol=2e-11)


def test_dynamic_public_output_uses_content_conditioning() -> None:
    torch.manual_seed(12)
    dynamic = LSSO(
        LSSOConfig(16, 2, rank=4, core_mode=CoreMode.DYNAMIC, )
    ).double()
    static = LSSO(
        LSSOConfig(16, 2, rank=4, core_mode=CoreMode.STATIC, )
    ).double()
    with torch.no_grad():
        dynamic.w_bc.weight.normal_(std=0.2)
        dynamic.w_o.weight.normal_(std=0.2)
        dynamic.eta_raw.copy_(torch.tensor([-0.3, 0.5], dtype=torch.float64))
        assert dynamic.core_base_raw is not None
        assert dynamic.core_drive_weight is not None
        dynamic.core_base_raw.normal_(std=0.15)
        dynamic.core_drive_weight.normal_(std=0.4)

        static.w_bc.weight.copy_(dynamic.w_bc.weight)
        static.w_o.weight.copy_(dynamic.w_o.weight)
        static.eta_raw.copy_(dynamic.eta_raw)
        assert static.core_base_raw is not None
        static.core_base_raw.copy_(dynamic.core_base_raw)

    x = torch.randn(3, 7, 16, dtype=torch.float64)
    assert not torch.allclose(dynamic(x), static(x))


def test_dynamic_cross_moment_is_replication_invariant() -> None:
    torch.manual_seed(13)
    layer = LSSO(
        LSSOConfig(16, 2, rank=4)
    ).double().eval()
    with torch.no_grad():
        assert layer.core_drive_weight is not None
        layer.core_drive_weight.normal_(std=0.05)

    x = torch.randn(2, 5, 16, dtype=torch.float64)
    output = layer(x)

    copies = 3
    repeated_x = x.repeat_interleave(copies, dim=1)
    repeated_output = layer(repeated_x)
    torch.testing.assert_close(
        repeated_output,
        output.repeat_interleave(copies, dim=1),
        rtol=2e-11,
        atol=2e-11,
    )


def test_dynamic_zero_init_receives_gradient() -> None:
    torch.manual_seed(14)
    layer = LSSO(LSSOConfig(16, 2, rank=4)).double()
    x = torch.randn(3, 9, 16, dtype=torch.float64)
    layer(x).square().mean().backward()
    assert layer.core_base_raw is not None
    assert layer.core_drive_weight is not None
    assert layer.core_base_raw.grad is not None
    assert layer.core_drive_weight.grad is not None
    assert torch.count_nonzero(layer.core_base_raw.grad) > 0
    assert torch.count_nonzero(layer.core_drive_weight.grad) > 0


def test_static_core_is_batch_independent() -> None:
    torch.manual_seed(15)
    layer = LSSO(
        LSSOConfig(16, 2, rank=4, core_mode=CoreMode.STATIC, )
    ).double().eval()
    with torch.no_grad():
        assert layer.core_base_raw is not None
        layer.core_base_raw.normal_(std=0.1)

    first = torch.randn(1, 7, 16, dtype=torch.float64)
    second = torch.randn(1, 7, 16, dtype=torch.float64)
    expected = torch.cat((layer(first), layer(second)), dim=0)
    actual = layer(torch.cat((first, second), dim=0))
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)


def test_zero_core_stays_exact_without_core_parameters() -> None:
    torch.manual_seed(16)
    layer = LSSO(
        LSSOConfig(16, 2, rank=4, core_mode=CoreMode.ZERO, )
    ).double()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    x = torch.randn(2, 7, 16, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    layer(x).square().mean().backward()
    optimizer.step()

    expected = _reference_unmasked_output(layer, x)
    torch.testing.assert_close(layer(x), expected, rtol=2e-11, atol=2e-11)
    names = set(dict(layer.named_parameters()))
    assert "core_base_raw" not in names
    assert "core_drive_weight" not in names


def test_learned_complement_is_a_strict_interior_tanh() -> None:
    layer = LSSO(LSSOConfig(dim=8, num_heads=2, rank=4))
    torch.testing.assert_close(layer.complement(), torch.full((2,), 0.9))
    assert layer.eta_raw.shape == (2,)
    with torch.no_grad():
        layer.eta_raw.copy_(torch.tensor([-3.0, 4.0]))
    eta = layer.complement().detach()
    torch.testing.assert_close(eta, bounded_complement(layer.eta_raw.detach()))
    assert torch.all(eta.abs() < 1.0)
    assert float(eta[0]) < 0.0 < float(eta[1])


def test_forward_uses_the_complement_method() -> None:
    class ComplementSpy(LSSO):
        def __init__(self, config: LSSOConfig) -> None:
            super().__init__(config)
            self.complement_calls = 0

        def complement(self) -> torch.Tensor:
            self.complement_calls += 1
            return super().complement()

    layer = ComplementSpy(
        LSSOConfig(8, 2, rank=4)
    )
    layer(torch.randn(1, 3, 8))
    assert layer.complement_calls == 1


def test_removed_configuration_knobs_are_not_accepted() -> None:
    with pytest.raises(TypeError):
        LSSOConfig(8, 2, eta=0.9)
    with pytest.raises(TypeError):
        LSSOConfig(8, 2, learn_eta=False)
    with pytest.raises(TypeError):
        LSSOConfig(8, 2, rotary_base=1000.0)


def test_configuration_requires_boolean_switches() -> None:
    with pytest.raises(TypeError, match="bias"):
        LSSOConfig(8, 2, rank=4, bias=1)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("dim", 8.0), ("num_heads", True), ("rank", 4.0)],
)
def test_configuration_requires_integer_dimensions(
    keyword: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {"dim": 8, "num_heads": 2, "rank": 4}
    arguments[keyword] = value
    with pytest.raises(TypeError, match=keyword):
        LSSOConfig(**arguments)  # type: ignore[arg-type]


def test_core_mode_accepts_enum_values_and_strings() -> None:
    assert LSSOConfig(8, 2, rank=4, core_mode="zero").core_mode is CoreMode.ZERO
    with pytest.raises(ValueError, match="core_mode"):
        LSSOConfig(8, 2, rank=4, core_mode="unknown")


def test_gapped_nan_padding_matches_cropped_sequence() -> None:
    torch.manual_seed(17)
    config = LSSOConfig(dim=16, num_heads=2, rank=8, bias=True)
    layer = LSSO(config).double().eval()
    reference = copy.deepcopy(layer)
    clean = torch.randn(1, 8, 16, dtype=torch.float64)
    mask = torch.tensor([[True, False, True, False, False, True, False, False]])
    poisoned = torch.where(
        mask[:, :, None], clean, torch.full_like(clean, float("nan"))
    )
    output = layer(poisoned, valid_mask=mask, )
    kept = mask[0].nonzero(as_tuple=False).flatten()
    expected = reference(clean[:, kept], )
    torch.testing.assert_close(output[:, kept], expected, rtol=2e-11, atol=2e-11)
    assert torch.count_nonzero(output[:, ~mask[0]]) == 0


def test_all_masked_returns_zero() -> None:
    layer = LSSO(LSSOConfig(16, 2, rank=8, bias=True)).double()
    x = torch.full((2, 5, 16), float("nan"), dtype=torch.float64)
    output = layer(x, valid_mask=torch.zeros(2, 5, dtype=torch.bool))
    assert torch.count_nonzero(output) == 0


def test_mask_dtype_is_strict() -> None:
    layer = LSSO(LSSOConfig(8, 2, rank=4))
    x = torch.randn(1, 3, 8)
    with pytest.raises(TypeError, match="torch.bool"):
        layer(x, valid_mask=torch.tensor([[1, 0, 1]]))


def test_checkpoint_contract_rejects_semantic_mismatch() -> None:
    source = LSSO(LSSOConfig(16, 2, rank=4))
    target = LSSO(LSSOConfig(16, 2, rank=4, skew_coupling=False))
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        target.load_state_dict(source.state_dict(), strict=True)


@pytest.mark.parametrize("strict", (True, False))
def test_checkpoint_contract_cannot_be_omitted(
    strict: bool,
) -> None:
    source = LSSO(LSSOConfig(16, 2, rank=4))
    state = copy.deepcopy(source.state_dict())
    del state["_extra_state"]

    target = LSSO(LSSOConfig(16, 2, rank=4))
    with pytest.raises(RuntimeError, match="missing its configuration contract"):
        target.load_state_dict(state, strict=strict)


def test_checkpoint_contract_cannot_be_bypassed_in_a_parent_load() -> None:
    class Container(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mixer = LSSO(LSSOConfig(16, 2, rank=4))

    source = Container()
    state = copy.deepcopy(source.state_dict())
    del state["mixer._extra_state"]

    target = Container()
    with pytest.raises(RuntimeError, match="missing its configuration contract"):
        target.load_state_dict(state, strict=False)


def test_checkpoint_contract_mismatch_cannot_be_bypassed_non_strictly() -> None:
    source = LSSO(LSSOConfig(16, 2, rank=4))
    target = LSSO(LSSOConfig(16, 2, rank=4, skew_coupling=False))
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        target.load_state_dict(source.state_dict(), strict=False)


def test_checkpoint_contract_rejects_previous_numerics() -> None:
    source = LSSO(LSSOConfig(16, 2, rank=4))
    legacy_state = copy.deepcopy(source.state_dict())
    extra_state = legacy_state["_extra_state"]
    assert isinstance(extra_state, dict)
    assert extra_state["version"] == 13
    assert extra_state["numerics"] == "tf32-wbc-ieee-fgram-tc16-v6"
    extra_state["numerics"] = "tf32-fp32-wbc-tc16-v4"

    target = LSSO(LSSOConfig(16, 2, rank=4))
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        target.load_state_dict(legacy_state, strict=True)


def test_checkpoint_contract_round_trips() -> None:
    torch.manual_seed(20)
    config = LSSOConfig(16, 2, rank=4)
    source = LSSO(config)
    target = LSSO(config)
    target.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 7, 16)
    torch.testing.assert_close(target(x), source(x))


def test_empty_sequence_is_rejected_before_mask_reduction() -> None:
    layer = LSSO(LSSOConfig(8, 2, rank=4))
    x = torch.empty(2, 0, 8)
    with pytest.raises(ValueError, match="sequence length"):
        layer(x)
    with pytest.raises(ValueError, match="sequence length"):
        layer(x, valid_mask=torch.empty(2, 0, dtype=torch.bool))


@pytest.mark.parametrize("implementation", ("reference", "cuda"))
def test_empty_batch_is_rejected_before_backend_dispatch(
    implementation: str,
) -> None:
    layer = LSSO(LSSOConfig(8, 2, rank=4))
    x = torch.empty(0, 3, 8)

    with pytest.raises(ValueError, match="batch size"):
        layer(x, implementation=implementation)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("core_mode", list(CoreMode))
def test_mixed_precision_reference_matches_fp64_oracle_for_outputs_and_gradients(
    core_mode: CoreMode,
) -> None:
    torch.manual_seed(22)
    layer = LSSO(
        LSSOConfig(
            dim=32,
            num_heads=2,
            rank=8,
            core_mode=core_mode,

            bias=True,
        )
    ).cuda()
    with torch.no_grad():
        layer.w_bc.weight.normal_(std=0.08)
        layer.w_o.weight.normal_(std=0.08)
        assert layer.w_bc.bias is not None
        assert layer.w_o.bias is not None
        layer.w_bc.bias.normal_(std=0.02)
        layer.w_o.bias.normal_(std=0.02)
        layer.eta_raw.normal_(std=0.3)
        if layer.core_base_raw is not None:
            layer.core_base_raw.normal_(std=0.06)
        if layer.core_drive_weight is not None:
            layer.core_drive_weight.normal_(std=0.06)

    oracle = copy.deepcopy(layer).double()
    x = (0.4 * torch.randn(2, 17, 32, device="cuda")).requires_grad_()
    oracle_x = x.detach().double().requires_grad_()
    upstream = torch.randn_like(x)

    output = layer(x, )
    oracle_output = oracle(oracle_x, )
    actual_parameters = dict(layer.named_parameters())
    oracle_parameters = dict(oracle.named_parameters())
    assert actual_parameters.keys() == oracle_parameters.keys()
    actual_gradients = torch.autograd.grad(
        (output * upstream).sum(),
        (x, *actual_parameters.values()),
    )
    oracle_gradients = torch.autograd.grad(
        (oracle_output * upstream.double()).sum(),
        (oracle_x, *oracle_parameters.values()),
    )

    assert output.dtype is torch.float32
    assert torch.isfinite(output).all()
    assert _relative_l2(output, oracle_output) <= 5e-3
    for actual_gradient, oracle_gradient in zip(actual_gradients, oracle_gradients):
        assert torch.isfinite(actual_gradient).all()
        assert _relative_l2(actual_gradient, oracle_gradient) <= 3e-2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_mixed_precision_reference_preserves_public_input_dtype(dtype: torch.dtype) -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=8)).cuda().eval()
    x = torch.randn(2, 9, 32, device="cuda", dtype=dtype, requires_grad=True)
    output = layer(x)
    output.float().square().mean().backward()

    assert output.dtype is dtype
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


def test_reference_accepts_bfloat16_input() -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=16))
    x = torch.randn(2, 9, 32, dtype=torch.bfloat16, requires_grad=True)
    output = layer(x)
    assert output.dtype is torch.bfloat16
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_reference_rejects_float8_input(dtype: torch.dtype) -> None:
    layer = LSSO(LSSOConfig(dim=32, num_heads=2, rank=8)).eval()
    x = torch.zeros(2, 9, 32, dtype=dtype)

    with pytest.raises(TypeError, match="does not support x with dtype"):
        layer(x, implementation="reference")


def test_symmetric_core_ablation_matches_triangular_coordinates_and_gradients() -> None:
    from lsso.ball.reference import accretive_generator
    raw = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
    actual = accretive_generator(raw, skew_coupling=False)
    expected = accretive_generator(torch.tril(raw))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, actual.mT)
    weight = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * weight).sum(), raw)[0]
    expected_grad = torch.autograd.grad((expected * weight).sum(), raw)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.count_nonzero(torch.triu(actual_grad, diagonal=1)) == 0


def test_zero_complement_ablation_is_fixed_and_has_distinct_checkpoint_contract() -> None:
    model = LSSO(LSSOConfig(8, 2, rank=4, scalar_complement=False)).double()
    assert not model.eta_raw.requires_grad
    assert torch.count_nonzero(model.complement()) == 0
    x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    output = model(x, implementation="reference")
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert model.eta_raw.grad is None
    default = LSSO(LSSOConfig(8, 2, rank=4)).double()
    with pytest.raises(RuntimeError, match="contract"):
        default.load_state_dict(model.state_dict())


@pytest.mark.parametrize("option", ["skew_coupling", "scalar_complement"])
def test_new_ablations_require_boolean_flags_and_explicit_reference(option: str) -> None:
    with pytest.raises(TypeError, match=option):
        LSSOConfig(16, 1, rank=16, **{option: "false"})
    model = LSSO(LSSOConfig(16, 1, rank=16, **{option: False}))
    with pytest.raises(ValueError, match="requires"):
        model(torch.randn(1, 3, 16), implementation="cuda")

@pytest.mark.parametrize("mode", list(CoreMode))
def test_unrotated_operator_is_token_permutation_equivariant(mode: CoreMode) -> None:
    torch.manual_seed(194)
    layer = LSSO(LSSOConfig(16, 2, rank=5, core_mode=mode)).double()
    with torch.no_grad():
        if layer.core_base_raw is not None:
            layer.core_base_raw.normal_(0, 0.1)
        if layer.core_drive_weight is not None:
            layer.core_drive_weight.normal_(0, 0.1)
    x = torch.randn(2, 11, 16, dtype=torch.float64)
    mask = torch.rand(2, 11) > 0.25
    permutation = torch.randperm(11)
    expected = layer(x, valid_mask=mask)[:, permutation]
    actual = layer(x[:, permutation], valid_mask=mask[:, permutation])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
