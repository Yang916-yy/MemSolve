from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from memsolve import CoreMode, MemSolve, MemSolveConfig
from memsolve.ball.reference import (
    accretive_equilibrium_mix,
    accretive_generator,
    bounded_complement,
    qr_soft_frame,
    rotary_position_factors,
    rotate_content,
)


pytestmark = pytest.mark.core


def _reference_unmasked_output(layer: MemSolve, x: torch.Tensor) -> torch.Tensor:
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
    if config.core_mode is CoreMode.EQUILIBRIUM:
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


@pytest.mark.parametrize(("keyword", "value", "error"), [
    ("rotary", 1, TypeError), ("rotary_train_length", 0, ValueError),
    ("rotary_train_length", 128., TypeError), ("rotary_base", True, TypeError),
    ("rotary_base", float("inf"), ValueError), ("rotary_base", 1., ValueError),
    ("rotary_scaling", "linear", ValueError),
])
def test_rotary_config_validates_external_settings(keyword, value, error) -> None:
    with pytest.raises(error, match="rotary"):
        MemSolveConfig(8, 2, **{keyword: value})


def test_rotary_contract_is_explicit_json_safe_and_does_not_add_parameters() -> None:
    plain = MemSolve(MemSolveConfig(8, 2, rank=4))
    rotary = MemSolve(MemSolveConfig(8, 2, rank=4, rotary=True))
    assert "rotary" not in plain.get_extra_state()
    assert plain.get_extra_state() == MemSolve(MemSolveConfig(8, 2, rank=4, rotary_base=1000.)).get_extra_state()
    assert set(dict(plain.named_parameters())) == set(dict(rotary.named_parameters()))
    rotary.set_extra_state(json.loads(json.dumps(rotary.get_extra_state())))
    for config in (plain.config, replace(rotary.config, rotary_base=1000.),
                   replace(rotary.config, rotary_train_length=64),
                   replace(rotary.config, rotary_scaling="none")):
        with pytest.raises(RuntimeError, match="checkpoint contract"):
            MemSolve(config).load_state_dict(rotary.state_dict())
    MemSolve(MemSolveConfig(6, 2))
    with pytest.raises(ValueError, match="even head_dim"):
        MemSolveConfig(6, 2, rotary=True)


@pytest.mark.parametrize("mode", list(CoreMode))
def test_rotary_forward_matches_zero_phase_and_is_shift_invariant(mode: CoreMode) -> None:
    torch.manual_seed(602)
    layer = MemSolve(MemSolveConfig(8, 2, rank=4, core_mode=mode,
                                   rotary=True, rotary_train_length=4, bias=True)).double()
    if layer.core_base_raw is not None:
        with torch.no_grad():
            layer.core_base_raw.normal_(std=.3)
    x = torch.randn(2, 9, 8, dtype=torch.float64, requires_grad=True)
    plain = copy.deepcopy(layer)
    plain.config = replace(plain.config, rotary=False)
    zero_positions = torch.zeros(9, dtype=torch.long)
    torch.testing.assert_close(layer(x, position_ids=zero_positions), plain(x), rtol=0, atol=0)
    ids = torch.arange(9)
    context_length = torch.tensor([12, 18])
    expected = layer(x, position_ids=ids, rotary_context_length=context_length)
    shifted = layer(x, position_ids=ids[None] + torch.tensor([[4096], [12345]]),
                    rotary_context_length=context_length)
    torch.testing.assert_close(shifted, expected, rtol=2e-11, atol=2e-11)
    expected_grad = torch.autograd.grad(expected.square().sum(), (x, *layer.parameters()), retain_graph=True)
    shifted_grad = torch.autograd.grad(shifted.square().sum(), (x, *layer.parameters()))
    for left, right in zip(expected_grad, shifted_grad):
        torch.testing.assert_close(left, right, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("mode", list(CoreMode))
@pytest.mark.parametrize("length", [1, 7])
def test_rotary_padding_does_not_change_frequency_or_readout(mode: CoreMode, length: int) -> None:
    torch.manual_seed(603)
    layer = MemSolve(MemSolveConfig(8, 2, rank=16, core_mode=mode, rotary=True,
                                   rotary_train_length=4, bias=True)).double()
    x = torch.randn(1, length, 8, dtype=torch.float64)
    expected = layer(x)
    padded = torch.full((2, length + 9, 8), float("nan"), dtype=torch.float64)
    mask = torch.zeros(2, length + 9, dtype=torch.bool)
    padded[0, 3:3+length] = x[0]
    mask[0, 3:3+length] = True
    padded.requires_grad_()
    actual = layer(padded, mask)
    torch.testing.assert_close(actual[0, 3:3+length], expected[0], rtol=2e-12, atol=2e-12)
    assert torch.count_nonzero(actual[~mask]) == 0
    actual.square().sum().backward()
    assert torch.isfinite(padded.grad).all()
    assert torch.count_nonzero(padded.grad[~mask]) == 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def test_rotary_context_uses_mask_span_and_sequence_parallel_global_coordinates() -> None:
    layer = MemSolve(MemSolveConfig(8, 2, rank=4, rotary=True, rotary_train_length=2))
    x = torch.randn(2, 8, 8, dtype=torch.float64)
    mask = torch.tensor([[False, True, False, False, True, False, False, False], [False]*8])
    actual = layer._rotary_factors(x, mask, None, None)
    expected = rotary_position_factors(torch.arange(8)[None], head_dim=4, base=10000.,
        train_length=2, context_length=torch.tensor([4, 1]), scaling="yarn", dtype=torch.float64)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    context = SimpleNamespace(offset=16, global_length=64)
    actual = layer._rotary_factors(x, None, None, None, sequence_parallel=context)
    expected = rotary_position_factors(torch.arange(16, 24)[None], head_dim=4, base=10000.,
        train_length=2, context_length=64, scaling="yarn", dtype=torch.float64)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with pytest.raises(ValueError, match="global rotary_context_length"):
        layer._rotary_factors(x, mask, None, None, sequence_parallel=context)
    explicit = layer._rotary_factors(x, mask, None, torch.tensor([4, 1]), sequence_parallel=context)
    assert explicit[0].shape == (2, 8, 2)
    layer.config = replace(layer.config, rotary_train_length=64)
    shared = layer._rotary_factors(x, mask, None, None, sequence_parallel=context)
    assert shared[0].shape == (1, 8, 2)
    layer.config = replace(layer.config, rotary_train_length=2, rotary_scaling="none")
    unscaled = layer._rotary_factors(x, mask, None, None, sequence_parallel=context)
    assert unscaled[0].shape == (1, 8, 2)


@pytest.mark.parametrize(("kwargs", "error", "message"), [
    ({"position_ids": torch.ones(4)}, TypeError, "integer tensor"),
    ({"position_ids": torch.ones(4, dtype=torch.bool)}, TypeError, "integer tensor"),
    ({"position_ids": torch.ones(2, 3, dtype=torch.long)}, ValueError, "shape"),
    ({"rotary_context_length": 0}, ValueError, "positive"),
    ({"rotary_context_length": 4.}, TypeError, "integer"),
    ({"rotary_context_length": torch.ones(2)}, TypeError, "integer tensor"),
    ({"rotary_context_length": torch.tensor([4])}, ValueError, "shape"),
    ({"rotary_context_length": torch.tensor([4, 0])}, RuntimeError, "positive"),
])
def test_rotary_forward_validates_position_inputs(kwargs, error, message) -> None:
    layer = MemSolve(MemSolveConfig(8, 2, rank=4, rotary=True))
    with pytest.raises(error, match=message):
        layer(torch.randn(2, 4, 8), **kwargs)


def test_rotary_disabled_rejects_position_arguments() -> None:
    layer = MemSolve(MemSolveConfig(8, 2, rank=4))
    with pytest.raises(ValueError, match="rotary=True"):
        layer(torch.randn(1, 4, 8), position_ids=torch.arange(4))


def test_rotary_diagnostics_uses_transported_compact_state() -> None:
    torch.manual_seed(604)
    layer = MemSolve(MemSolveConfig(8, 2, rank=4, rotary=True, rotary_train_length=4)).double()
    x = torch.randn(2, 9, 8, dtype=torch.float64)
    rhs = torch.randn(2, 2, 4, 4, dtype=torch.float64)
    actual = layer.diagnostics(x, adjoint_rhs=rhs, position_ids=torch.arange(9))
    shifted = layer.diagnostics(x, adjoint_rhs=rhs, position_ids=torch.arange(9)+4096)
    for key in ("q", "contraction_slack", "mu", "state_ratio", "state_bound_usage"):
        torch.testing.assert_close(actual[key], shifted[key], rtol=2e-12, atol=2e-12)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", list(CoreMode))
def test_rotary_cuda_reference_keeps_rotated_products_in_fp32(mode: CoreMode) -> None:
    torch.manual_seed(605)
    layer = MemSolve(MemSolveConfig(128, 2, rank=16, core_mode=mode, rotary=True,
                                   rotary_train_length=16)).cuda()
    if layer.core_base_raw is not None:
        with torch.no_grad():
            layer.core_base_raw.normal_(std=.2)
    x = torch.randn(2, 37, 128, device="cuda", dtype=torch.bfloat16)
    cos, sin = layer._rotary_factors(x, None, None, torch.tensor([37, 74], device="cuda"))
    _, frame, state, content, generator, eta, _ = layer._reference_compact_problem(
        x, None, rotary_cos=cos, rotary_sin=sin,
    )
    actual = accretive_equilibrium_mix(frame, generator, state, content, eta,
                                      rotary_cos=cos, rotary_sin=sin)
    # Compare the same packed projections and phase tables; projection rounding
    # is outside this check of the two rotary-content matrix products.
    frame64 = frame.detach().double()
    content64 = content.detach().double().requires_grad_()
    state64 = frame64.mT @ rotate_content(content64, cos.double(), sin.double())
    expected = accretive_equilibrium_mix(
        frame64, None if generator is None else generator.detach().double(),
        state64, content64, eta.detach().double(),
        rotary_cos=cos.double(), rotary_sin=sin.double(),
    )
    assert _relative_l2(state, state64) < 3e-6
    assert _relative_l2(actual, expected) < 3e-6
    cotangent = torch.randn_like(actual)
    gradient = torch.autograd.grad((actual * cotangent).sum(), content)[0]
    gradient64 = torch.autograd.grad((expected * cotangent.double()).sum(), content64)[0]
    assert _relative_l2(gradient, gradient64) < 3e-6


@pytest.mark.parametrize("core_mode", list(CoreMode))
def test_forward_backward(
    core_mode: CoreMode,
) -> None:
    torch.manual_seed(10)
    layer = MemSolve(
        MemSolveConfig(
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


def test_default_is_equilibrium() -> None:
    config = MemSolveConfig(dim=24, num_heads=3, rank=6)
    assert config.core_mode is CoreMode.EQUILIBRIUM


def test_model_diagnostics_follow_masked_reference_problem() -> None:
    torch.manual_seed(71)
    layer = MemSolve(
        MemSolveConfig(dim=24, num_heads=3, rank=6, bias=True)
    ).double().eval()
    with torch.no_grad():
        assert layer.core_base_raw is not None
        layer.core_base_raw.normal_(std=0.1)
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
            CoreMode.EQUILIBRIUM,
            {"core_base_raw"},
            {"core_drive_weight"},
        ),
        (
            CoreMode.COMPLEMENT,
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
        dict(MemSolve(MemSolveConfig(16, 2, rank=4, core_mode=mode)).named_parameters())
    )
    assert {"eta_raw", "w_bc.weight", "w_o.weight"} <= names
    assert present <= names
    assert names.isdisjoint(absent)


@pytest.mark.parametrize("core_mode", list(CoreMode))
def test_public_forward_matches_direct_equilibrium_reference(
    core_mode: CoreMode,
) -> None:
    torch.manual_seed(11)
    layer = MemSolve(
        MemSolveConfig(
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

    x = torch.randn(3, 7, 16, dtype=torch.float64)
    expected = _reference_unmasked_output(layer, x)
    torch.testing.assert_close(layer(x), expected, rtol=2e-11, atol=2e-11)




def test_equilibrium_mix_is_replication_invariant() -> None:
    torch.manual_seed(13)
    layer = MemSolve(
        MemSolveConfig(16, 2, rank=4)
    ).double().eval()
    with torch.no_grad():
        layer.core_base_raw.normal_(std=0.05)

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


def test_equilibrium_zero_init_receives_gradient() -> None:
    torch.manual_seed(14)
    layer = MemSolve(MemSolveConfig(16, 2, rank=4)).double()
    x = torch.randn(3, 9, 16, dtype=torch.float64)
    layer(x).square().mean().backward()
    assert layer.core_base_raw is not None
    assert layer.core_base_raw.grad is not None
    assert torch.count_nonzero(layer.core_base_raw.grad) > 0


def test_equilibrium_core_is_batch_independent() -> None:
    torch.manual_seed(15)
    layer = MemSolve(
        MemSolveConfig(16, 2, rank=4, core_mode=CoreMode.EQUILIBRIUM, )
    ).double().eval()
    with torch.no_grad():
        assert layer.core_base_raw is not None
        layer.core_base_raw.normal_(std=0.1)

    first = torch.randn(1, 7, 16, dtype=torch.float64)
    second = torch.randn(1, 7, 16, dtype=torch.float64)
    expected = torch.cat((layer(first), layer(second)), dim=0)
    actual = layer(torch.cat((first, second), dim=0))
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)


def test_complement_core_stays_exact_without_core_parameters() -> None:
    torch.manual_seed(16)
    layer = MemSolve(
        MemSolveConfig(16, 2, rank=4, core_mode=CoreMode.COMPLEMENT, )
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
    layer = MemSolve(MemSolveConfig(dim=8, num_heads=2, rank=4))
    torch.testing.assert_close(layer.complement(), torch.full((2,), 0.9))
    assert layer.eta_raw.shape == (2,)
    with torch.no_grad():
        layer.eta_raw.copy_(torch.tensor([-3.0, 4.0]))
    eta = layer.complement().detach()
    torch.testing.assert_close(eta, bounded_complement(layer.eta_raw.detach()))
    assert torch.all(eta.abs() < 1.0)
    assert float(eta[0]) < 0.0 < float(eta[1])


def test_forward_uses_the_complement_method() -> None:
    class ComplementSpy(MemSolve):
        def __init__(self, config: MemSolveConfig) -> None:
            super().__init__(config)
            self.complement_calls = 0

        def complement(self) -> torch.Tensor:
            self.complement_calls += 1
            return super().complement()

    layer = ComplementSpy(
        MemSolveConfig(8, 2, rank=4)
    )
    layer(torch.randn(1, 3, 8))
    assert layer.complement_calls == 1


def test_removed_configuration_knobs_are_not_accepted() -> None:
    with pytest.raises(TypeError):
        MemSolveConfig(8, 2, eta=0.9)
    with pytest.raises(TypeError):
        MemSolveConfig(8, 2, learn_eta=False)


def test_configuration_requires_boolean_switches() -> None:
    with pytest.raises(TypeError, match="bias"):
        MemSolveConfig(8, 2, rank=4, bias=1)


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
        MemSolveConfig(**arguments)  # type: ignore[arg-type]


def test_core_mode_accepts_enum_values_and_strings() -> None:
    assert MemSolveConfig(8, 2, rank=4, core_mode="complement").core_mode is CoreMode.COMPLEMENT
    with pytest.raises(ValueError, match="core_mode"):
        MemSolveConfig(8, 2, rank=4, core_mode="unknown")


def test_gapped_nan_padding_matches_cropped_sequence() -> None:
    torch.manual_seed(17)
    config = MemSolveConfig(dim=16, num_heads=2, rank=8, bias=True)
    layer = MemSolve(config).double().eval()
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
    layer = MemSolve(MemSolveConfig(16, 2, rank=8, bias=True)).double()
    x = torch.full((2, 5, 16), float("nan"), dtype=torch.float64)
    output = layer(x, valid_mask=torch.zeros(2, 5, dtype=torch.bool))
    assert torch.count_nonzero(output) == 0


def test_mask_dtype_is_strict() -> None:
    layer = MemSolve(MemSolveConfig(8, 2, rank=4))
    x = torch.randn(1, 3, 8)
    with pytest.raises(TypeError, match="torch.bool"):
        layer(x, valid_mask=torch.tensor([[1, 0, 1]]))


def test_checkpoint_contract_rejects_semantic_mismatch() -> None:
    source = MemSolve(MemSolveConfig(16, 2, rank=4))
    target = MemSolve(MemSolveConfig(16, 2, rank=4, skew_coupling=False))
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        target.load_state_dict(source.state_dict(), strict=True)


@pytest.mark.parametrize("strict", (True, False))
def test_checkpoint_contract_cannot_be_omitted(
    strict: bool,
) -> None:
    source = MemSolve(MemSolveConfig(16, 2, rank=4))
    state = copy.deepcopy(source.state_dict())
    del state["_extra_state"]

    target = MemSolve(MemSolveConfig(16, 2, rank=4))
    with pytest.raises(RuntimeError, match="missing its configuration contract"):
        target.load_state_dict(state, strict=strict)


def test_checkpoint_contract_cannot_be_bypassed_in_a_parent_load() -> None:
    class Container(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mixer = MemSolve(MemSolveConfig(16, 2, rank=4))

    source = Container()
    state = copy.deepcopy(source.state_dict())
    del state["mixer._extra_state"]

    target = Container()
    with pytest.raises(RuntimeError, match="missing its configuration contract"):
        target.load_state_dict(state, strict=False)


def test_checkpoint_contract_mismatch_cannot_be_bypassed_non_strictly() -> None:
    source = MemSolve(MemSolveConfig(16, 2, rank=4))
    target = MemSolve(MemSolveConfig(16, 2, rank=4, skew_coupling=False))
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        target.load_state_dict(source.state_dict(), strict=False)


def test_checkpoint_contract_rejects_previous_numerics() -> None:
    source = MemSolve(MemSolveConfig(16, 2, rank=4))
    legacy_state = copy.deepcopy(source.state_dict())
    extra_state = legacy_state["_extra_state"]
    assert isinstance(extra_state, dict)
    assert extra_state["version"] == 26
    assert extra_state["numerics"] == "tf32-wbc-ieee-fgram-tc16-v6"
    extra_state["numerics"] = "tf32-fp32-wbc-tc16-v4"

    target = MemSolve(MemSolveConfig(16, 2, rank=4))
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        target.load_state_dict(legacy_state, strict=True)


def test_checkpoint_contract_round_trips() -> None:
    torch.manual_seed(20)
    config = MemSolveConfig(16, 2, rank=4)
    source = MemSolve(config)
    target = MemSolve(config)
    target.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 7, 16)
    torch.testing.assert_close(target(x), source(x))


def test_empty_sequence_is_rejected_before_mask_reduction() -> None:
    layer = MemSolve(MemSolveConfig(8, 2, rank=4))
    x = torch.empty(2, 0, 8)
    with pytest.raises(ValueError, match="sequence length"):
        layer(x)
    with pytest.raises(ValueError, match="sequence length"):
        layer(x, valid_mask=torch.empty(2, 0, dtype=torch.bool))


@pytest.mark.parametrize("implementation", ("reference", "cuda"))
def test_empty_batch_is_rejected_before_backend_dispatch(
    implementation: str,
) -> None:
    layer = MemSolve(MemSolveConfig(8, 2, rank=4))
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
    layer = MemSolve(
        MemSolveConfig(
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
    layer = MemSolve(MemSolveConfig(dim=32, num_heads=2, rank=8)).cuda().eval()
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
    layer = MemSolve(MemSolveConfig(dim=32, num_heads=2, rank=16))
    x = torch.randn(2, 9, 32, dtype=torch.bfloat16, requires_grad=True)
    output = layer(x)
    assert output.dtype is torch.bfloat16
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
def test_reference_rejects_float8_input(dtype: torch.dtype) -> None:
    layer = MemSolve(MemSolveConfig(dim=32, num_heads=2, rank=8)).eval()
    x = torch.zeros(2, 9, 32, dtype=dtype)

    with pytest.raises(TypeError, match="does not support x with dtype"):
        layer(x, implementation="reference")


def test_symmetric_core_ablation_matches_triangular_coordinates_and_gradients() -> None:
    from memsolve.ball.reference import accretive_generator
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
    model = MemSolve(MemSolveConfig(8, 2, rank=4, scalar_complement=False)).double()
    assert not model.eta_raw.requires_grad
    assert torch.count_nonzero(model.complement()) == 0
    x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    output = model(x, implementation="reference")
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert model.eta_raw.grad is None
    default = MemSolve(MemSolveConfig(8, 2, rank=4)).double()
    with pytest.raises(RuntimeError, match="contract"):
        default.load_state_dict(model.state_dict())


@pytest.mark.parametrize("option", ["skew_coupling", "scalar_complement"])
def test_new_ablations_require_boolean_flags_and_explicit_reference(option: str) -> None:
    with pytest.raises(TypeError, match=option):
        MemSolveConfig(16, 1, rank=16, **{option: "false"})
    model = MemSolve(MemSolveConfig(16, 1, rank=16, **{option: False}))
    with pytest.raises(ValueError, match="requires"):
        model(torch.randn(1, 3, 16), implementation="cuda")

@pytest.mark.parametrize("mode", list(CoreMode))
def test_unrotated_operator_is_token_permutation_equivariant(mode: CoreMode) -> None:
    torch.manual_seed(194)
    layer = MemSolve(MemSolveConfig(16, 2, rank=5, core_mode=mode)).double()
    with torch.no_grad():
        if layer.core_base_raw is not None:
            layer.core_base_raw.normal_(0, 0.1)
    x = torch.randn(2, 11, 16, dtype=torch.float64)
    mask = torch.rand(2, 11) > 0.25
    permutation = torch.randperm(11)
    expected = layer(x, valid_mask=mask)[:, permutation]
    actual = layer(x[:, permutation], valid_mask=mask[:, permutation])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("removed_mode", ["dynamic", "static", "zero"])
def test_removed_mode_names_are_rejected(removed_mode):
    with pytest.raises(ValueError, match="core_mode must be one of"):
        MemSolveConfig(16, 2, rank=4, core_mode=removed_mode)


@pytest.mark.parametrize("mode", list(CoreMode))
def test_content_conditioned_generator_is_absent(mode):
    layer = MemSolve(MemSolveConfig(16, 2, rank=4, core_mode=mode))
    assert not hasattr(layer, "core_drive_weight")
    assert layer.get_extra_state()["version"] == 26
