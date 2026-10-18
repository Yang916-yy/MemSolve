from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from memsolve.ball.reference import (
    accretive_equilibrium_mix,
    accretive_generator,
    bounded_complement,
    compact_equilibrium_diagnostics,
    qr_soft_frame,
    rotary_position_factors,
    rotate_content,
    tensor_core_linear,
    tensor_core_matmul,
)


pytestmark = pytest.mark.core


def test_rotary_yarn_uses_frequency_ramp_and_unit_amplitude() -> None:
    position = torch.arange(9)[None]
    cos, sin = rotary_position_factors(
        position, head_dim=64, base=10000., train_length=128,
        context_length=torch.tensor([128, 512]), scaling="yarn", dtype=torch.float64,
    )
    # beta=32 and beta=1 give transition indices [0,11] at D=64,N_train=128.
    pair = torch.arange(32, dtype=torch.float64)
    ramp = (pair / 11).clamp(0, 1)
    frequency = 10000. ** (-pair / 32)
    expected = position[..., None] * torch.stack((frequency, frequency * (1-ramp+ramp/4)))[:, None]
    torch.testing.assert_close(cos, expected.cos(), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(sin, expected.sin(), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(cos.square() + sin.square(), torch.ones_like(cos))
    assert cos.is_contiguous() and sin.is_contiguous()
    assert not cos.requires_grad and not sin.requires_grad
    low_precision = rotary_position_factors(
        position, head_dim=64, base=10000., train_length=128,
        context_length=512, scaling="yarn", dtype=torch.bfloat16,
    )
    assert all(table.dtype == torch.float32 for table in low_precision)


@pytest.mark.parametrize("equilibrium", [False, True])
@pytest.mark.parametrize(("length", "rank"), [(1, 16), (5, 32), (7, 48), (3, 64)])
def test_rotary_mix_matches_dense_orthogonal_conjugation_and_gradients(
    equilibrium: bool, length: int, rank: int,
) -> None:
    torch.manual_seed(501 + rank)
    batch, heads, width = 2, 2, 4
    relation = torch.randn(batch, heads, length, rank, dtype=torch.float64, requires_grad=True)
    content = torch.randn(batch, heads, length, width, dtype=torch.float64, requires_grad=True)
    raw = (0.15 * torch.randn(heads, rank, rank, dtype=torch.float64)).requires_grad_()
    eta = torch.tensor([.7, -.4], dtype=torch.float64, requires_grad=True)
    frame = qr_soft_frame(relation)
    generator = accretive_generator(raw) if equilibrium else None
    positions = torch.arange(length)[None] + torch.tensor([[0], [37]])
    cos, sin = rotary_position_factors(
        positions, head_dim=width, base=10000., train_length=4,
        context_length=torch.tensor([length, length * 2]), scaling="yarn", dtype=torch.float64,
    )
    state = frame.mT @ rotate_content(content, cos, sin)
    actual = accretive_equilibrium_mix(
        frame, generator, state, content, eta, rotary_cos=cos, rotary_sin=sin,
    )
    identity = torch.eye(rank, dtype=torch.float64)
    reflected = torch.linalg.solve(identity + generator, identity - generator) if equilibrium else torch.zeros_like(raw)
    token_map = eta[:, None, None] * torch.eye(length, dtype=torch.float64) + frame @ (
        reflected - eta[:, None, None] * identity
    ) @ frame.mT
    dense_rows = []
    for b in range(batch):
        rotations = []
        for n in range(length):
            blocks = [torch.stack((torch.stack((cos[b,n,p], -sin[b,n,p])),
                                   torch.stack((sin[b,n,p], cos[b,n,p]))))
                      for p in range(width//2)]
            rotations.append(torch.block_diag(*blocks))
        rotation = torch.block_diag(*rotations)
        dense_heads = []
        for h in range(heads):
            dense_map = rotation.mT @ torch.kron(token_map[b,h].contiguous(), torch.eye(width, dtype=torch.float64)) @ rotation
            dense_heads.append((dense_map @ content[b,h].flatten()).reshape(length, width))
            torch.testing.assert_close(torch.linalg.matrix_norm(dense_map, ord=2),
                                       torch.linalg.matrix_norm(token_map[b,h], ord=2))
            assert torch.linalg.matrix_norm(dense_map, ord=2) <= 1 + 1e-12
        dense_rows.append(torch.stack(dense_heads))
    expected = torch.stack(dense_rows)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    inputs = (relation, content, eta, raw) if equilibrium else (relation, content, eta)
    cotangent = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * cotangent).sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad((expected * cotangent).sum(), inputs)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("equilibrium", [False, True])
def test_rotary_zero_angles_and_local_content_are_unchanged(equilibrium: bool) -> None:
    torch.manual_seed(509)
    frame = qr_soft_frame(torch.randn(1, 2, 5, 3, dtype=torch.float64))
    content = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    eta = torch.tensor([.8, -.5], dtype=torch.float64)
    generator = accretive_generator(torch.randn(2, 3, 3, dtype=torch.float64)) if equilibrium else None
    cos, sin = rotary_position_factors(
        torch.zeros(1, 5, dtype=torch.long), head_dim=4, base=10000.,
        train_length=4, context_length=5, scaling="yarn", dtype=torch.float64,
    )
    state = frame.mT @ content
    actual = accretive_equilibrium_mix(frame, generator, state, content, eta,
                                      rotary_cos=cos, rotary_sin=sin)
    expected = accretive_equilibrium_mix(frame, generator, state, content, eta)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cos, sin = rotary_position_factors(
        torch.arange(5)[None], head_dim=4, base=10000.,
        train_length=4, context_length=5, scaling="yarn", dtype=torch.float64,
    )
    local_only = accretive_equilibrium_mix(torch.zeros_like(frame), generator,
        torch.zeros_like(state), content, eta, rotary_cos=cos, rotary_sin=sin)
    torch.testing.assert_close(local_only, eta[None, :, None, None] * content, rtol=0, atol=0)


def test_qr_soft_frame_preserves_gram_identity() -> None:
    torch.manual_seed(0)
    relation = torch.randn(3, 9, 5, dtype=torch.float64)
    frame = qr_soft_frame(relation)

    rank_eye = torch.eye(5, dtype=torch.float64)
    expected = relation @ torch.linalg.solve(
        rank_eye + relation.mT @ relation,
        relation.mT,
    )
    torch.testing.assert_close(frame @ frame.mT, expected)
    assert torch.all(torch.linalg.matrix_norm(frame, ord=2) <= 1.0 + 2e-12)


def test_qr_soft_frame_has_normal_range_gradients() -> None:
    torch.manual_seed(1)
    relation = torch.randn(2, 11, 4, dtype=torch.float32, requires_grad=True)
    frame = qr_soft_frame(relation)
    frame.square().mean().backward()

    assert torch.isfinite(frame).all()
    assert relation.grad is not None and torch.isfinite(relation.grad).all()


def test_tensor_core_matmul_keeps_fp64_as_the_autograd_oracle() -> None:
    torch.manual_seed(9)
    left = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(2, 3, 5, dtype=torch.float64)

    output = tensor_core_matmul(left, right)
    expected = left @ right
    gradients = torch.autograd.grad((output * upstream).sum(), (left, right))
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(),
        (left, right),
    )

    assert output.dtype is torch.float64
    torch.testing.assert_close(output, expected)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tensor_core_matmul_cuda_has_fp32_output_and_vjp() -> None:
    torch.manual_seed(10)
    left = torch.randn(2, 3, 4, 5, device="cuda", requires_grad=True)
    right = torch.randn(3, 5, 6, device="cuda", requires_grad=True)
    upstream = torch.randn(2, 3, 4, 6, device="cuda")

    output = tensor_core_matmul(left, right)
    gradients = torch.autograd.grad((output * upstream).sum(), (left, right))

    left_batches = left.detach().to(dtype=torch.bfloat16).reshape(-1, 4, 5)
    right_batches = right.detach().to(dtype=torch.bfloat16).expand(
        2,
        -1,
        -1,
        -1,
    ).reshape(-1, 5, 6)
    expected = torch.bmm(
        left_batches,
        right_batches,
        out_dtype=torch.float32,
    ).reshape(2, 3, 4, 6)
    upstream_batches = upstream.to(dtype=torch.bfloat16).reshape(-1, 4, 6)
    expected_left_gradient = torch.bmm(
        upstream_batches,
        right_batches.mT,
        out_dtype=torch.float32,
    ).reshape_as(left)
    expected_right_gradient = torch.bmm(
        left_batches.mT,
        upstream_batches,
        out_dtype=torch.float32,
    ).reshape(2, 3, 5, 6).sum(dim=0)

    assert output.dtype is torch.float32
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradients[0], expected_left_gradient, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradients[1], expected_right_gradient, rtol=0.0, atol=0.0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_tensor_core_linear_cuda_uses_flattened_bf16_vjp(
    input_dtype: torch.dtype,
) -> None:
    torch.manual_seed(12)
    value = torch.randn(
        2,
        3,
        5,
        device="cuda",
        dtype=input_dtype,
        requires_grad=True,
    )
    weight = torch.randn(7, 5, device="cuda", requires_grad=True)
    bias = torch.randn(7, device="cuda", requires_grad=True)
    upstream = torch.randn(2, 3, 7, device="cuda")

    actual = tensor_core_linear(value, weight, bias)
    gradients = torch.autograd.grad((actual * upstream).sum(), (value, weight, bias))

    value_bf16 = value.detach().to(dtype=torch.bfloat16).reshape(-1, 5)
    weight_bf16 = weight.detach().to(dtype=torch.bfloat16)
    gradient_bf16 = upstream.to(dtype=torch.bfloat16).reshape(-1, 7)
    expected = torch.mm(
        value_bf16,
        weight_bf16.mT,
        out_dtype=torch.float32,
    ).reshape_as(actual) + bias.detach()
    expected_value_gradient = torch.mm(
        gradient_bf16,
        weight_bf16,
        out_dtype=torch.float32,
    ).reshape_as(value).to(dtype=input_dtype)
    expected_weight_gradient = torch.mm(
        gradient_bf16.mT,
        value_bf16,
        out_dtype=torch.float32,
    )
    expected_bias_gradient = upstream.sum(dim=(0, 1))

    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        gradients[0], expected_value_gradient, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        gradients[1], expected_weight_gradient, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        gradients[2], expected_bias_gradient, rtol=0.0, atol=0.0
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", ((5,), (2, 3, 5)))
@pytest.mark.parametrize("with_bias", (False, True))
def test_tensor_core_linear_final_cast_preserves_fp32_input_vjp(
    shape: tuple[int, ...],
    with_bias: bool,
) -> None:
    torch.manual_seed(103)
    value = torch.randn(*shape, device="cuda", requires_grad=True)
    weight = torch.randn(7, 5, device="cuda", requires_grad=True)
    bias = torch.randn(7, device="cuda", requires_grad=True) if with_bias else None
    inputs = (value, weight) if bias is None else (value, weight, bias)
    actual = tensor_core_linear(value, weight, bias, output_dtype=torch.float16)
    upstream = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, inputs, upstream)

    # Reproduce the original FP32-output projection and external cast.
    expected = tensor_core_linear(value, weight, bias).half()
    expected_gradients = torch.autograd.grad(expected, inputs, upstream)
    assert actual.dtype is torch.float16
    assert gradients[0].dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-6)


def test_accretive_generator_matches_its_parameterization() -> None:
    torch.manual_seed(2)
    raw = torch.randn(2, 4, 4, dtype=torch.float64)
    generator = accretive_generator(raw)

    offset = math.log(math.expm1(1.0))
    diagonal = torch.diagonal(raw, dim1=-2, dim2=-1)
    factor = torch.tril(raw, diagonal=-1) + torch.diag_embed(
        torch.nn.functional.softplus(diagonal + offset)
    )
    upper = torch.triu(raw, diagonal=1)
    expected = factor @ factor.mT + upper - upper.mT

    torch.testing.assert_close(generator, expected)
    symmetric = 0.5 * (generator + generator.mT)
    assert torch.all(torch.linalg.eigvalsh(symmetric) > 0.0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_accretive_generator_cuda_forces_ieee_factor_gram_vjp() -> None:
    """The compact factor is full FP32 even when the ambient policy is TF32."""

    torch.manual_seed(22)
    raw_seed = 4.0 * torch.randn(2, 3, 16, 16, device="cuda")
    upstream = torch.randn_like(raw_seed)
    actual_raw = raw_seed.detach().clone().requires_grad_()
    expected_raw = raw_seed.detach().clone().requires_grad_()

    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = "tf32"
        actual = accretive_generator(actual_raw)
        actual_gradient = torch.autograd.grad((actual * upstream).sum(), actual_raw)[0]
        assert matmul.fp32_precision == "tf32"

        matmul.fp32_precision = "ieee"
        diagonal = torch.diagonal(expected_raw, dim1=-2, dim2=-1)
        factor = torch.tril(expected_raw, diagonal=-1) + torch.diag_embed(
            functional.softplus(diagonal + math.log(math.expm1(1.0)))
        )
        upper = torch.triu(expected_raw, diagonal=1)
        expected = factor @ factor.mT + upper - upper.mT
        expected_gradient = torch.autograd.grad(
            (expected * upstream).sum(), expected_raw
        )[0]
    finally:
        matmul.fp32_precision = previous

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=5e-6,
        atol=5e-6,
    )


def test_zero_raw_generator_is_identity() -> None:
    raw = torch.zeros(3, 4, 4, dtype=torch.float64)
    expected = torch.eye(4, dtype=torch.float64).expand_as(raw)
    torch.testing.assert_close(accretive_generator(raw), expected)




def test_equilibrium_mix_satisfies_direct_solve_equation() -> None:
    torch.manual_seed(4)
    batch, heads, length, rank, head_dim = 2, 3, 8, 4, 5
    frame = qr_soft_frame(
        torch.randn(batch, heads, length, rank, dtype=torch.float64)
    )
    raw = torch.randn(batch, heads, rank, rank, dtype=torch.float64)
    generator = accretive_generator(raw)
    content = torch.randn(batch, heads, length, head_dim, dtype=torch.float64)
    compact_state = frame.mT @ content
    eta = bounded_complement(torch.tensor([-0.7, 0.2, 0.9], dtype=torch.float64))

    output = accretive_equilibrium_mix(
        frame,
        generator,
        compact_state,
        content,
        eta,
    )

    rank_eye = torch.eye(rank, dtype=torch.float64)
    equilibrium = torch.linalg.solve(generator + rank_eye, compact_state)
    torch.testing.assert_close(
        (generator + rank_eye) @ equilibrium,
        compact_state,
    )
    eta_batch = eta.view(1, heads, 1, 1)
    expected = eta_batch * content + frame @ (
        2.0 * equilibrium - (1.0 + eta_batch) * compact_state
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("batched", [False, True])
def test_equilibrium_mix_accepts_shared_and_batched_generators(
    batched: bool,
) -> None:
    torch.manual_seed(5)
    batch, heads, length, rank, head_dim = 2, 3, 7, 4, 6
    frame = qr_soft_frame(torch.randn(batch, heads, length, rank))
    shared_generator = accretive_generator(torch.randn(heads, rank, rank))
    generator = (
        shared_generator.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        if batched
        else shared_generator
    )
    content = torch.randn(batch, heads, length, head_dim)
    compact_state = frame.mT @ content
    eta = bounded_complement(torch.linspace(-0.6, 0.9, heads))

    output = accretive_equilibrium_mix(
        frame,
        generator,
        compact_state,
        content,
        eta,
    )
    assert output.shape == content.shape
    assert torch.isfinite(output).all()


def test_complement_mode_uses_the_unscaled_complement_formula() -> None:
    torch.manual_seed(6)
    batch, heads, length, rank, head_dim = 2, 1, 7, 4, 3
    frame = qr_soft_frame(
        torch.randn(batch, heads, length, rank, dtype=torch.float64)
    )
    content = torch.randn(batch, heads, length, head_dim, dtype=torch.float64)
    compact_state = frame.mT @ content
    eta = torch.tensor([0.9], dtype=torch.float64)

    output = accretive_equilibrium_mix(
        frame,
        None,
        compact_state,
        content,
        eta,
    )
    expected = eta.view(1, heads, 1, 1) * (
        content - frame @ compact_state
    )
    torch.testing.assert_close(output, expected)


def test_direct_equilibrium_has_normal_range_gradients() -> None:
    torch.manual_seed(7)
    relation = torch.randn(2, 2, 8, 4, dtype=torch.float64, requires_grad=True)
    raw = torch.randn(2, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    content = torch.randn(2, 2, 8, 5, dtype=torch.float64, requires_grad=True)
    eta_raw = torch.tensor([-0.4, 0.6], dtype=torch.float64, requires_grad=True)

    frame = qr_soft_frame(relation)
    generator = accretive_generator(raw)
    compact_state = frame.mT @ content
    output = accretive_equilibrium_mix(
        frame,
        generator,
        compact_state,
        content,
        bounded_complement(eta_raw),
    )
    gradients = torch.autograd.grad(
        output.square().mean(),
        (relation, raw, content, eta_raw),
    )
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert torch.count_nonzero(gradients[1]) > 0


def test_compact_diagnostics_match_dense_operator_and_certificate_bounds() -> None:
    torch.manual_seed(70)
    batch, heads, length, rank, head_dim = 2, 3, 9, 4, 5
    frame = qr_soft_frame(
        torch.randn(batch, heads, length, rank, dtype=torch.float64)
    )
    raw = torch.randn(batch, heads, rank, rank, dtype=torch.float64)
    generator = accretive_generator(raw)
    content = torch.randn(batch, heads, length, head_dim, dtype=torch.float64)
    compact_state = frame.mT @ content
    eta = bounded_complement(torch.linspace(-0.4, 0.7, heads, dtype=torch.float64))
    adjoint_rhs = torch.randn_like(compact_state)

    diagnostics = compact_equilibrium_diagnostics(
        frame,
        generator,
        compact_state,
        eta,
        adjoint_rhs,
    )

    identity_r = torch.eye(rank, dtype=torch.float64)
    identity_n = torch.eye(length, dtype=torch.float64)
    system = identity_r + generator
    reflected = 2.0 * torch.linalg.solve(system, identity_r) - identity_r
    eta_batch = eta.view(1, heads, 1, 1)
    dense = eta_batch * identity_n + frame @ (reflected - eta_batch * identity_r) @ frame.mT
    expected_q = torch.linalg.matrix_norm(dense, ord=2)
    torch.testing.assert_close(diagnostics["q"], expected_q, rtol=2e-11, atol=2e-12)

    symmetric_system = 0.5 * (system + system.mT)
    expected_mu = torch.linalg.eigvalsh(symmetric_system)[..., 0]
    torch.testing.assert_close(diagnostics["mu"], expected_mu)
    assert torch.all(diagnostics["q"] < 1.0)
    assert torch.all(diagnostics["mu"] > 1.0)
    assert torch.all(diagnostics["state_bound_usage"] <= 1.0 + 2e-12)
    assert torch.all(diagnostics["adjoint_bound_usage"] <= 1.0 + 2e-12)


def test_compact_diagnostics_reject_sequence_shorter_than_rank() -> None:
    frame = qr_soft_frame(torch.randn(1, 1, 3, 4, dtype=torch.float64))
    generator = accretive_generator(torch.randn(1, 4, 4, dtype=torch.float64))
    compact_state = torch.randn(1, 1, 4, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="N >= rank R"):
        compact_equilibrium_diagnostics(
            frame,
            generator,
            compact_state,
            torch.tensor([0.5], dtype=torch.float64),
            torch.randn_like(compact_state),
        )


def test_direct_equilibrium_passes_first_and_second_order_gradcheck() -> None:
    torch.manual_seed(8)
    frame = qr_soft_frame(torch.randn(1, 1, 5, 3, dtype=torch.float64))
    content = torch.randn(1, 1, 5, 2, dtype=torch.float64)
    compact_state = frame.mT @ content
    eta = torch.tensor([0.35], dtype=torch.float64)
    raw = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)

    def mix(coordinates: torch.Tensor) -> torch.Tensor:
        return accretive_equilibrium_mix(
            frame,
            accretive_generator(coordinates),
            compact_state,
            content,
            eta,
        )

    assert torch.autograd.gradcheck(
        mix,
        (raw,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-3,
    )
    assert torch.autograd.gradgradcheck(
        mix,
        (raw,),
        eps=1e-6,
        atol=3e-5,
        rtol=3e-3,
    )


def test_bounded_complement_is_a_strict_interior_tanh() -> None:
    raw = torch.tensor([-3.0, 0.0, 4.0], dtype=torch.float32)
    actual = bounded_complement(raw)
    expected = (1.0 - torch.finfo(torch.float32).eps) * raw.tanh()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert torch.all(actual.abs() < 1.0)


def test_bounded_complement_retains_the_fp32_tanh_tail_gradient() -> None:
    raw = torch.tensor([-9.5, 9.5], dtype=torch.float32, requires_grad=True)
    complement = bounded_complement(raw)
    complement.sum().backward()

    assert torch.all(complement.abs() < 1.0)
    assert raw.grad is not None
    assert torch.all(torch.isfinite(raw.grad))
    assert torch.all(raw.grad.abs() > 0.0)


def test_reference_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="matrix dimensions"):
        qr_soft_frame(torch.randn(4))
    with pytest.raises(TypeError, match="floating-point"):
        qr_soft_frame(torch.ones(4, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="square"):
        accretive_generator(torch.randn(4, 3))

    frame = torch.randn(2, 3, 7, 4)
    content = torch.randn(2, 3, 7, 5)
    compact_state = frame.mT @ content
    eta = torch.ones(3)
    with pytest.raises(ValueError, match="batched generator"):
        accretive_equilibrium_mix(
            frame,
            torch.randn(1, 3, 4, 4),
            compact_state,
            content,
            eta,
        )
    with pytest.raises(ValueError, match="compact_state"):
        accretive_equilibrium_mix(
            frame,
            torch.randn(3, 4, 4),
            compact_state[..., :-1],
            content,
            eta,
        )
    with pytest.raises(ValueError, match="eta"):
        accretive_equilibrium_mix(
            frame,
            torch.randn(3, 4, 4),
            compact_state,
            content,
            torch.tensor(0.9),
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("amp_dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("with_bias", (False, True))
def test_bf16_linear_retains_wide_range_and_fp32_accumulation(amp_dtype, with_bias) -> None:
    # Both products and their sum exceed FP16 range. Ambient FP16 AMP must
    # neither recast the operands nor truncate the FP32 accumulated result.
    value = torch.full((2, 16), 131072.0, device="cuda", requires_grad=True)
    weight = torch.full((3, 16), 2.0, device="cuda", requires_grad=True)
    bias = torch.full((3,), 262144.0, device="cuda", requires_grad=True) if with_bias else None
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with torch.autocast("cuda", dtype=amp_dtype):
        output = tensor_core_linear(value, weight, bias)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == previous
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.full_like(output, 4456448.0 if with_bias else 4194304.0), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 6.0), rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 262144.0), rtol=0, atol=0)

    if bias is not None:
        torch.testing.assert_close(bias.grad, torch.full_like(bias, 2.0), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_linear_direct_bf16_store_keeps_fp32_accumulation() -> None:
    value = torch.full((2, 256), 32768.0, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(3, 256, device="cuda", requires_grad=True)
    output = tensor_core_linear(value, weight, output_dtype=torch.bfloat16)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.full_like(output, 8388608.0), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 3.0), rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 65536.0), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_linear_fp16_output_does_not_round_through_bf16() -> None:
    value = torch.ones(1, 2, device="cuda", dtype=torch.bfloat16)
    weight = torch.tensor([[1.0, 1.0 / 512]], device="cuda")
    output = tensor_core_linear(value, weight, output_dtype=torch.float16)
    torch.testing.assert_close(output, torch.full_like(output, 1.0 + 1.0 / 512), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_biased_linear_fused_store_preserves_fp32_bias(dtype) -> None:
    # Rounding this bias to BF16 before addition would erase the entire result.
    value = torch.ones(3, 1, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.ones(1, 1, device="cuda", requires_grad=True)
    bias = torch.tensor([-1.0 + 2.0**-9], device="cuda", requires_grad=True)
    output = tensor_core_linear(value, weight, bias, output_dtype=dtype)
    torch.testing.assert_close(output, torch.full_like(output, 2.0**-9), rtol=0, atol=0)
    output.float().sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 3), rtol=0, atol=0)
    torch.testing.assert_close(bias.grad, torch.full_like(bias, 3), rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(
            tensor_core_linear(value, weight, bias, output_dtype=dtype),
            output, rtol=0, atol=0,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_biased_linear_fused_store_handles_strides_and_tails(dtype) -> None:
    torch.manual_seed(123)
    value = torch.randn(2, 67, 70, device="cuda", dtype=dtype)[..., ::2].requires_grad_()
    weight = torch.randn(35, 51, device="cuda").T.requires_grad_()
    bias = torch.randn(102, device="cuda")[::2].requires_grad_()
    output = tensor_core_linear(value, weight, bias, output_dtype=dtype)
    expected = tensor_core_linear(value, weight, bias).to(dtype)
    # Different FP32 reduction groupings can straddle one output rounding bin.
    torch.testing.assert_close(output, expected, rtol=torch.finfo(dtype).eps, atol=0)
    upstream = torch.randn_like(output)
    inputs = (value, weight, bias)
    actual_grad = torch.autograd.grad(output, inputs, upstream)
    expected_grad = torch.autograd.grad(expected, inputs, upstream)
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('scale', [0.0, 1e-6, 0.3])
def test_base_correction_matches_resolvent_forward_and_all_gradients(shared, scale):
    torch.manual_seed(619)
    dtype = torch.float64
    frame = qr_soft_frame(torch.randn(3, 2, 9, 4, dtype=dtype)).detach().requires_grad_()
    content = torch.randn(3, 2, 9, 5, dtype=dtype, requires_grad=True)
    # Treat Z independently to check every argument's derivative.
    state = torch.randn(3, 2, 4, 5, dtype=dtype, requires_grad=True)
    raw = (scale * torch.randn((2, 4, 4) if shared else (3, 2, 4, 4), dtype=dtype)).requires_grad_()
    generator = accretive_generator(raw)
    eta = torch.tensor([0.9, -0.2], dtype=dtype, requires_grad=True)
    actual = accretive_equilibrium_mix(frame, generator, state, content, eta)
    expanded = generator[None].expand(3, -1, -1, -1) if shared else generator
    equilibrium = torch.linalg.solve(torch.eye(4, dtype=dtype) + expanded, state)
    e = eta[None, :, None, None]
    expected = e * content + frame @ (2 * equilibrium - (1 + e) * state)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(actual)
    variables = (frame, state, content, raw, eta)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), variables, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * upstream).sum(), variables)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
    if scale == 0:
        assert actual_grads[3].norm() > 0  # zero correction must not mean zero core gradient


def test_equilibrium_core_solve_is_shared_and_not_cached_across_updates(monkeypatch):
    calls = []
    solve = torch.linalg.solve
    def observed(system, rhs):
        calls.append((system.shape, rhs.shape))
        return solve(system, rhs)
    monkeypatch.setattr(torch.linalg, 'solve', observed)
    torch.manual_seed(19)
    frame = qr_soft_frame(torch.randn(7, 2, 9, 4, dtype=torch.float64))
    content = torch.randn(7, 2, 9, 5, dtype=torch.float64)
    state = frame.mT @ content
    raw = torch.zeros(2, 4, 4, dtype=torch.float64, requires_grad=True)
    eta = torch.full((2,), .9, dtype=torch.float64)
    first = accretive_equilibrium_mix(frame, accretive_generator(raw), state, content, eta)
    gradient, = torch.autograd.grad(first.square().sum(), raw)
    with torch.no_grad():
        raw.add_(gradient, alpha=-.01)
    second = accretive_equilibrium_mix(frame, accretive_generator(raw), state, content, eta)
    assert calls == [(torch.Size([2, 4, 4]), torch.Size([2, 4, 4]))] * 2
    assert not torch.allclose(first, second)


@pytest.mark.parametrize('length', [3, 9])
@pytest.mark.parametrize('mode', ['equilibrium', 'complement'])
def test_no_frame_identity_matches_original_resolvent_and_its_differential(length, mode):
    """Check algebra against the original readout, independent of CUDA precision."""
    torch.manual_seed(123)
    rank = 4
    a = (torch.randn(2, 2, length, rank, dtype=torch.float64) / length**0.5).requires_grad_()
    c = torch.randn(2, 2, length, 3, dtype=torch.float64, requires_grad=True)
    base = (torch.randn(2, rank, rank, dtype=torch.float64) * .1).requires_grad_()
    eta = torch.tensor([.1, -.2], dtype=torch.float64, requires_grad=True)
    eye = torch.eye(rank, dtype=torch.float64)

    def generator(z):
        if mode == 'complement':
            return eye
        return accretive_generator(base)

    p = qr_soft_frame(a)
    z = p.mT @ c
    k = generator(z)
    e = eta[None, :, None, None]
    original = 2 * p @ torch.linalg.solve(eye + k, z) - (1 + e) * (p @ z) + e * c
    lower = torch.linalg.cholesky(eye + a.mT @ a)
    z2 = torch.linalg.solve_triangular(lower, a.mT @ c, upper=False)
    k2 = generator(z2)
    delta = torch.linalg.solve(eye + k2, (eye - k2) @ z2)
    v = torch.linalg.solve_triangular(lower.mT, delta - e * z2, upper=True)
    no_frame = e * c + a @ v
    torch.testing.assert_close(original, no_frame, atol=1e-12, rtol=1e-10)
    # The common base is a regularized inverse, not an orthogonal complement.
    token_eye = torch.eye(length, dtype=torch.float64)
    base_inverse = torch.linalg.solve(token_eye + a @ a.mT, c)
    torch.testing.assert_close(c - p @ z, base_inverse, atol=1e-12, rtol=1e-10)
    upstream = torch.randn_like(original)
    arguments = (a, c, base, eta)
    first = torch.autograd.grad(original, arguments, upstream, allow_unused=True, retain_graph=True)
    second = torch.autograd.grad(no_frame, arguments, upstream, allow_unused=True)
    for left, right in zip(first, second):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-9)


def test_bounded_complement_has_the_tanh_derivative_at_zero():
    raw = torch.tensor([-1e-6, 0.0, 1e-6], dtype=torch.float64, requires_grad=True)
    actual = torch.autograd.grad(bounded_complement(raw).sum(), raw)[0]
    expected = (1 - torch.finfo(raw.dtype).eps) / raw.cosh().square()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_complement_preserves_values_near_zero(dtype):
    raw = torch.tensor([-1e-10, -1e-8, 0.0, 1e-8, 1e-10], dtype=dtype,
                       requires_grad=True)
    actual = bounded_complement(raw)
    # tanh(x)=x+O(x^3); these inputs must not disappear in 1-exp(-2x).
    scale = 1.0 - torch.finfo(dtype).eps
    torch.testing.assert_close(actual, scale * raw, rtol=2 * torch.finfo(dtype).eps, atol=0)
    derivative = torch.autograd.grad(actual.sum(), raw)[0]
    torch.testing.assert_close(derivative, torch.full_like(raw, scale),
                               rtol=2 * torch.finfo(dtype).eps, atol=0)


def test_bounded_complement_higher_derivatives_at_zero():
    raw = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    value = bounded_complement(raw)
    for expected in (1.0 - torch.finfo(raw.dtype).eps, 0.0,
                     -2.0 * (1.0 - torch.finfo(raw.dtype).eps)):
        value = torch.autograd.grad(value.sum(), raw, create_graph=True)[0]
        torch.testing.assert_close(value, torch.full_like(raw, expected), rtol=1e-14, atol=1e-14)
