import pytest
import torch
from ridgon.ball.reference import (
    tensor_core_matmul,
    tensor_core_linear,
    ridge_query_readout,
    head_rms_norm,
)

pytestmark = pytest.mark.core


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
    right_batches = (
        right.detach()
        .to(dtype=torch.bfloat16)
        .expand(
            2,
            -1,
            -1,
            -1,
        )
        .reshape(-1, 5, 6)
    )
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
    expected_right_gradient = (
        torch.bmm(
            left_batches.mT,
            upstream_batches,
            out_dtype=torch.float32,
        )
        .reshape(2, 3, 5, 6)
        .sum(dim=0)
    )

    assert output.dtype is torch.float32
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradients[0], expected_left_gradient, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        gradients[1], expected_right_gradient, rtol=0.0, atol=0.0
    )


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
    expected = (
        torch.mm(
            value_bf16,
            weight_bf16.mT,
            out_dtype=torch.float32,
        ).reshape_as(actual)
        + bias.detach()
    )
    expected_value_gradient = (
        torch.mm(
            gradient_bf16,
            weight_bf16,
            out_dtype=torch.float32,
        )
        .reshape_as(value)
        .to(dtype=input_dtype)
    )
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
    torch.testing.assert_close(gradients[2], expected_bias_gradient, rtol=0.0, atol=0.0)


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
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-6, atol=1e-6
        )








@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("amp_dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("with_bias", (False, True))
def test_bf16_linear_retains_wide_range_and_fp32_accumulation(
    amp_dtype, with_bias
) -> None:
    # Both products and their sum exceed FP16 range. Ambient FP16 AMP must
    # neither recast the operands nor truncate the FP32 accumulated result.
    value = torch.full((2, 16), 131072.0, device="cuda", requires_grad=True)
    weight = torch.full((3, 16), 2.0, device="cuda", requires_grad=True)
    bias = (
        torch.full((3,), 262144.0, device="cuda", requires_grad=True)
        if with_bias
        else None
    )
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with torch.autocast("cuda", dtype=amp_dtype):
        output = tensor_core_linear(value, weight, bias)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == previous
    assert output.dtype == torch.float32
    torch.testing.assert_close(
        output,
        torch.full_like(output, 4456448.0 if with_bias else 4194304.0),
        rtol=0,
        atol=0,
    )
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 6.0), rtol=0, atol=0)
    torch.testing.assert_close(
        weight.grad, torch.full_like(weight, 262144.0), rtol=0, atol=0
    )

    if bias is not None:
        torch.testing.assert_close(
            bias.grad, torch.full_like(bias, 2.0), rtol=0, atol=0
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_linear_direct_bf16_store_keeps_fp32_accumulation() -> None:
    value = torch.full(
        (2, 256), 32768.0, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.ones(3, 256, device="cuda", requires_grad=True)
    output = tensor_core_linear(value, weight, output_dtype=torch.bfloat16)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output, torch.full_like(output, 8388608.0), rtol=0, atol=0
    )
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 3.0), rtol=0, atol=0)
    torch.testing.assert_close(
        weight.grad, torch.full_like(weight, 65536.0), rtol=0, atol=0
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_linear_fp16_output_does_not_round_through_bf16() -> None:
    value = torch.ones(1, 2, device="cuda", dtype=torch.bfloat16)
    weight = torch.tensor([[1.0, 1.0 / 512]], device="cuda")
    output = tensor_core_linear(value, weight, output_dtype=torch.float16)
    torch.testing.assert_close(
        output, torch.full_like(output, 1.0 + 1.0 / 512), rtol=0, atol=0
    )


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
            output,
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_biased_linear_fused_store_handles_strides_and_tails(dtype) -> None:
    torch.manual_seed(123)
    value = torch.randn(2, 67, 70, device="cuda", dtype=dtype)[
        ..., ::2
    ].requires_grad_()
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_biased_linear_covers_full_and_partial_tile_groups(dtype) -> None:
    torch.manual_seed(127)
    # Ten row tiles and three column tiles exercise full/partial groups.
    # Dyadic inputs make the FP32 sum exact, so this detects omitted, repeated
    # or misaddressed tiles without conflating GEMM reduction roundoff.
    value = (torch.randint(-16, 17, (2, 577, 70), device="cuda").to(dtype) / 8)[..., ::2]
    weight = (torch.randint(-16, 17, (35, 259), device="cuda").float() / 8).T
    bias = (torch.randint(-16, 17, (518,), device="cuda").float() / 64)[::2]
    output = tensor_core_linear(value, weight, bias, output_dtype=dtype)
    expected = (value.double() @ weight.double().T + bias.double()).to(dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("length,rank", [(2, 4), (7, 4), (4, 4)])
def test_direct_readout_equals_raw_coordinate_system_and_gradients(length, rank):
    torch.manual_seed(9)
    q = torch.randn(2, 2, length, rank, dtype=torch.float64, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(2, 2, length, 5, dtype=torch.float64, requires_grad=True)
    raw = (torch.randn(2, rank, rank, dtype=torch.float64) * 0.2).requires_grad_()
    identity = torch.eye(rank, dtype=torch.float64)
    ak, aq = k / length**0.5, q / length**0.5
    r = torch.linalg.cholesky(identity + ak.mT @ ak).mT
    # Independent ridge memory plus a statistic-conditioned query transform.
    memory = torch.linalg.solve(identity + ak.mT @ ak, ak.mT @ v)
    query = torch.linalg.solve(r.mT, aq.mT).mT @ raw @ r
    expected = query @ memory
    actual = ridge_query_readout(q, k, v, raw)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    e = torch.randn_like(actual)
    da = torch.autograd.grad((actual * e).sum(), (q, k, v, raw), retain_graph=True)
    de = torch.autograd.grad((expected * e).sum(), (q, k, v, raw))
    for a, b in zip(da, de):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_independent_queries_do_not_change_other_rows_or_memory():
    torch.manual_seed(12)
    q = torch.randn(1, 1, 5, 3, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(1, 1, 5, 4, dtype=torch.float64)
    raw = torch.randn(1, 3, 3, dtype=torch.float64) * 0.2
    base = ridge_query_readout(q, k, v, raw)
    q2 = q.clone()
    q2[:, :, 0] += 2
    changed = ridge_query_readout(q2, k, v, raw)
    torch.testing.assert_close(changed[:, :, 1:], base[:, :, 1:], rtol=0, atol=0)
    assert not torch.equal(changed[:, :, 0], base[:, :, 0])


def test_query_key_value_and_core_gradcheck():
    torch.manual_seed(8)
    inputs = (
        torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True),
        (torch.randn(1, 2, 2, dtype=torch.float64) * 0.1).requires_grad_(),
    )
    assert torch.autograd.gradcheck(ridge_query_readout, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(ridge_query_readout, inputs, fast_mode=True)


def test_direct_map_accepts_singular_core_and_exact_query_scale_gauge():
    torch.manual_seed(101)
    q = torch.randn(2, 2, 9, 4, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(2, 2, 9, 6, dtype=torch.float64)
    mapping = torch.randn(2, 4, 4, dtype=torch.float64)
    mapping[:, -1] = 0  # A singular T is valid: only the SPD key Gram is solved.
    expected = ridge_query_readout(q, k, v, mapping)
    scale = mapping.norm(dim=(-2, -1), keepdim=True) / 2  # radius sqrt(4) = 2
    actual = ridge_query_readout(q * scale, k, v, mapping / scale)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(ridge_query_readout(q, k, v, torch.zeros_like(mapping))) == 0


def test_direct_core_updates_are_observed_without_a_core_solve(monkeypatch):
    calls = []
    solve = torch.linalg.solve_ex

    def capture(a, b, *args, **kwargs):
        calls.append(tuple(a.shape))
        return solve(a, b, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve_ex", capture)
    q = torch.randn(5, 2, 7, 3, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(5, 2, 7, 4, dtype=torch.float64)
    raw = torch.zeros(2, 3, 3, dtype=torch.float64)
    first = ridge_query_readout(q, k, v, raw)
    raw[:, 0, 1] = 0.4
    second = ridge_query_readout(q, k, v, raw)
    assert calls == []
    assert not torch.equal(first, second)


@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_head_rmsnorm_does_not_mix_tokens_or_heads(scale):
    x = (torch.randn(2, 7, 3, 5, dtype=torch.float64) * scale).requires_grad_()
    w = torch.randn(5, dtype=torch.float64, requires_grad=True)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(head_rms_norm(x, w), expected)
    assert torch.autograd.gradcheck(head_rms_norm, (x, w), fast_mode=True)
    probe = torch.randn_like(x)
    gain_grad, = torch.autograd.grad(head_rms_norm(x, w), w, probe)
    normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(gain_grad, (normalized * probe).sum((0, 1, 2)))
