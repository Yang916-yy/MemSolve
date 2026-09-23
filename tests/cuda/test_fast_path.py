from copy import deepcopy

import pytest
import torch

from ridgon import Ridgon, RidgonConfig
from ridgon.ball import cuda
from ridgon.ball.reference import ridge_query_readout, head_rms_norm, split_qkv

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _require_cuda():
    cuda.load()


def relative(actual, expected):
    return (
        actual.double() - expected.double()
    ).norm() / expected.double().norm().clamp_min(1e-12)


@pytest.mark.parametrize(
    "rank,length,dim,scale",
    [
        (16, 1, 7, 1),
        (16, 17, 7, 64),
        (16, 129, 64, 1),
        (32, 197, 64, 1),
        (48, 37, 80, 8),
        (64, 257, 16, 1),
    ],
)
@pytest.mark.parametrize("core_scale", [0.0, 0.3, 3.0])
def test_direct_qkv_forward_and_all_adjoints_against_fp64(
    rank, length, dim, scale, core_scale
):
    _require_cuda()
    torch.manual_seed(21)
    batch, heads = 3, 2
    packed = torch.randn(
        batch, length, heads * (2 * rank + dim), device="cuda", dtype=torch.bfloat16
    )
    packed[..., heads * rank : 2 * heads * rank] *= scale
    # Include gapped padding and an entirely empty sample.
    mask = torch.rand(batch, length, device="cuda") > 0.3
    mask[-1] = False
    packed = torch.where(mask[..., None], packed, 0).requires_grad_()
    counts = mask.sum(-1).float().clamp_min(1)
    raw = (torch.randn(heads, rank, rank, device="cuda") * core_scale).requires_grad_()
    actual = cuda.fast_mix(packed, raw, counts)
    p64 = packed.detach().double().requires_grad_()
    r64 = raw.detach().double().requires_grad_()
    q, k, v = split_qkv(p64, heads, rank)
    expected = (
        ridge_query_readout(q, k, v, r64, counts.double())
        .transpose(1, 2)
        .reshape_as(actual)
    )
    e = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * e).sum(), (packed, raw))
    ge = torch.autograd.grad((expected * e.double()).sum(), (p64, r64))
    assert relative(actual, expected) < 5e-4
    assert relative(ga[0], ge[0]) < 5e-3
    assert relative(ga[1], ge[1]) < 1e-3
    assert torch.count_nonzero(actual[-1]) == 0
    assert torch.count_nonzero(ga[0][-1]) == 0


@pytest.mark.parametrize("dim", [7, 64, 129])
@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_fla_group_rmsnorm_and_vjp_match_reference(dim, scale):
    torch.manual_seed(30)
    value = (torch.randn(2, 37, 3, dim, device="cuda") * scale).requires_grad_()
    weight = torch.randn(dim, device="cuda", requires_grad=True)
    output = cuda.head_rms_norm(value, weight)
    e = torch.randn_like(output)
    actual = torch.autograd.grad((output * e).sum(), (value, weight))
    v64 = value.detach().double().requires_grad_()
    w64 = weight.detach().double().requires_grad_()
    expected = head_rms_norm(v64, w64)
    gradients = torch.autograd.grad((expected * e.double()).sum(), (v64, w64))
    assert relative(output, expected) < 0.003
    for a, b in zip(actual, gradients):
        assert relative(a, b) < 2e-5


@pytest.mark.parametrize("rank", [16, 32, 48, 64])
@pytest.mark.parametrize("delta_scale", [0.0, 0.3])
def test_public_model_production_reference_and_cuda_gradients(rank, delta_scale):
    _require_cuda()
    torch.manual_seed(41)
    ref = Ridgon(RidgonConfig(64, 2, rank, bias=True)).cuda()
    with torch.no_grad():
        ref.core_delta.normal_(std=delta_scale)
    fast = deepcopy(ref)
    x = torch.randn(3, 37, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(3, 37, device="cuda") > 0.2
    mask[-1] = False
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    yr = ref(a, mask)
    yc = fast(b, mask, implementation="cuda")
    e = torch.randn_like(yr)
    (yr * e).sum().backward()
    (yc * e).sum().backward()
    assert relative(yc, yr) < 0.005
    assert relative(b.grad, a.grad) < 0.01
    for (_, p), (_, q) in zip(ref.named_parameters(), fast.named_parameters()):
        assert relative(q.grad, p.grad) < 0.015


@pytest.mark.parametrize(
    "rank,length,dim,key_scale,value_scale",
    [(16, 1, 7, 1, 1), (16, 17, 7, 64, 1), (32, 197, 64, 1, 1),
     (48, 37, 80, 8, 1), (64, 257, 16, 1, 1), (16, 129, 129, 1, 1e-3)],
)
def test_fused_readout_norm_and_all_adjoints_match_fp64(
    rank, length, dim, key_scale, value_scale,
):
    torch.manual_seed(312)
    batch, heads = 2, 2
    packed = torch.randn(batch, length, heads * (2 * rank + dim),
                         device="cuda", dtype=torch.bfloat16)
    packed[..., heads * rank:2 * heads * rank] *= key_scale
    packed[..., 2 * heads * rank:] *= value_scale
    mask = torch.rand(batch, length, device="cuda") > 0.3
    mask[0, 0] = True
    mask[-1] = False
    packed = torch.where(mask[..., None], packed, 0).requires_grad_()
    counts = mask.sum(-1).float().clamp_min(1)
    core = torch.randn(heads, rank, rank, device="cuda", requires_grad=True)
    weight = torch.randn(dim, device="cuda", requires_grad=True)
    actual = cuda.fast_mix(packed, core, counts, norm_weight=weight)
    p64, t64, w64 = (t.detach().double().requires_grad_() for t in (packed, core, weight))
    q, k, v = split_qkv(p64, heads, rank)
    raw = ridge_query_readout(q, k, v, t64, counts.double()).transpose(1, 2)
    expected = head_rms_norm(raw, w64).reshape_as(actual)
    e = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (packed, core, weight), e)
    ge = torch.autograd.grad(expected, (p64, t64, w64), e.double())
    assert relative(actual, expected) < 0.003
    for a, b, limit in zip(ga, ge, (0.005, 0.001, 0.001)):
        assert torch.isfinite(a).all()
        # One valid token makes core changes almost purely radial before RMSNorm.
        # Its near-zero core VJP needs an absolute FP32 rounding tolerance too.
        assert relative(a, b) < limit or (a.double() - b).abs().max() < 1e-6
    assert torch.count_nonzero(actual[-1]) == 0
    assert torch.count_nonzero(ga[0][-1]) == 0


def test_fused_readout_norm_supports_weight_only_gradients():
    packed = torch.randn(2, 5, 96, device="cuda", dtype=torch.bfloat16)
    core = torch.eye(16, device="cuda").repeat(2, 1, 1)
    weight = torch.ones(16, device="cuda", requires_grad=True)
    y = cuda.fast_mix(packed, core, norm_weight=weight)
    y.float().sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_cuda_graph_replays_changed_inputs_and_core_without_stale_map():
    _require_cuda()
    torch.manual_seed(27)
    model = Ridgon(RidgonConfig(64, 2, 16)).cuda()
    x = torch.randn(2, 37, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def step():
        y = model(x, implementation="cuda")
        y.float().square().mean().backward()
        return y

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            x.grad = None
            step()
    torch.cuda.current_stream().wait_stream(stream)
    model.zero_grad(set_to_none=True)
    x.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = step()
    with torch.no_grad():
        x.copy_(torch.randn_like(x))
        model.core_delta.add_(torch.randn_like(model.core_delta) * 0.05)
    model.zero_grad(set_to_none=False)
    x.grad.zero_()
    graph.replay()
    replay = output.clone()
    gradients = [p.grad.clone() for p in model.parameters()]
    eager_model = deepcopy(model)
    eager_x = x.detach().clone().requires_grad_()
    eager = eager_model(eager_x, implementation="cuda")
    eager.float().square().mean().backward()
    torch.testing.assert_close(replay, eager, rtol=0, atol=0)
    for a, p in zip(gradients, eager_model.parameters()):
        torch.testing.assert_close(a, p.grad, rtol=0, atol=0)


def test_cuda_contract_and_public_rejections():
    _require_cuda()
    assert cuda._CUDA_CONTRACT_VERSION == 17
    model = Ridgon(RidgonConfig(64, 2, 16)).cuda()
    with pytest.raises(TypeError):
        model(torch.randn(1, 7, 64, device="cuda"), implementation="cuda")
    with pytest.raises(ValueError):
        model(torch.randn(1, 7, 64, dtype=torch.bfloat16), implementation="cuda")
    with pytest.raises(TypeError):
        cuda.fast_mix(
            torch.randn(1, 7, 128, device="cuda"),
            model.core_delta + torch.eye(16, device="cuda"),
        )


def test_graph_training_replays_fused_adamw_and_matches_eager():
    torch.manual_seed(113)
    model = Ridgon(RidgonConfig(64, 2, 16)).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, fused=True, capturable=True,
    )
    x = torch.randn(2, 37, 64, device="cuda", dtype=torch.bfloat16)
    target = torch.randn_like(x)

    def step(m, opt):
        opt.zero_grad(set_to_none=False)
        loss = (m(x, implementation="cuda").float() - target).square().mean()
        loss.backward()
        opt.step()
        return loss

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step(model, optimizer)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = step(model, optimizer)
    eager = deepcopy(model)
    eager_opt = torch.optim.AdamW(
        eager.parameters(), lr=0.002, fused=True, capturable=True,
    )
    eager_opt.load_state_dict(deepcopy(optimizer.state_dict()))
    for _ in range(4):
        x.copy_(torch.randn_like(x))
        expected_loss = step(eager, eager_opt)
        graph.replay()
        torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)
        for p, q in zip(model.parameters(), eager.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
    assert model.core_delta.norm() > 0


def test_fused_amp_skipped_step_preserves_delta_and_moments():
    model = Ridgon(RidgonConfig(64, 2, 16)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    for skip in (False, True):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.core_delta * torch.randn_like(model.core_delta)).sum()
        scaler.scale(loss).backward()
        if skip:
            model.core_delta.grad.fill_(float("inf"))
            before = model.core_delta.detach().clone()
            state = deepcopy(optimizer.state_dict())
        scaler.step(optimizer)
        scaler.update()
        if skip:
            torch.testing.assert_close(model.core_delta, before, rtol=0, atol=0)
            for key, value in optimizer.state_dict()["state"].items():
                for name, tensor in value.items():
                    torch.testing.assert_close(tensor, state["state"][key][name], rtol=0, atol=0)






















def test_cuda_architecture_normalizes_sm121(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cuda.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cuda.torch.cuda,
        "get_device_capability",
        lambda device: (12, 1),
    )
    assert cuda._device_architecture() == 120


def test_cuda_architecture_rejects_turing_without_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cuda.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cuda.torch.cuda,
        "get_device_capability",
        lambda device: (7, 5),
    )
    with pytest.raises(RuntimeError, match="supports SM80"):
        cuda._device_architecture()
