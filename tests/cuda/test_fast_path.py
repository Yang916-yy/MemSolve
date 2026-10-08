from copy import deepcopy

import pytest
import torch

from memsolve import MemSolve, MemSolveConfig
from memsolve.ball import cuda
from memsolve.ball.reference import ridge_query_readout, head_rms_norm, split_qkv

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


@pytest.mark.parametrize('rank', [16, 32, 48, 64])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_packed_rope_and_conjugate_backward_match_reference(rank, dtype):
    from memsolve.ball.reference import axial_rotary_tables, rotary_qk

    torch.manual_seed(232)
    # Odd packed width exercises row alignment; transposed cotangent exercises
    # noncontiguous incoming gradients. Nonzero positions test both grid axes.
    x = torch.randn(2, 15, 2 * rank + 7, device='cuda', dtype=dtype, requires_grad=True)
    cos, sin = axial_rotary_tables(rank, (3, 5), device=x.device, dtype=torch.float32)
    actual = cuda.rotary_qk(x, 1, cos, sin)
    expected = rotary_qk(x, 1, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    e = torch.randn(2, x.shape[-1], 15, device='cuda', dtype=dtype).transpose(1, 2)
    torch.testing.assert_close(torch.autograd.grad(actual, x, e)[0],
                               torch.autograd.grad(expected, x, e)[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[..., 2 * rank:], x[..., 2 * rank:], rtol=0, atol=0)


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


def test_compiled_sigmoid_output_gate_matches_forward_and_both_adjoints():
    from memsolve.ball.reference import sigmoid_output_gate
    import torch._functorch.config as functorch_config

    torch.manual_seed(171)
    value = torch.randn(2, 37, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Cover useful gate tails as well as the unsaturated central region.
    logits = torch.linspace(-12, 12, value.numel(), device="cuda", dtype=torch.bfloat16)
    logits = logits.reshape_as(value).requires_grad_()
    compiled = torch.compile(sigmoid_output_gate, fullgraph=True,
                             options={"triton.cudagraphs": False, "emulate_precision_casts": True})
    cotangent = torch.randn_like(value)
    with functorch_config.patch(backward_pass_autocast="off"):
        actual = compiled(value, logits)
        actual_grads = torch.autograd.grad(actual, (value, logits), cotangent)
    expected = sigmoid_output_gate(value, logits)
    expected_grads = torch.autograd.grad(expected, (value, logits), cotangent)
    assert relative(actual, expected) < 0.001
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert relative(actual_grad, expected_grad) < 0.001


@pytest.mark.parametrize("rank", [16, 32, 48, 64])
@pytest.mark.parametrize("delta_scale", [0.0, 0.3])
@pytest.mark.parametrize("conv_dim", [1, 2])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_public_model_production_reference_and_cuda_gradients(rank, delta_scale, conv_dim, kernel_size, dtype):
    _require_cuda()
    torch.manual_seed(41)
    ref = MemSolve(MemSolveConfig(64, 2, rank, bias=True, qk_conv_dim=conv_dim,
                             qk_conv_kernel_size=kernel_size)).cuda()
    with torch.no_grad():
        ref.core_delta.normal_(std=delta_scale)
        ref.qk_conv.weight.add_(torch.randn_like(ref.qk_conv.weight) * .1)
        ref.gate_up.weight.normal_(std=.7)
    fast = deepcopy(ref)
    x = torch.randn(3, 35, 64, device="cuda", dtype=dtype)
    # Public masks may be strided even though the packed CUDA mask is dense.
    mask = (torch.rand(35, 3, device="cuda") > 0.2).transpose(0, 1)
    kwargs = {"spatial_shape": (5, 7)} if conv_dim == 2 else {}
    mask[-1] = False
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    yr = ref(a, mask, **kwargs)
    yc = fast(b, mask, implementation="cuda", **kwargs)
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


@pytest.mark.parametrize("rank", [16, 32, 48, 64])
def test_fused_gate_rounds_once_and_matches_all_adjoints(rank):
    from memsolve.ball.reference import sigmoid_output_gate

    torch.manual_seed(187)
    batch, length, heads, dim = 2, 37, 2, 32
    x = torch.randn(batch, length, heads * (2 * rank + dim),
                    device="cuda", dtype=torch.bfloat16, requires_grad=True)
    core = torch.randn(heads, rank, rank, device="cuda", requires_grad=True)
    w = torch.randn(dim, device="cuda", requires_grad=True)
    logits = torch.linspace(-12, 12, batch * length * heads * dim,
                            device="cuda", dtype=torch.bfloat16).reshape(batch, length, -1).requires_grad_()
    e = torch.randn_like(logits)
    fused = cuda.fast_mix(x, core, norm_weight=w, gate_logits=logits)
    # Independent reference: RMS and sigmoid are FP32 until the final store.
    q, k, v = split_qkv(x, heads, rank)
    raw = ridge_query_readout(q, k, v, core).transpose(1, 2)
    normalized = head_rms_norm(raw, w).reshape_as(logits)
    separate = sigmoid_output_gate(normalized, logits).to(torch.bfloat16)
    assert relative(fused, separate) < 0.001
    for actual, expected in zip(torch.autograd.grad(fused, (x, core, w, logits), e),
                                torch.autograd.grad(separate, (x, core, w, logits), e)):
        assert relative(actual, expected) < 0.001
    # A trainable gate alone must still select the autograd implementation.
    only_gate = logits.detach().requires_grad_()
    cuda.fast_mix(x.detach(), core.detach(), norm_weight=w.detach(),
                  gate_logits=only_gate).float().sum().backward()
    assert only_gate.grad is not None and torch.isfinite(only_gate.grad).all()


@pytest.mark.parametrize("rank", [16, 32, 48, 64])
def test_fp16_projected_readout_matches_oracle_and_retains_small_gradients(rank):
    torch.manual_seed(417)
    b, n, h, d = 2, 37, 2, 32
    packed = torch.randn(b, n, h*(2*rank+d), device="cuda", dtype=torch.bfloat16)
    packed[..., h*rank:2*h*rank] *= 8
    packed[-1] = 0
    packed.requires_grad_()
    core = torch.randn(h, rank, rank, device="cuda", requires_grad=True)
    gain = torch.randn(d, device="cuda", requires_grad=True)
    logits = torch.linspace(-12, 12, b*n*h*d, device="cuda", dtype=torch.bfloat16).reshape(b,n,h*d).requires_grad_()
    wo = (torch.randn(h*d, h*d, device="cuda")*.1).requires_grad_()
    bias = torch.randn(h*d, device="cuda", requires_grad=True)
    inputs = (packed, core, gain, logits, wo, bias)
    y = cuda.fast_mix(packed, core, norm_weight=gain, gate_logits=logits,
                     output_weight=wo, output_bias=bias, output_dtype=torch.float32)
    oracle = tuple(x.detach().double().requires_grad_() for x in inputs)
    q, k, v = split_qkv(oracle[0], h, rank)
    raw = ridge_query_readout(q,k,v,oracle[1]).transpose(1,2)
    norm = head_rms_norm(raw, oracle[2]).reshape_as(logits)
    ref = (norm * oracle[3].sigmoid()) @ oracle[4].T + oracle[5]
    assert relative(y, ref) < .001
    e = torch.randn_like(y)
    expected = torch.autograd.grad(ref, oracle, e.double())
    actual = torch.autograd.grad(y, inputs, e, retain_graph=True)
    for a, z in zip(actual, expected):
        assert relative(a, z) < .008
    # Upstream gradient scaling must not change the VJP, beyond BF16 rounding.
    for scale in (1e-3, 1e-7, 1e-9):
        grads = torch.autograd.grad(y, inputs, e*scale, retain_graph=True)
        for a, z in zip(grads, actual):
            assert torch.isfinite(a).all()
            assert relative(a.float()/scale, z) < .008
    # Output-projection-only gradients still require the custom autograd path.
    frozen = [x.detach() for x in inputs]
    frozen[-1].requires_grad_()
    cuda.fast_mix(frozen[0], frozen[1], norm_weight=frozen[2], gate_logits=frozen[3],
                  output_weight=frozen[4], output_bias=frozen[5], output_dtype=torch.float32).sum().backward()
    assert torch.equal(frozen[-1].grad, torch.full_like(bias, b*n))


@pytest.mark.parametrize("shape,kernel", [(None, 1), (None, 5), ((5, 7), 1), ((5, 7), 5)])
def test_native_local_pack_matches_centered_filter_rope_and_mask(shape, kernel):
    from memsolve.ball.reference import axial_rotary_tables, convolve_qk, rotary_qk

    torch.manual_seed(280)
    # QK and V share a partial 512-element block; exercise gapped/empty masks.
    x = torch.randn(2, 35, 39, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(2, 35, device="cuda") > .3
    mask[-1] = False
    x = torch.where(mask[..., None], x, 0).requires_grad_()
    w = torch.randn((32, 1) + (kernel,) * (1 if shape is None else 2),
                    device="cuda", requires_grad=True)
    cos, sin = (None, None) if shape is None else axial_rotary_tables(
        16, shape, device=x.device, dtype=torch.float32)
    expected = convolve_qk(x, w, shape)
    if shape is not None:
        expected = rotary_qk(expected, 1, cos, sin)
    expected = torch.where(mask[..., None], expected, 0)
    actual = cuda.local_qk(x, w, cos=cos, sin=sin, valid_mask=mask, spatial_shape=shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A strided incoming gradient must work in the packing adjoint.
    e = torch.randn(2, 39, 35, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    for a, b in zip(torch.autograd.grad(actual, (x, w), e), torch.autograd.grad(expected, (x, w), e)):
        assert relative(a, b) < 0.005


@pytest.mark.parametrize("conv_dim", [1, 2])
def test_cuda_graph_replays_changed_inputs_and_core_without_stale_map(conv_dim):
    _require_cuda()
    torch.manual_seed(27)
    model = MemSolve(MemSolveConfig(64, 2, 16, qk_conv_dim=conv_dim)).cuda()
    kwargs = {"spatial_shape": (5, 7)} if conv_dim == 2 else {}
    x = torch.randn(2, 35, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def step():
        y = model(x, implementation="cuda", **kwargs)
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
        model.qk_conv.weight.add_(torch.randn_like(model.qk_conv.weight) * .1)
        if conv_dim == 2:
            # Eager evaluation on a different same-length grid must not release
            # or overwrite the tables still referenced by the captured graph.
            model(x, implementation='cuda', spatial_shape=(7, 5))
    model.zero_grad(set_to_none=False)
    x.grad.zero_()
    graph.replay()
    replay = output.clone()
    gradients = [p.grad.clone() for p in model.parameters()]
    eager_model = deepcopy(model)
    eager_x = x.detach().clone().requires_grad_()
    eager = eager_model(eager_x, implementation="cuda", **kwargs)
    eager.float().square().mean().backward()
    torch.testing.assert_close(replay, eager, rtol=0, atol=0)
    for a, p in zip(gradients, eager_model.parameters()):
        torch.testing.assert_close(a, p.grad, rtol=0, atol=0)


def test_cuda_contract_and_public_rejections():
    _require_cuda()
    assert cuda._CUDA_CONTRACT_VERSION == 19
    model = MemSolve(MemSolveConfig(64, 2, 16)).cuda()
    with pytest.raises(TypeError):
        model(torch.randn(1, 7, 64, device="cuda"), implementation="cuda")
    with pytest.raises(ValueError):
        model(torch.randn(1, 7, 64, dtype=torch.bfloat16), implementation="cuda")
    with pytest.raises(TypeError):
        cuda.fast_mix(
            torch.randn(1, 7, 128, device="cuda"),
            model.core_delta + torch.eye(16, device="cuda"),
        )


@pytest.mark.parametrize("candidate", range(5))
@pytest.mark.parametrize("rank", [16, 32, 48, 64])
def test_launch_candidates_preserve_reference_forward_and_gradients(monkeypatch, candidate, rank):
    # Force each candidate, independent of timing noise: no fast but numerically
    # invalid schedule may enter the tuning search. Include ragged token tiles,
    # masks, wide values and strong keys, with both raw and RMS readouts.
    original = cuda._launch_candidates
    monkeypatch.setattr(cuda, "_LAUNCH_PLANS", {})

    def only_candidate(default):
        choices = original(default)
        return (choices[min(candidate, len(choices) - 1)],)

    monkeypatch.setattr(cuda, "_launch_candidates", only_candidate)
    test_direct_qkv_forward_and_all_adjoints_against_fp64(rank, 257, 80, 8, .3)
    test_fused_readout_norm_and_all_adjoints_match_fp64(rank, 257, 80, 8, 1)
    test_fused_gate_rounds_once_and_matches_all_adjoints(rank)
    test_fp16_projected_readout_matches_oracle_and_retains_small_gradients(rank)
    # The sensitive near-radial RMS VJP must pass even with a 16-token tile.
    if rank == 16:
        test_fused_readout_norm_and_all_adjoints_match_fp64(16, 1, 7, 1, 1)
        test_direct_qkv_forward_and_all_adjoints_against_fp64(16, 17, 7, 64, .3)


def test_resource_selection_uses_compiled_shared_memory_and_fails_explicitly(monkeypatch):
    torch.manual_seed(186)
    monkeypatch.setattr(cuda, "_LAUNCH_PLANS", {})
    real_resources = cuda._device_resources
    resources = real_resources(0)
    # Simulated hardware limit; actual kernels still execute on the real GPU.
    monkeypatch.setattr(cuda, "_device_resources", lambda index: (*resources[:4], 16384, *resources[5:]))
    x = torch.randn(2, 197, 192, device="cuda", dtype=torch.bfloat16)
    a, b = cuda._statistics(x, 1, 64, None)
    records = cuda.launch_report()
    assert any("rejected" in item for item in records[0]["candidates"])
    assert records[0]["selected"]["tokens"] < 256
    q, k, v = split_qkv(x.double(), 1, 64)
    torch.testing.assert_close(a.double(), k.mT @ k / 197, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b.double(), k.mT @ v / 197**.5, rtol=1e-4, atol=1e-6)
    cuda._LAUNCH_PLANS.clear()
    monkeypatch.setattr(cuda, "_device_resources", lambda index: (*resources[:4], 0, *resources[5:]))
    with pytest.raises(RuntimeError, match="No resource-compatible"):
        cuda._statistics(x, 1, 64, None)


def test_autotuned_launches_cache_without_tensors_and_replay_graph(monkeypatch):
    import json
    from triton import testing

    monkeypatch.setattr(cuda, "_LAUNCH_PLANS", {})
    torch.manual_seed(184)
    x = torch.randn(2, 37, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    core = torch.eye(16, device="cuda").repeat(2, 1, 1).requires_grad_()
    gain = torch.ones(16, device="cuda", requires_grad=True)

    def step():
        y = cuda.fast_mix(x, core, norm_weight=gain)
        return y, torch.autograd.grad(y, (x, core, gain), torch.ones_like(y))

    with cuda.autotune():
        expected, grads = step()
    expected = expected.detach()
    records = cuda.launch_report()
    assert len(records) == 4 and all(row["tuned"] for row in records)
    json.dumps(records)  # Reports/caches must not retain tensors or closures.
    assert all("registers" in row["candidates"][0] for row in records)

    def no_retuning(*args, **kwargs):
        raise AssertionError("a warmed shape must not benchmark again")

    monkeypatch.setattr(testing, "do_bench_cudagraph", no_retuning)
    with cuda.autotune():
        step()
    # Autograd warmup must share the capture stream on recent PyTorch versions.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual, actual_grads = step()
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(actual_grads, grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_launch_plans_are_isolated_between_devices(monkeypatch):
    torch.manual_seed(188)
    monkeypatch.setattr(cuda, "_LAUNCH_PLANS", {})
    first = torch.randn(1, 17, 48, device="cuda:0", dtype=torch.bfloat16)
    second = first.to("cuda:1")
    a = cuda._statistics(first, 1, 16, None)
    # Deliberately leave device 0 current while dispatching device 1's inputs.
    with torch.cuda.device(0):
        b = cuda._statistics(second, 1, 16, None)
        assert torch.cuda.current_device() == 0
    assert len(cuda.launch_report()) == 2
    assert {row["device"][0] for row in cuda.launch_report()} == {0, 1}
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y.to(x.device), rtol=1e-5, atol=1e-6)


def test_uncached_capture_requires_eager_warmup(monkeypatch):
    monkeypatch.setattr(cuda, "_LAUNCH_PLANS", {})
    x = torch.zeros(1, 17, 48, device="cuda", dtype=torch.bfloat16)
    # Exercise the pre-launch guard without invalidating a live CUDA capture.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eager forward/backward warmup"):
        cuda._statistics(x, 1, 16, None)
    assert not cuda.launch_report()


def test_launch_search_skips_candidate_workspace_oom(monkeypatch):
    torch.manual_seed(189)
    monkeypatch.setattr(cuda, "_LAUNCH_PLANS", {})
    launch = cuda._launch_token

    def limited_launch(kind, kernel, device, signature, default, prepare):
        def limited_prepare(config):
            if config.tokens == 256:
                raise torch.cuda.OutOfMemoryError("simulated candidate allocation failure")
            return prepare(config)
        return launch(kind, kernel, device, signature, default, limited_prepare)

    monkeypatch.setattr(cuda, "_launch_token", limited_launch)
    x = torch.randn(1, 129, 48, device="cuda", dtype=torch.bfloat16)
    gram, cross = cuda._statistics(x, 1, 16, None)
    q, k, v = split_qkv(x.double(), 1, 16)
    torch.testing.assert_close(gram.double(), k.mT @ k / 129, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cross.double(), k.mT @ v / 129**.5, rtol=1e-4, atol=1e-6)
    plan, = cuda.launch_report()
    assert plan["selected"]["tokens"] == 128
    assert "allocation failure" in plan["candidates"][0]["rejected"]


def test_graph_training_replays_fused_adamw_and_matches_eager():
    torch.manual_seed(113)
    model = MemSolve(MemSolveConfig(64, 2, 16)).cuda()
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
    model = MemSolve(MemSolveConfig(64, 2, 16)).cuda()
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
