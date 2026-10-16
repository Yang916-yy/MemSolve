"""LRA adapter masking, RoVE, routed softmax and numerical-failure contract."""
import copy
import math
import pytest
import torch
from torch.nn import functional as F
from experiments.sequence_attention import LRAAttention, LRA_ATTENTION, pooled_landmarks, routed_attention, masked_softmax
from experiments.train_transformers import NonFiniteTrainingError, require_finite
from memsolve.ball.reference import (
    _ieee_fp32_matmul,
    axial_rotary_position_factors,
    rotary_position_factors,
    rotate_content,
)


def factors(n, d, axial=False, device='cpu'):
    p = torch.arange(n, device=device)[None]
    if axial:
        return axial_rotary_position_factors(torch.stack((p % 4, p // 4), -1), head_dim=d)
    return rotary_position_factors(p, head_dim=d, base=10000., train_length=n, context_length=n, scaling='none')


@pytest.mark.parametrize('kind', sorted(LRA_ATTENTION - {'bigbird'}))
@pytest.mark.parametrize('axial', [False, True])
def test_adapters_masking_rotation_and_gradients(kind, axial):
    torch.manual_seed(19)
    module = LRAAttention(32, 4, kind, landmarks=4, features=16, routed_keys=3,
                          grid_shape=(4, 4) if axial else None).double()
    x = torch.randn(3, 16, 32, dtype=torch.float64, requires_grad=True)
    valid = torch.rand(3, 16) > .3
    valid[1] = False
    cos, sin = (f.double() for f in factors(16, 8, axial))
    y = module(x, valid, (cos, sin))
    # Arbitrary padding contents must not affect any valid readout.
    changed = x.detach().clone()
    changed[~valid] = 1e6
    torch.testing.assert_close(module(changed, valid, (cos, sin)), y)
    assert torch.count_nonzero(y[~valid]) == 0
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[~valid]) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_mita_joint_softmax_reference():
    torch.manual_seed(31)
    q, k, v = [torch.randn(2, 2, 12, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    valid = torch.rand(2, 12) > .25
    routers, rvalid = pooled_landmarks(q, valid, 3)
    actual = routed_attention(q, k, v, routers, rvalid, valid, 3)
    expected = torch.zeros_like(q)
    for b in range(2):
        for h in range(2):
            rk = routers[b, h, rvalid[b]]
            raw_k, raw_v = k[b, h, valid[b]], v[b, h, valid[b]]
            av = (rk @ raw_k.T / math.sqrt(8)).softmax(-1) @ raw_v
            for i in torch.where(valid[b])[0].tolist():
                query = q[b, h, i]
                expert = (rk @ query).argmax()
                chosen = (rk[expert] @ raw_k.T).topk(3).indices
                keys = torch.cat((rk / math.sqrt(8), raw_k[chosen]))
                values = torch.cat((av, raw_v[chosen]))
                expected[b, h, i] = (query @ keys.T / math.sqrt(8)).softmax(-1) @ values
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    got = torch.autograd.grad((actual * weight).sum(), (q, k, v), retain_graph=True)
    ref = torch.autograd.grad((expected * weight).sum(), (q, k, v))
    for a, e in zip(got, ref):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('stage', ['loss', 'gradient', 'parameter', 'evaluation'])
def test_nonfinite_failure_carries_stage_and_location(value, stage):
    with pytest.raises(NonFiniteTrainingError) as caught:
        require_finite([torch.ones(4), torch.tensor(value)], stage, epoch=3, step=7)
    assert caught.value.record == dict(state='nonfinite', stage=stage, epoch=3, step=7)


def test_finite_check_accepts_finite_values():
    require_finite([torch.randn(4), torch.zeros(2)], 'loss')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_mita_varlen_matches_reference_forward_backward():
    torch.manual_seed(33)
    tensors = [torch.randn(2, 2, 32, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    valid = torch.rand(2, 32, device='cuda') > .25
    q, k, v = tensors
    routers, rvalid = pooled_landmarks(q, valid, 4)
    y = routed_attention(q, k, v, routers, rvalid, valid, 4)
    # FP32 CUDA uses the SDPA reference path with the same BF16 router inputs.
    ref = routed_attention(q.float(), k.float(), v.float(), routers.float(), rvalid, valid, 4)
    assert (y.float() - ref).norm() / ref.norm() < .012
    weight = torch.randn_like(y)
    actual_grads = torch.autograd.grad((y * weight).sum(), tensors, retain_graph=True)
    ref_grads = torch.autograd.grad((ref * weight.float()).sum(), tensors)
    for a, b in zip(actual_grads, ref_grads):
        assert (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6) < .025


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('kind', sorted(LRA_ATTENTION - {'mita', 'bigbird'}))
def test_compiled_graph_adapter_updates(kind):
    torch.manual_seed(12)
    model = LRAAttention(128, 4, kind, landmarks=8, features=32, routed_keys=8).cuda()
    reference = copy.deepcopy(model)
    if kind == "local":
        model.prepare_local(64, "cuda")
    import torch._functorch.config as functorch_config
    functorch_config.backward_pass_autocast = 'off'
    model.compile(dynamic=False, options={'triton.cudagraphs': False, 'emulate_precision_casts': True})
    x = torch.randn(2, 64, 128, device='cuda')
    valid = torch.rand(2, 64, device='cuda') > .2
    rotation = factors(64, 32, device='cuda')
    def step(module):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = module(x, valid, rotation)
            loss = out.float().square().mean()
        loss.backward()
        return loss
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            step(model)
    torch.cuda.current_stream().wait_stream(stream)
    model.zero_grad(set_to_none=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = step(model)
    # Change inputs and replay: captured graph must not return stale outputs.
    x.normal_()
    for p in model.parameters():
        p.grad.zero_()
    graph.replay()
    expected = step(reference)
    assert (loss - expected).abs() / expected < .02
    for p, rp in zip(model.parameters(), reference.parameters()):
        assert torch.isfinite(p.grad).all()
        assert (p.grad.float() - rp.grad.float()).norm() / rp.grad.float().norm().clamp_min(1e-6) < .045


def test_nonfinite_entrypoint_writes_failure_and_exit86(monkeypatch, tmp_path):
    from experiments import train_transformers as runner
    def invalid(_args):
        raise runner.NonFiniteTrainingError('gradient', epoch=2, step=9)
    monkeypatch.setattr(runner, '_main', invalid)
    with pytest.raises(SystemExit) as error:
        runner.main(['--suite', 'lra', '--task', 'image', '--output', str(tmp_path),
                     '--fail-on-nonfinite'])
    assert error.value.code == 86
    import json
    record = json.loads((tmp_path / 'failure.json').read_text())
    assert record['state'] == 'nonfinite'
    assert record['stage'] == 'gradient'
    assert record['epoch'] == 2 and record['step'] == 9
    assert json.loads((tmp_path / 'status.json').read_text()) == record


@pytest.mark.parametrize('kind', sorted(LRA_ATTENTION - {'bigbird'}))
def test_lra_encoder_uses_new_adapter_with_rove(kind):
    from experiments.train_transformers import SequenceEncoder
    encoder = SequenceEncoder(input_kind='tokens', vocab_size=20, pad_token_id=0,
        max_length=16, dim=32, depth=1, num_heads=4, rank=4, mixer=kind,
        implementation='reference', mlp_ratio=2, dropout=0, bias=True,
        lra_baseline=True, position_encoding='rove_1d')
    assert isinstance(encoder.blocks[0].mixer, LRAAttention)
    x = torch.randint(1, 20, (2, 16))
    valid = torch.ones(2, 16, dtype=torch.bool)
    assert torch.isfinite(encoder(x, valid)).all()


@pytest.mark.parametrize('kind', ['performer', 'polaformer'])
def test_linear_contractions_match_explicit_pairwise_kernel(kind):
    torch.manual_seed(51)
    layer = LRAAttention(16, 2, kind, features=16).double()
    q, k, v = [torch.randn(2, 2, 12, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    valid = torch.rand(2, 12) > .3
    if kind == 'performer':
        def feature(x, query):
            z = x * 8 ** -.25
            projected = z @ layer.projection.T
            shift = projected.amax(-1, keepdim=True) if query else projected.masked_fill(
                ~valid[:, None, :, None], -torch.inf).amax((-2, -1), keepdim=True)
            return ((projected - z.square().sum(-1, keepdim=True)/2 - shift.detach()).exp() + 1e-4) / 4
        weights = feature(q, True) @ feature(k, False).transpose(-1, -2)
        weights = weights.masked_fill(~valid[:, None, None], 0)
        expected = weights @ v / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        actual = layer._performer(q, k, v, valid)
    else:
        exponent = 1 + 4 * layer.power.sigmoid()
        def feature(x):
            x = x / F.softplus(layer.scale)
            return torch.cat((x.relu().pow(exponent), (-x).relu().pow(exponent)), -1)
        qf, kf = feature(q), feature(k)
        opposite = torch.cat(qf.chunk(2, -1)[::-1], -1)
        values = v.chunk(2, -1)
        parts = []
        for query, value in zip((qf, opposite), values):
            weights = (query @ kf.transpose(-1, -2)).masked_fill(~valid[:, None, None], 0)
            eps = valid.sum(-1).clamp_min(1)[:, None, None, None] * 1e-6
            parts.append(weights @ value / (weights.sum(-1, keepdim=True) + eps))
        expected = torch.cat(parts, -1)
        actual = layer._pola(q, k, v, valid)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), (q, k, v), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * upstream).sum(), (q, k, v))
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize('kind', ['linear', 'local'])
def test_conventional_attention_dense_oracle(kind):
    torch.manual_seed(5)
    model = LRAAttention(16, 2, kind).double()
    q, k, v = [torch.randn(2, 2, 271, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    valid = torch.rand(2, 271) > .2
    if kind == 'linear':
        weights = (F.elu(q) + 1) @ (F.elu(k) + 1).transpose(-1, -2)
        weights = weights.masked_fill(~valid[:, None, None], 0)
        expected = (weights / (weights.sum(-1, keepdim=True) + 1e-6)) @ v
        actual = model._linear(q, k, v, valid)
    else:
        positions = torch.arange(271)
        delta = positions[:, None] - positions[None, :]
        mask = ((delta <= 128) & (delta > -128))[None, None] & valid[:, None, None]
        expected = masked_softmax(q @ k.transpose(-1, -2) / math.sqrt(8), mask) @ v
        actual = model._local(q, k, v, valid)
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * weight).sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad((expected * weight).sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('length', [1024, 2048, 4096])
def test_bigbird_flex_matches_huggingface(length):
    pytest.importorskip('transformers')
    from transformers import BigBirdConfig
    from transformers.models.big_bird.modeling_big_bird import BigBirdBlockSparseAttention, BigBirdModel
    with _ieee_fp32_matmul(torch.device('cuda')):
        module = LRAAttention(64, 2, 'bigbird').cuda()
        module.prepare_bigbird(length, 'cuda', seed=9)
        config = BigBirdConfig(hidden_size=64, num_attention_heads=2, block_size=64,
                               num_random_blocks=3, max_position_embeddings=4096)
        upstream = BigBirdBlockSparseAttention(config, seed=9).cuda().train()
        q, k, v = [torch.randn(2, 2, length, 32, device='cuda', requires_grad=True) for _ in range(3)]
        valid = torch.rand(2, length, device='cuda') > .25
        valid[0, -128:] = False
        blocked, band, fm, tm = BigBirdModel.create_masks_for_block_sparse_attn(valid.float(), 64)
        expected, _ = upstream.bigbird_block_sparse_attention(q, k, v, band, fm, tm,
            blocked, blocked, 2, 3, 32, 64, 64, 2, length, length, 9, None, None, False)
        expected = expected.transpose(1, 2)
        actual = module._bigbird(q, k, v, valid).masked_fill(~valid[:, None, :, None], 0)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        weight = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * weight).sum(), (q, k, v), retain_graph=True)
        ge = torch.autograd.grad((expected * weight).sum(), (q, k, v))
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=3e-5, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_local_sparse_boundary_and_empty_rows():
    with _ieee_fp32_matmul(torch.device('cuda')):
        model = LRAAttention(64, 2, 'local').cuda()
        reference = copy.deepcopy(model)
        model.prepare_local(384, 'cuda')
        x = torch.randn(3, 384, 64, device='cuda', requires_grad=True)
        valid = torch.rand(3, 384, device='cuda') > .25
        valid[1] = False
        valid[2, 128:] = False
        actual, expected = model(x, valid), reference(x, valid)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        ga = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        ge = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(ga, ge, atol=2e-5, rtol=2e-4)
