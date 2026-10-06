from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import assembly101

pytestmark = pytest.mark.experiment


@pytest.mark.parametrize('rank', [16, 32])
@pytest.mark.parametrize('length,bucket', [(65, 128), (128, 256)])
def test_full_model_bucket_padding_preserves_real_frame_loss_and_gradients(rank, length, bucket):
    pytest.importorskip('ltc.model.ltcontext')
    from ltc.model.ltcontext import LTC
    from ltc.model.loss import get_loss_func
    import ltc.model.ltcontext as upstream_model
    cfg = assembly101.training_config(SimpleNamespace(data_root='unused', load_type='numpy',
        ltcontext=Path(upstream_model.__file__).parents[2], workers=0, seed=0))
    torch.manual_seed(19)
    model = assembly101.enable_sdpa(assembly101.replace_stage1(
        LTC(cfg.MODEL), rank, implementation='reference')).double().eval()
    loss_fn = get_loss_func(cfg)
    x = torch.randn(1, 2048, length, dtype=torch.float64)
    targets = torch.randint(202, (1, length))
    masks = torch.ones(1, 1, length, dtype=torch.bool)
    expected = model(x, masks)
    expected_loss = loss_fn(expected, targets)['loss']
    expected_loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    assembly101.enable_bucket_padding(model)
    padded = torch.nn.functional.pad(x, (0, bucket-length))
    mask = torch.nn.functional.pad(masks, (0, bucket-length), value=False)
    actual = model(padded, mask)[..., :length]
    actual_loss = loss_fn(actual, targets)['loss']
    actual_loss.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-8, atol=1e-9)
    for p, grad in zip(model.parameters(), grads):
        torch.testing.assert_close(p.grad, grad, rtol=1e-7, atol=1e-9)


def test_execution_upgrade_does_not_bypass_data_or_core_identity():
    old = {'rank': 16, 'audit_sha256': 'data', 'sources_sha256': {
        'experiments/assembly101.py': 'old', 'memsolve/ball/model.py': 'same'}}
    new = {**old, 'execution_backend': 'graph-compile', 'graph_buckets': [4096],
           'sources_sha256': {**old['sources_sha256'], 'experiments/assembly101.py': 'new'}}
    with pytest.raises(ValueError, match='identity mismatch'):
        assembly101.check_resume_identity(old, new)
    assembly101.check_resume_identity(old, new, True)
    for changed in ({**new, 'rank': 32}, {**new, 'audit_sha256': 'other'},
                    {**new, 'sources_sha256': {**new['sources_sha256'], 'memsolve/ball/model.py': 'other'}}):
        with pytest.raises(ValueError, match='identity mismatch'):
            assembly101.check_resume_identity(old, changed, True)


def test_parallel_audit_preserves_order_and_exact_hashes():
    samples = [{'video_name': str(i), 'features': torch.full((2048, 17+i), float(i)),
                'targets': torch.full((17+i,), i, dtype=torch.long)} for i in range(3)]
    assert list(assembly101.audit_records(samples, 2)) == list(assembly101.audit_records(samples, 1))


def test_sdpa_matches_upstream_attention_and_masked_gradients():
    pytest.importorskip('ltc.model.attention_utils')
    from ltc.model.attention_utils import ScaledDotProduct
    torch.manual_seed(7)
    original = ScaledDotProduct(dropout=.2).eval()
    fast = assembly101.TorchSDPA(.2).eval()
    inputs = [torch.randn(3, n, 16, dtype=torch.float64, requires_grad=True) for n in (7, 11, 11)]
    copies = [x.detach().clone().requires_grad_() for x in inputs]
    mask = torch.ones(3, 7, 11, dtype=torch.bool)
    mask[1, :, 8:] = False
    mask[2] = False
    expected = original(*inputs, att_mask=mask)
    actual = fast(*copies, att_mask=mask)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in zip(inputs, copies):
        torch.testing.assert_close(a.grad, b.grad)
    assert not actual[2].any()


def test_predictions_reject_wrong_length_and_non_class_values():
    for prediction in (np.array([0, 1]), np.array([0., 1., 2.]), np.array([0, 202, 1])):
        with pytest.raises(ValueError):
            assembly101.validate_prediction(prediction, 3)


def test_missing_data_fails_before_upstream_can_skip(tmp_path):
    pytest.importorskip('ltc.dataset.assembly101')
    split_dir = tmp_path / 'data/assembly101'
    split_dir.mkdir(parents=True)
    (split_dir / 'val.csv').write_text('action_type,video_id,view\nassembly,video,view\n')
    args = SimpleNamespace(ltcontext=tmp_path, data_root=tmp_path, load_type='numpy')
    with pytest.raises(FileNotFoundError, match='required files missing'):
        assembly101.dataset(args, 'val')


def test_evaluation_uses_macro_official_metrics_and_requires_full_coverage(tmp_path, monkeypatch):
    pytest.importorskip('ltc.utils.metrics')
    labels = tmp_path / 'labels'
    (labels / 'coarse_labels').mkdir(parents=True)
    (labels / 'actions.csv').write_text('action_id,action_cls\n0,a\n1,b\n')
    rows, samples = [], []
    for video, length in [('short', 2), ('long', 8)]:
        rows.append({'action_type': 'assembly', 'video_id': video, 'view': 'cam'})
        (labels / 'coarse_labels' / f'assembly_{video}.txt').write_text(f'0\t{length}\ta\t\n')
        samples.append({'video_name': f'assembly/{video}/cam', 'features': torch.zeros(2048, length),
                        'targets': torch.zeros(length, dtype=torch.long)})
        path = tmp_path / 'predictions' / f'assembly/{video}/cam'
        path.mkdir(parents=True)
        np.save(path / 'pred.npy', np.full(length, 0 if video == 'short' else 1, dtype=np.int64))
    split_path = tmp_path / 'val.csv'
    split_path.write_text('fixture')
    monkeypatch.setattr(assembly101, 'dataset', lambda args, split: (samples, split_path, labels, rows))
    args = SimpleNamespace(command='evaluate', load_type='numpy', predictions=tmp_path/'predictions',
                           output=tmp_path/'metrics.json')
    assembly101.run(args)
    result = json.loads(args.output.read_text())
    assert result['validation_metrics_percent'] == {name: 50.0 for name in assembly101.METRICS}
    assert result['checkpoint_selection_score'] == 2.5
    (args.predictions / 'assembly/long/cam/pred.npy').unlink()
    with pytest.raises(ValueError, match='coverage mismatch'):
        assembly101.run(args)


def test_pinned_loader_alignment_is_preserved(tmp_path, monkeypatch):
    pytest.importorskip('ltc.dataset.assembly101')
    split_dir = tmp_path / 'data/assembly101'
    labels = split_dir / 'coarse-annotations'
    (labels / 'coarse_labels').mkdir(parents=True)
    (split_dir / 'val.csv').write_text('action_type,video_id,view,video_end_frame\nassembly,video,cam,8\n')
    (labels / 'actions.csv').write_text('action_id,action_cls\n0,a\n1,b\n')
    (labels / 'coarse_labels/assembly_video.txt').write_text('0\t4\ta\t\n4\t8\tb\t\n')
    feature_dir = tmp_path / 'TSM_features/video/cam'
    feature_dir.mkdir(parents=True)
    np.save(feature_dir / 'features.npy', np.ones((2048, 4), dtype=np.float32))
    args = SimpleNamespace(ltcontext=tmp_path, data_root=tmp_path, load_type='numpy')
    monkeypatch.chdir(tmp_path)
    samples, *_ = assembly101.dataset(args, 'val')
    sample = samples[0]
    assembly101.validate_sample(sample)
    assert sample['targets'].tolist() == [0, 0, 1, 1]


def test_stage1_replacement_preserves_local_and_refinement_modules():
    pytest.importorskip('ltc.model.ltcontext')
    from ltc.model.ltcontext import LTC
    from ltc.config.defaults import _C
    cfg = _C.clone()
    cfg.MODEL.LTC.NUM_STAGES = 4
    model = LTC(cfg.MODEL)
    preserved = {name: module for name, module in model.named_modules()
                 if '.ltc_attn' not in name or not name.startswith('stage1.layers.')}
    assembly101.replace_stage1(model, 16, implementation='reference')
    current = dict(model.named_modules())
    assert all(current[name] is module for name, module in preserved.items())
    assert sum(isinstance(m, assembly101.Stage1MemSolve) for m in model.modules()) == 9
    assert all(layer.ltc_attn.mixer.config.rank == 16 for layer in model.stage1.layers)


@pytest.mark.parametrize('rank', [16, 32])
def test_stage1_padding_does_not_change_valid_outputs_or_gradients(rank):
    torch.manual_seed(0)
    adapter = assembly101.Stage1MemSolve(64, rank, implementation='reference').double()
    original = torch.randn(1, 64, 39, dtype=torch.float64, requires_grad=True)
    padded = torch.cat([original.detach(), torch.full((1, 64, 7), float('nan'), dtype=torch.float64)], -1)
    padded.requires_grad_()
    mask = torch.arange(46)[None, None] < 39
    expected = adapter(original)
    actual = adapter(padded, masks=mask)
    torch.testing.assert_close(actual[..., :39], expected)
    assert torch.count_nonzero(actual[..., 39:]) == 0
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(padded.grad[..., :39], original.grad)
    assert torch.count_nonzero(padded.grad[..., 39:]) == 0
    with pytest.raises(ValueError, match='stage-1 self-attention only'):
        adapter(original, v=original)
