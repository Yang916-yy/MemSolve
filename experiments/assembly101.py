"""Assembly101 data audit, stage-1 Ridgon training and LTContext evaluation.

The external LTContext checkout owns data alignment and metric definitions.
No upstream code or operator mathematics is copied into this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import random
import time
import tempfile
import zipfile
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch

from ridgon import Ridgon, RidgonConfig

LTC_REVISION = 'ac74722b00b52b7eb9eb3d6fa8600c762e6f369b'
METRICS = ('MoF', 'Edit', 'F1@10', 'F1@25', 'F1@50')


class Stage1Ridgon(torch.nn.Module):
    """Adapt LTContext's channel-first self-attention interface to public Ridgon."""

    def __init__(self, dim: int, rank: int, implementation: str = 'cuda'):
        super().__init__()
        self.mixer = Ridgon(RidgonConfig(dim=dim, num_heads=1, rank=rank, bias=True))
        self.implementation = implementation

    def forward(self, qk, v=None, masks=None):
        if v is not None:
            raise ValueError('Stage1Ridgon replaces stage-1 self-attention only')
        valid = None if masks is None else masks[:, 0, :]
        inputs = qk.transpose(1, 2).contiguous()
        # Native activations are BF16; upstream layers and all parameters stay FP32.
        if self.implementation == 'cuda':
            inputs = inputs.to(torch.bfloat16)
        output = self.mixer(inputs, valid_mask=valid, implementation=self.implementation)
        return output.transpose(1, 2).to(qk.dtype).contiguous()


class TorchSDPA(torch.nn.Module):
    """Execute upstream attention with PyTorch SDPA, preserving its geometry.

    This evaluates dropout(softmax(Q K^T / sqrt(d) + mask)) V, with
    the original attention dropout probability and FP32 activations. LTContext
    discards the returned attention matrix; no dense matrix is materialized here.
    """

    def __init__(self, dropout):
        super().__init__()
        self.attn_drop = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, att_mask=None, position_bias=None, return_attn_matrix=False):
        if position_bias is not None:
            raise ValueError('LTContext SDPA adapter does not use position bias')
        y = torch.nn.functional.scaled_dot_product_attention(
            q[:, None], k[:, None], v[:, None],
            attn_mask=None if att_mask is None else att_mask[:, None],
            dropout_p=self.attn_drop.p if self.training else 0.0)[:, 0]
        return (y, None) if return_attn_matrix else y


def enable_sdpa(model):
    from ltc.model.attention_utils import ScaledDotProduct
    for module in model.modules():
        original = getattr(module, 'attention', None)
        if isinstance(original, ScaledDotProduct):
            module.attention = TorchSDPA(original.attn_drop.p).train(original.training)
    return model


def replace_stage1(model, rank: int, implementation: str = 'cuda'):
    """Replace only the nine stage-1 long-range branches, in place."""
    for layer in model.stage1.layers:
        dim = layer.out_linear.in_channels
        layer.ltc_attn = Stage1Ridgon(dim, rank, implementation)
    return model


def _mask_first_block_input(module, inputs):
    # A padded timestep must enter the first temporal convolution as zero,
    # just like its implicit padding in the original, unpadded sequence.
    return (inputs[0] * inputs[1], *inputs[1:])


def _device_pad_sequence(original, x, pad_size, masks=None):
    if masks is None:
        masks = torch.ones((x.shape[0], 1, x.shape[-1]), dtype=torch.bool, device=x.device)
    return original(x, pad_size, masks)


def enable_bucket_padding(model):
    """Preserve valid-frame computations when padding a full video to a bucket.

    All subsequent blocks already mask their output. Only the input projection
    can introduce a nonzero bias into padding before the first convolution.
    Attention groups keep their original stride; padded keys remain masked.
    This requires the official Assembly101 Identity normalization setting.
    """
    import ltc.model.attention_utils as attention_utils
    for stage in (model.stage1, *model.stages):
        if any(not isinstance(block.instance_norm, torch.nn.Identity) for block in stage.layers):
            raise ValueError('Bucket execution requires Identity instance normalization')
        stage.layers[0].register_forward_pre_hook(_mask_first_block_input)
    if not isinstance(attention_utils.pad_sequence, partial):
        attention_utils.pad_sequence = partial(_device_pad_sequence, attention_utils.pad_sequence)
    return model


class _GraphModule(torch.nn.Module):
    """Give each bucket its own graphed forward, sharing the original weights."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, masks):
        return self.model(features, masks)


def _training_forward(module, eager, compiled, *args, **kwargs):
    # Validation uses original lengths and must not compile every distinct one.
    return (compiled if module.training else eager)(*args, **kwargs)


class BucketExecution:
    """Independent graph pools for shuffled lengths; eager loss on real frames.

    Warmup/capture never update weights and restore RNG state. Graphs are rebuilt
    on resume; model/optimizer state dictionaries retain their original names.
    """

    def __init__(self, model, backend, buckets):
        self.model = enable_bucket_padding(model)
        self.buckets = tuple(sorted(set(buckets)))
        if not self.buckets or self.buckets[0] < 1:
            raise ValueError('Graph buckets must be positive lengths')
        self.graphs = {}
        self.compiled = model
        if backend == 'graph-compile':
            # 36 independently compiled convolution regions x 8 buckets exceed
            # Dynamo's default accumulated limit of 256 for this code object.
            torch._dynamo.config.accumulated_recompile_limit = max(
                torch._dynamo.config.accumulated_recompile_limit, 1024)
            # Fuse convolution/GELU/masking while retaining the efficient eager
            # implementations of overlapping-window attention and native Ridgon.
            # Compiling whole attention blocks increased compile time and did
            # not consistently improve replay throughput on the target GPU.
            for stage in (model.stage1, *model.stages):
                for block in stage.layers:
                    module = block.dilated_conv
                    eager = module.forward
                    compiled = torch.compile(eager, dynamic=None, fullgraph=True,
                        isolate_recompiles=True, recompile_limit=16, options={
                        'triton.cudagraphs': False, 'fallback_random': True})
                    module.forward = partial(_training_forward, module, eager, compiled)

    def __call__(self, features, masks):
        length = features.shape[-1]
        bucket = next((n for n in self.buckets if n >= length), None)
        if bucket is None:
            raise ValueError(f'Sequence length {length} exceeds largest graph bucket')
        features = torch.nn.functional.pad(features, (0, bucket-length))
        masks = torch.nn.functional.pad(masks, (0, bucket-length), value=False)
        if bucket not in self.graphs:
            started = time.perf_counter()
            device = features.device.index
            # Compiled backward kernels must be materialized outside capture.
            with torch.random.fork_rng(devices=[device]):
                proxy = _GraphModule(self.compiled).train()
                self.graphs[bucket] = torch.cuda.make_graphed_callables(
                    proxy, (features, masks), num_warmup_iters=3)
            print(json.dumps({'graph_captured': bucket, 'seconds': time.perf_counter()-started,
                              'reserved_bytes': torch.cuda.memory_reserved()}), flush=True)
        return self.graphs[bucket](features, masks)[..., :length]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def local_lock(path: Path):
    """Coordinate this host without depending on the NFS lock manager."""
    directory = Path(tempfile.gettempdir()) / 'ridgon-assembly101-locks'
    directory.mkdir(exist_ok=True)
    name = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    return (directory / name).open('w')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample(sample: dict) -> None:
    features, targets = sample['features'], sample['targets']
    if features.ndim != 2 or features.shape[0] != 2048 or features.shape[1] == 0:
        raise ValueError(f"Invalid feature shape: {sample['video_name']}: {features.shape}")
    if targets.ndim != 1 or targets.numel() != features.shape[1]:
        raise ValueError(f"Target/feature mismatch: {sample['video_name']}")
    if not torch.isfinite(features).all():
        raise ValueError(f"Nonfinite features: {sample['video_name']}")
    if targets.min() < 0 or targets.max() >= 202:
        raise ValueError(f"Invalid action IDs: {sample['video_name']}")


def validate_prediction(prediction: np.ndarray, length: int) -> None:
    if prediction.shape != (length,) or prediction.dtype.kind not in 'iu':
        raise ValueError(f'Expected integer predictions [{length}], got {prediction.shape}/{prediction.dtype}')
    if np.any(prediction < 0) or np.any(prediction >= 202):
        raise ValueError('Prediction action IDs must be in [0, 201]')


@contextmanager
def upstream(root: Path):
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != LTC_REVISION:
        raise ValueError(f'Expected LTContext {LTC_REVISION}, found {revision}')
    changes = subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--', 'ltc', 'data/assembly101'], text=True)
    if changes:
        raise ValueError('LTContext data/model/metric sources have tracked changes; use the pinned clean checkout')
    previous = Path.cwd()
    sys.path.insert(0, str(root))
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)
        sys.path.remove(str(root))


def dataset(args, split):
    from yacs.config import CfgNode
    from ltc.dataset.assembly101 import Assembly101
    split_path = args.ltcontext / 'data/assembly101' / f'{split}.csv'
    with split_path.open() as stream:
        rows = list(csv.DictReader(stream))
    keys = [(r['action_type'], r['video_id'], r['view']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate records in {split_path}')
    labels = args.ltcontext / 'data/assembly101/coarse-annotations'
    needed = {labels / 'actions.csv'} | {
        labels / 'coarse_labels' / f"{r['action_type']}_{r['video_id']}.txt" for r in rows
    }
    if args.load_type == 'numpy':
        needed |= {args.data_root / 'TSM_features' / r['video_id'] / r['view'] / 'features.npy' for r in rows}
    missing = sorted(str(p) for p in needed if not p.is_file())
    if missing:
        raise FileNotFoundError(f'{len(missing)} required files missing; first entries: {missing[:5]}')
    cfg = CfgNode({'DATA': {'LOAD_TYPE': args.load_type, 'PATH_TO_DATA_DIR': str(args.data_root),
                          'FRAME_SAMPLING_RATE': 1, 'DATA_FRACTION': 1.0}})
    result = Assembly101(cfg, split)
    if len(result) != len(rows):
        raise ValueError(f'{split}: upstream silently excluded {len(rows)-len(result)} records')
    return result, split_path, labels, rows


def _audit_sample(sample):
    validate_sample(sample)
    return {'video_name': sample['video_name'], 'frames': sample['targets'].numel(),
            'targets_sha256': hashlib.sha256(sample['targets'].numpy().tobytes()).hexdigest(),
            'features_sha256': hashlib.sha256(sample['features'].numpy().tobytes()).hexdigest()}


def _audit_worker_init(samples):
    global _audit_dataset
    torch.set_num_threads(1)
    _audit_dataset = samples


def _audit_worker(index):
    return _audit_sample(_audit_dataset[index])


def audit_records(samples, workers):
    if workers <= 1:
        for index in range(len(samples)):
            yield _audit_sample(samples[index])
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        # Each worker reads and hashes locally; only small records cross processes.
        # Spawn avoids inheriting a live OpenMP thread pool or LMDB environment.
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_audit_worker_init, initargs=(samples,)) as pool:
            yield from pool.map(_audit_worker, range(len(samples)), chunksize=8)


def run(args):
    from ltc.utils.metrics import calculate_metrics
    manifest = {'ltcontext_revision': LTC_REVISION, 'load_type': args.load_type,
                'frame_sampling_rate': 1, 'splits': {}}
    if args.command == 'audit':
        manifest['data_root'] = str(args.data_root)
    for split in (('train', 'val') if args.command == 'audit' else ('val',)):
        samples, split_path, labels, rows = dataset(args, split)
        label_paths = sorted({labels / 'actions.csv'} | {
            labels / 'coarse_labels' / f"{r['action_type']}_{r['video_id']}.txt" for r in rows})
        records = []
        if args.command == 'evaluate':
            expected = {f"{r['action_type']}/{r['video_id']}/{r['view']}/pred.npy" for r in rows}
            actual = {p.relative_to(args.predictions).as_posix() for p in args.predictions.rglob('pred.npy')}
            if actual != expected:
                raise ValueError(f'Prediction coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
        iterator = (enumerate(audit_records(samples, args.workers)) if args.command == 'audit'
                    else ((i, samples[i]) for i in range(len(samples))))
        for index, item in iterator:
            if args.command == 'audit':
                record = item
            else:
                sample = item
                validate_sample(sample)
                record = {'video_name': sample['video_name'], 'frames': sample['targets'].numel()}
                record['targets_sha256'] = hashlib.sha256(sample['targets'].numpy().tobytes()).hexdigest()
                path = args.predictions / sample['video_name'] / 'pred.npy'
                prediction = np.load(path, allow_pickle=False)
                validate_prediction(prediction, record['frames'])
                record['prediction_sha256'] = sha256(path)
                values = calculate_metrics(sample['targets'][None], torch.from_numpy(prediction.astype(np.int64))[None], [-100])
                record.update({name: float(values[name]) * (1 if name == 'Edit' else 100) for name in METRICS})
            records.append(record)
            if (index + 1) % 100 == 0:
                print(f'{split}: checked {index+1}/{len(samples)}', flush=True)
        manifest['splits'][split] = {'split_sha256': sha256(split_path),
            'annotation_sha256': {p.relative_to(labels).as_posix(): sha256(p) for p in label_paths},
            'count': len(records), 'records': records}
        lengths = np.array([r['frames'] for r in records])
        manifest['splits'][split]['length_percentiles'] = dict(zip(
            ('min', 'p25', 'p50', 'p75', 'p90', 'p99', 'max'),
            np.quantile(lengths, [0, .25, .5, .75, .9, .99, 1]).tolist()))
        if args.command == 'evaluate':
            means = {name: float(np.mean([r[name] for r in records])) for name in METRICS}
            manifest['validation_metrics_percent'] = means
            manifest['checkpoint_selection_score'] = sum(means.values()) / 100
            manifest['aggregation'] = 'unweighted mean of per-sequence metrics; all 202 classes included'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {args.output}')


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    args.data_root.mkdir(parents=True, exist_ok=True)
    provenance = args.data_root / 'download_provenance.json'
    revision = (json.loads(provenance.read_text())['revision'] if provenance.exists()
                else api.dataset_info('cvml-nus/assembly101').sha)
    patterns = ['annotations/coarse-annotations/**', 'TSM_features/*.zip', 'TSM_features/read_lmdb.py']
    provenance.write_text(json.dumps({'repo': 'cvml-nus/assembly101', 'revision': revision,
                                      'patterns': patterns, 'complete': False}, indent=2) + '\n')
    # Normal HF login/token discovery; no credentials are stored in manifests.
    snapshot_download('cvml-nus/assembly101', repo_type='dataset', revision=revision,
                      allow_patterns=patterns, local_dir=args.data_root, max_workers=2)
    (args.data_root / 'download_provenance.json').write_text(json.dumps(
        {'repo': 'cvml-nus/assembly101', 'revision': revision, 'patterns': patterns, 'complete': True}, indent=2) + '\n')



def extract_archive(archive: Path):
    import fcntl
    destination = archive.parent.parent / 'db_TSM_features' / archive.stem
    destination.mkdir(parents=True, exist_ok=True)
    with local_lock(destination / 'extraction.lock') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with zipfile.ZipFile(archive) as source:
            entries = [i for i in source.infolist() if Path(i.filename).name == 'data.mdb']
            if len(entries) != 1:
                raise ValueError(f'Expected one data.mdb in {archive}')
            entry = entries[0]
            target = destination / 'data.mdb'
            stamp = destination / 'extraction.json'
            identity = {'archive': archive.name, 'size': entry.file_size, 'crc32': entry.CRC}
            if target.exists() and target.stat().st_size == entry.file_size and stamp.exists() and json.loads(stamp.read_text()) == identity:
                return archive.name
            partial = target.with_suffix('.mdb.partial')
            with source.open(entry) as src, partial.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            # zipfile verifies the member CRC while streaming to EOF.
            partial.replace(target)
            stamp.write_text(json.dumps(identity) + '\n')
    return archive.name


def extract(args):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    archives = sorted((args.data_root / 'TSM_features').glob('*.zip'))
    if len(archives) != 16:
        raise ValueError(f'Expected all 16 TSM view archives, found {len(archives)}')
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(extract_archive, archive) for archive in archives]
        for future in as_completed(futures):
            print(f'Extracted {future.result()}', flush=True)
    annotations = args.data_root / 'annotations/coarse-annotations'
    if not (annotations / 'actions.csv').is_file():
        raise FileNotFoundError(f'Missing coarse annotations at {annotations}')
    link = args.ltcontext / 'data/assembly101/coarse-annotations'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(annotations, target_is_directory=True)
    elif link.resolve() != annotations.resolve():
        raise ValueError(f'Existing annotations point elsewhere: {link}')


def training_config(args):
    from ltc.config.defaults import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(str(args.ltcontext / 'configs/Assembly101/LTContext.yaml'))
    cfg.DATA.PATH_TO_DATA_DIR = str(args.data_root)
    cfg.DATA.LOAD_TYPE = args.load_type
    cfg.DATA_LOADER.NUM_WORKERS = args.workers
    cfg.TEST.ENABLE = False
    cfg.RNG_SEED = args.seed
    cfg.freeze()
    return cfg


def check_audit(args):
    audit = json.loads(args.audit.read_text())
    if audit.get('data_root') != str(args.data_root):
        raise ValueError('Audit belongs to another data root')
    if audit['ltcontext_revision'] != LTC_REVISION or audit['load_type'] != args.load_type:
        raise ValueError('Audit does not match the requested source/data format')
    if audit['frame_sampling_rate'] != 1:
        raise ValueError('Full-resolution features are required')
    for split, count in [('train', 4684), ('val', 1424)]:
        info = audit['splits'][split]
        if info['count'] != count or len(info['records']) != count:
            raise ValueError(f'Incomplete {split} audit')
        if info['split_sha256'] != sha256(args.ltcontext / 'data/assembly101' / f'{split}.csv'):
            raise ValueError(f'{split} split changed after audit')
        for name, digest in info['annotation_sha256'].items():
            if sha256(args.ltcontext / 'data/assembly101/coarse-annotations' / name) != digest:
                raise ValueError(f'Annotation changed after audit: {name}')
    return audit


def metrics_percent(target, prediction):
    from ltc.utils.metrics import calculate_metrics
    values = calculate_metrics(target, prediction, [-100])
    return {name: float(values[name]) * (1 if name == 'Edit' else 100) for name in METRICS}


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, best_score, generator, identity):
    """Persist the evaluated weights, their metrics, and the next epoch's RNG state."""
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'metrics': metrics,
                'best_score': best_score, 'identity': identity,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'loader_rng': generator.get_state()}, temporary)
    temporary.replace(path)


def check_resume_identity(previous, current, upgrade_execution=False):
    """Allow an explicit execution-only migration, never data/model changes."""
    if previous == current:
        return
    if upgrade_execution:
        old, new = dict(previous), dict(current)
        for value in (old, new):
            value.pop('execution_backend', None)
            value.pop('graph_buckets', None)
            value.pop('deterministic_algorithms', None)
            value.pop('cublas_workspace_config', None)
            value['sources_sha256'] = dict(value['sources_sha256'])
            value['sources_sha256'].pop('experiments/assembly101.py', None)
        if old == new:
            return
    raise ValueError('Resume source, config, rank, seed or data identity mismatch')


def train(args):
    """Use upstream model, data, loss, optimizer and metrics; own run bookkeeping."""
    import fcntl
    from ltc.model.ltcontext import LTC
    from ltc.model.loss import get_loss_func
    from ltc.model.optimizer import construct_optimizer, construct_lr_scheduler
    from ltc.dataset.utils import sequence_collate
    from torch.utils.data import DataLoader, RandomSampler, Subset
    from ridgon.ball import cuda as cuda_backend

    args.output.mkdir(parents=True, exist_ok=True)
    lock = local_lock(args.output / 'run.lock')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    audit = check_audit(args)
    cfg = training_config(args)
    execution_backend = getattr(args, 'execution_backend', 'eager')
    torch.set_num_threads(args.threads)
    # Avoid cuDNN convolution nondeterminism when comparing ranks or resuming.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if execution_backend != 'eager':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    cuda_backend.load(device=torch.device('cuda', torch.cuda.current_device()))
    model = replace_stage1(LTC(cfg.MODEL), args.rank).cuda()
    attention_backend = getattr(args, 'attention_backend', 'sdpa')
    if attention_backend == 'sdpa':
        enable_sdpa(model)
    optimizer = construct_optimizer(model, cfg)
    scheduler = construct_lr_scheduler(cfg, optimizer)
    loss_func = get_loss_func(cfg)
    if execution_backend == 'graph-compile':
        # Call the upstream loss on sliced real frames, fusing its pointwise
        # operations without introducing a second mathematical implementation.
        loss_func = torch.compile(loss_func, dynamic=True, fullgraph=True,
            options={'triton.cudagraphs': False, 'fallback_random': True})
    graph_buckets = tuple(getattr(args, 'graph_buckets', (2048, 4096, 6144, 8192, 12288, 16384, 24576, 49152)))
    if execution_backend != 'eager' and attention_backend != 'sdpa':
        raise ValueError('Graph execution requires SDPA attention')
    # Rank changes parameter initialization draws, but must not change data order.
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_data, *_ = dataset(args, 'train')
    val_data, *_ = dataset(args, 'val')
    for split, data in [('train', train_data), ('val', val_data)]:
        names = [f"{r['action_type']}/{r['video_id']}/{r['view']}" for r in data._data]
        if names != [r['video_name'] for r in audit['splits'][split]['records']]:
            raise ValueError(f'{split} data order differs from audit')

    sources = [Path(__file__).resolve(), *sorted((root / 'ridgon').rglob('*.py'))]
    identity = {'rank': args.rank, 'seed': args.seed, 'audit_sha256': sha256(args.audit),
                'ltcontext_revision': LTC_REVISION, 'config': cfg.dump(),
                'sources_sha256': {str(p.relative_to(root)): sha256(p) for p in sources},
                'cuda_contract': cuda_backend._CUDA_CONTRACT_VERSION,
                'precision': 'FP32 upstream and parameters; BF16 native Ridgon activations',
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'attention_backend': attention_backend,
                'execution_backend': execution_backend,
                'graph_buckets': list(graph_buckets) if execution_backend != 'eager' else [],
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'cudnn_deterministic': True, 'cudnn_benchmark': False,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}
    metadata = {**identity, 'parameters': sum(p.numel() for p in model.parameters()),
                'gpu': torch.cuda.get_device_name(), 'command': args.command,
                'started': time.time(), 'ltcontext_published_baseline': 'not retrained',
                'checkpoint_selection': 'sum of five per-sequence mean percentage metrics / 100'}
    if (args.output / 'metadata.json').exists() and not args.resume:
        raise FileExistsError('Output already contains a run; use --resume or a new output directory')
    if not args.resume:
        write_json(args.output / 'metadata.json', metadata)
        (args.output / 'config.yaml').write_text(cfg.dump())
    loaders = dict(batch_size=1, num_workers=args.workers, pin_memory=True,
                   collate_fn=partial(sequence_collate, pad_ignore_idx=-100),
                   persistent_workers=args.workers > 0)
    if args.workers > 0:
        loaders['prefetch_factor'] = 1
    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data, generator=generator),
                             generator=torch.Generator().manual_seed(args.seed + 1), **loaders)
    val_loader = DataLoader(val_data, shuffle=False,
                           generator=torch.Generator().manual_seed(args.seed + 2), **loaders)

    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        logits = execute(batch['features'].cuda(non_blocking=True), batch['masks'].cuda(non_blocking=True))
        loss = loss_func(logits, batch['targets'].cuda(non_blocking=True))['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss: {batch['video_name']}")
        loss.backward()
        # Stop before updating corrupt weights; do not clip or skip a sample.
        # Infinity norm detects NaN/Inf without squaring overflow or modifying gradients.
        torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None], float('inf'),
            error_if_nonfinite=True, foreach=True)
        optimizer.step()
        return float(loss.detach())

    start_epoch, best_score = 0, -1.0
    if args.resume:
        checkpoint = args.output / 'last.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        check_resume_identity(state['identity'], identity, getattr(args, 'upgrade_execution', False))
        if state['identity'] != identity:
            transition = {'checkpoint_sha256': sha256(checkpoint), 'completed_epoch': state['epoch']+1,
                          'previous_identity': state['identity'], 'current_identity': identity,
                          'updated': time.time()}
            with (args.output / 'execution-transitions.jsonl').open('a') as stream:
                stream.write(json.dumps(transition) + '\n')
            original = json.loads((args.output / 'metadata.json').read_text())
            write_json(args.output / 'metadata.json', {**original, **identity})
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        generator.set_state(state['loader_rng'])
        start_epoch, best_score = state['epoch'] + 1, state['best_score']
        del state
    execute = model if execution_backend == 'eager' else BucketExecution(model, execution_backend, graph_buckets)

    if args.command == 'pilot':
        # Exercise the actual extremes before spending a full epoch on a run.
        lengths = np.array([r['frames'] for r in audit['splits']['train']['records']])
        order = np.argsort(lengths)
        indices = sorted(set(int(order[round(q * (len(order)-1))]) for q in (0, .5, .9, 1)))
        model.train()
        probes = []
        for index in indices:
            batch = sequence_collate([train_data[index]])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = update(batch)
            torch.cuda.synchronize()
            probe = {'video_name': train_data._data[index]['video_id'], 'frames': int(lengths[index]),
                     'loss': loss, 'seconds': time.perf_counter()-started,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            probes.append(probe)
            print(json.dumps({'length_probe': probe}), flush=True)
        # LMDB handles opened by the parent probes must not be inherited by workers.
        if args.load_type == 'lmdb' and train_data.env is not None:
            for env in train_data.env.values():
                env.close()
            train_data.env = None
        val_loader = DataLoader(Subset(val_data, list(range(min(20, len(val_data))))),
                                shuffle=False, generator=torch.Generator().manual_seed(args.seed + 2), **loaders)
    else:
        probes = []

    total_epochs = 1 if args.command == 'pilot' else cfg.SOLVER.MAX_EPOCH
    for epoch in range(start_epoch, total_epochs):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            loss = update(batch)
            losses.append(loss)
            if step == 1 or step % 25 == 0:
                status = {'state': 'training', 'epoch': epoch+1, 'step': step,
                          'steps_per_epoch': len(train_loader), 'loss': loss, 'rank': args.rank,
                          'elapsed_seconds': time.perf_counter()-started, 'updated': time.time()}
                write_json(args.output / 'status.json', status)
                print(json.dumps(status), flush=True)
            if args.command == 'pilot' and step >= args.pilot_steps:
                break
        train_seconds = time.perf_counter()-started
        peak = torch.cuda.max_memory_allocated()
        model.eval()
        values, predictions = [], []
        started = time.perf_counter()
        with torch.no_grad():
            for step, batch in enumerate(val_loader, 1):
                logits = model(batch['features'].cuda(non_blocking=True), batch['masks'].cuda(non_blocking=True))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite validation logits')
                prediction = logits[-1].argmax(dim=1).cpu()
                values.append(metrics_percent(batch['targets'], prediction))
                predictions.append((batch['video_name'][0][0], prediction[0].numpy()))
                if step == 1 or step % 100 == 0:
                    status = {'state': 'validation', 'epoch': epoch+1, 'step': step,
                              'validation_entries': len(val_loader), 'rank': args.rank, 'updated': time.time()}
                    write_json(args.output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        means = {name: float(np.mean([v[name] for v in values])) for name in METRICS}
        score = sum(means.values()) / 100
        improved = score > best_score
        best_score = max(best_score, score)
        row = {'epoch': epoch+1, 'train_loss': float(np.mean(losses)), 'validation_metrics_percent': means,
               'selection_score': score, 'best_score': best_score, 'train_seconds': train_seconds,
               'validation_seconds': time.perf_counter()-started, 'train_steps': len(losses),
               'peak_allocated_bytes': peak, 'lr': optimizer.param_groups[0]['lr']}
        scheduler.step()
        if args.command == 'train':
            # Evaluate first: the checkpoint and selected metrics refer to these exact weights.
            save_checkpoint(args.output / 'epoch.pt', model, optimizer, scheduler, epoch, means,
                            best_score, generator, identity)
            if improved:
                shutil.copyfile(args.output / 'epoch.pt', args.output / 'best.pt.tmp')
                (args.output / 'best.pt.tmp').replace(args.output / 'best.pt')
                for name, prediction in predictions:
                    path = args.output / 'predictions' / name / 'pred.npy'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, prediction, allow_pickle=False)
                write_json(args.output / 'best_metrics.json', row)
            # Commit resume state only after best-checkpoint exports are complete.
            (args.output / 'epoch.pt').replace(args.output / 'last.pt')
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
    if args.command == 'pilot':
        write_json(args.output / 'pilot.json', {'passed': True, 'identity': identity,
                   'length_probes': probes, 'timing': row, 'not_a_formal_result': True})
    write_json(args.output / 'status.json', {'state': 'complete', 'rank': args.rank, 'updated': time.time()})
    lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download', 'extract', 'audit', 'evaluate', 'pilot', 'train'))
    parser.add_argument('--ltcontext', type=Path, default=Path('/root/LTContext'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--load-type', choices=('lmdb', 'numpy'), default='lmdb')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--rank', type=int, choices=(16, 32), default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--pilot-steps', type=int, default=200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--attention-backend', choices=('upstream', 'sdpa'), default='sdpa')
    parser.add_argument('--execution-backend', choices=('eager', 'graph', 'graph-compile'), default='eager')
    parser.add_argument('--graph-buckets', type=int, nargs='+',
                        default=[2048, 4096, 6144, 8192, 12288, 16384, 24576, 49152])
    parser.add_argument('--upgrade-execution', action='store_true',
                        help='Explicitly resume with a new runner/execution backend; log the transition')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for name in ('ltcontext', 'data_root', 'predictions', 'output', 'audit'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.command == 'download':
        download(args)
    elif args.command == 'extract':
        extract(args)
    elif args.command in ('pilot', 'train'):
        if args.output is None or args.audit is None:
            parser.error('pilot/train require --output and --audit')
        if args.workers < 0 or args.threads < 1 or args.pilot_steps < 1:
            parser.error('Invalid worker/thread/pilot-step count')
        with upstream(args.ltcontext):
            train(args)
    else:
        if args.output is None or (args.command == 'evaluate' and args.predictions is None):
            parser.error('audit/evaluate require --output; evaluate also requires --predictions')
        with upstream(args.ltcontext):
            run(args)


if __name__ == '__main__':
    main()
