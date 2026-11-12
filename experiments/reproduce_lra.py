"""Replay a published LRA configuration with local dataset/output paths."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from experiments import train_transformers as train
from memsolve.ball.reference import axial_rotary_position_factors


class PixelEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lookup = nn.Embedding(256, dim)
        nn.init.normal_(self.lookup.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = (x.squeeze(-1).float() * 255).round().long()
        return self.lookup(indices)


class PixelEncoder(train.SequenceEncoder):
    def __init__(self, **kwargs):
        kwargs = {**kwargs, 'position_encoding': 'absolute'}
        super().__init__(**kwargs)
        if kwargs['input_kind'] == 'values':
            with torch.random.fork_rng(devices=[]):
                self.input_embedding = PixelEmbedding(kwargs['dim'])
        self.position_encoding = 'rove_2d'
        self.position_embedding = self.row_embedding = self.column_embedding = None
        positions = torch.arange(kwargs['max_length'])[None]
        columns = self.grid_shape[1]
        coords = torch.stack((positions % columns, positions // columns), -1)
        cosine, sine = axial_rotary_position_factors(
            coords, head_dim=kwargs['dim'] // kwargs['num_heads']
        )
        self.register_buffer('rotary_cos', cosine, persistent=False)
        self.register_buffer('rotary_sin', sine, persistent=False)
        for block in self.blocks:
            if hasattr(block.mixer, 'config'):
                block.mixer.config = replace(
                    block.mixer.config, rotary=True, rotary_scaling='none'
                )


def _verify_recorded_bundle(bundle, record_data: dict) -> None:
    train.validate_formal_source_provenance(bundle)
    expected_dataset = record_data['dataset']
    actual_sha = bundle.metadata.get('source', {}).get('content_sha256')
    expected_sha = expected_dataset['source']['content_sha256']
    if actual_sha != expected_sha:
        raise ValueError(
            f'dataset content_sha256 mismatch: got {actual_sha!r}, '
            f'expected {expected_sha!r}'
        )
    for key in (
        'suite',
        'task',
        'source_definition',
        'split_protocol',
        'split_seed',
        'validation_fraction',
        'vocabulary_fingerprint',
        'split_fingerprints',
        'tokenization',
        'resolution',
        'value_masking',
    ):
        if bundle.metadata.get(key) != expected_dataset.get(key):
            raise ValueError(
                f'dataset {key} mismatch: got {bundle.metadata.get(key)!r}, '
                f'expected {expected_dataset.get(key)!r}'
            )
    actual_sizes = {
        'train': len(bundle.train),
        'validation': len(bundle.validation),
        'test': len(bundle.test),
    }
    if actual_sizes != record_data['split_sizes']:
        raise ValueError(
            f"split_sizes mismatch: got {actual_sizes!r}, "
            f"expected {record_data['split_sizes']!r}"
        )
    actual_contract = {
        'input_kind': bundle.input_kind,
        'max_length': bundle.max_length,
        'vocab_size': bundle.vocab_size,
        'paired': bundle.paired,
        'value_masking': (
            bundle.value_masking if bundle.input_kind == 'values' else None
        ),
    }
    if actual_contract != record_data['data_contract']:
        raise ValueError(
            f"data_contract mismatch: got {actual_contract!r}, "
            f"expected {record_data['data_contract']!r}"
        )


@contextmanager
def _apply_reproduction_protocol(record_data: dict):
    original_encoder = train.SequenceEncoder
    original_bundle = train.build_bundle
    original_payload = train._build_run_payload
    is_pathfinder = record_data['resolved_arguments']['task'] == 'pathfinder'

    def build_bundle(args):
        formal_args = argparse.Namespace(**{**vars(args), 'formal': True})
        bundle = original_bundle(formal_args)
        if is_pathfinder:
            assert args.task == 'pathfinder' and bundle.input_kind == 'values'
            bundle = replace(
                bundle,
                value_masking='nonzero',
                metadata={**bundle.metadata, 'value_masking': 'nonzero-pixels'},
            )
        _verify_recorded_bundle(bundle, record_data)
        return bundle

    def payload(*args, **kwargs):
        result = original_payload(*args, **kwargs)
        result['model']['position_encoding'] = 'rove_2d'
        result['model']['position_initialization'] = (
            'fixed-base10000-unit-amplitude-no-scaling'
        )
        result['model']['input_embedding'] = 'learned-256-level-pixel-lookup'
        result['model']['input_embedding_initialization'] = 'normal-std0.02'
        result['probe_source'] = {
            'path': 'experiments/reproduce_lra.py',
            'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'note': (
                'LRA mixer control with identical pixel lookup, nonzero mask, '
                'and 2D RoVE content rotations. Retain the absolute-position '
                'constructor RNG consumption before replacing positions; 2D '
                'RoVE replaces rather than augments LPE. Original raster '
                'coordinates retained. Mean over valid pixels.'
            ),
        }
        return result

    train.build_bundle = build_bundle
    if is_pathfinder:
        train.SequenceEncoder = PixelEncoder
        train._build_run_payload = payload
    try:
        yield
    finally:
        train.SequenceEncoder = original_encoder
        train.build_bundle = original_bundle
        train._build_run_payload = original_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', help='Published model/task, e.g. memsolve/image')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    opts = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = root / 'results/lra' / opts.record / 'config.json'
    record_data = json.loads(record.read_text())
    recorded = record_data['resolved_arguments']
    with _apply_reproduction_protocol(record_data):
        args = train.resolve_args(train.parse_args([
            '--config', str(root / 'experiments/configs/lra_rove.toml'),
            '--task', recorded['task'], '--output', str(opts.output)]))
        for key, value in recorded.items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.data_root = opts.data_root.resolve()
        args.cache_root = opts.cache_root.resolve()
        args.output = opts.output.resolve()
        args.config = root / 'experiments/configs/lra_rove.toml'
        args.formal = False  # Source tree may be modified; data provenance is verified against record_data.
        args.prepare_only = opts.prepare_only
        args.resume = False
        train._validate_resolved_args(args)
        try:
            train._main(args)
        except train.NonFiniteTrainingError as error:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / 'failure.json').write_text(json.dumps(error.record, indent=2))
            raise SystemExit(86) from error


if __name__ == '__main__':
    main()
