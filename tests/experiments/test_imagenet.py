from __future__ import annotations

import copy
import io
import json
import random
import sys
import runpy
import tarfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset

import experiments.imagenet as imagenet
from experiments.imagenet import (
    BatchingPlan,
    DEFAULT_CONFIG,
    DistributedState,
    ImageNetRun,
    LoaderRandomGenerators,
    _atomic_torch_save,
    _capture_resume_rng_state,
    _checkpoint,
    _checkpoint_model_state,
    _load_resume,
    _recipe_fidelity,
    _restore_resume_rng_state,
    build_loaders,
    build_optimizer,
    build_scheduler,
    checkpoint_contract_digest,
    load_run,
    parse_args,
    resolve_batching_plan,
    train_epoch,
)


pytestmark = pytest.mark.experiment


def test_direct_imagenet_launcher_imports_repository_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    original_path = sys.path.copy()
    try:
        namespace = runpy.run_path(
            str(root / "experiments" / "train_imagenet.py"),
            run_name="ridgon_direct_launcher_test",
        )
    finally:
        sys.path[:] = original_path
    assert callable(namespace["main"])


def _args(tmp_path: Path, *extra: str):
    return parse_args(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--tier",
            "small",
            "--data-root",
            str(tmp_path / "imagenet"),
            "--output",
            str(tmp_path / "run"),
            *extra,
        ]
    )


def _cpu_state(*, world_size: int = 1, rank: int = 0) -> DistributedState:
    return DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        device=torch.device("cpu"),
    )


def _checkpoint_batching_plan() -> BatchingPlan:
    return BatchingPlan(
        world_size=1,
        physical_batch_size=256,
        effective_batch_size=1024,
        grad_accum=4,
        samples_per_epoch=1_281_024,
        updates_per_epoch=1251,
    )


def _data_contract(*, source_views: int = 1) -> dict[str, object]:
    return {
        "format": "webdataset-v1",
        "source": imagenet.IMAGENET_WDS_SOURCE,
        "manifest_sha256": imagenet.IMAGENET_WDS_MANIFEST_SHA256,
        "train": {
            "samples": imagenet.IMAGENET_TRAIN_SAMPLES,
            "shards": imagenet.IMAGENET_TRAIN_SHARDS,
        },
        "validation": {
            "samples": imagenet.IMAGENET_VALIDATION_SAMPLES,
            "shards": imagenet.IMAGENET_VALIDATION_SHARDS,
        },
        "streaming": {
            "shard_order": "global-epoch-permutation-then-rank-stride-worker-quota",
            "sample_shuffle": {
                "buffer_size": imagenet.WDS_SAMPLE_SHUFFLE_SIZE,
                "initial_size": imagenet.WDS_SAMPLE_SHUFFLE_INITIAL,
            },
            "source_views": source_views,
            "repeated_augmentation_placement": "rank-local-physical-batch-interleave",
            "validation_partition": "worker-stride-full-per-rank",
            "worker_rng": "epoch-rank-worker-v1",
        },
    }


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (8, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _pixel_code_transform(image: object) -> torch.Tensor:
    pixel = image.getpixel((0, 0))  # type: ignore[union-attr]
    return torch.tensor(pixel, dtype=torch.int64)


def _random_pixel_transform(image: object) -> torch.Tensor:
    import numpy as np

    pixel = image.getpixel((0, 0))
    return torch.tensor((*pixel, random.random(), float(np.random.random()),
                         float(torch.rand(()))), dtype=torch.float64)


def _write_webdataset_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_shards: int = 2,
    validation_shards: int = 2,
    samples_per_shard: int = 4,
) -> Path:
    root = tmp_path / "imagenet-wds"
    root.mkdir()
    monkeypatch.setattr(imagenet, "IMAGENET_TRAIN_SHARDS", train_shards)
    monkeypatch.setattr(imagenet, "IMAGENET_VALIDATION_SHARDS", validation_shards)
    monkeypatch.setattr(
        imagenet,
        "IMAGENET_TRAIN_SAMPLES",
        train_shards * samples_per_shard,
    )
    monkeypatch.setattr(
        imagenet,
        "IMAGENET_VALIDATION_SAMPLES",
        validation_shards * samples_per_shard,
    )

    splits: dict[str, dict[str, object]] = {}
    for split, shard_count, width in (
        ("train", train_shards, 4),
        ("validation", validation_shards, 2),
    ):
        filenames = [
            f"imagenet1k-{split}-{index:0{width}d}.tar"
            for index in range(shard_count)
        ]
        for shard_index, filename in enumerate(filenames):
            with tarfile.open(root / filename, "w") as archive:
                for sample_index in range(samples_per_shard):
                    key = f"{split}-{shard_index:02d}-{sample_index:02d}"
                    image_bytes = _jpeg_bytes(
                        (shard_index * 40, sample_index * 40, 17)
                    )
                    label = (shard_index + sample_index) % 3
                    for suffix, value in (
                        ("jpg", image_bytes),
                        ("cls", str(label).encode("ascii")),
                        ("json", json.dumps({"label": label}).encode("utf-8")),
                    ):
                        member = tarfile.TarInfo(f"{key}.{suffix}")
                        member.size = len(value)
                        archive.addfile(member, io.BytesIO(value))
        splits[split] = {
            "name": split,
            "filenames": filenames,
            "shard_lengths": [samples_per_shard] * shard_count,
            "num_samples": shard_count * samples_per_shard,
        }
    manifest = {"name": "imagenet1k", "splits": splits}
    raw_manifest = json.dumps(manifest, indent=2).encode("utf-8")
    (root / "_info.json").write_bytes(raw_manifest)
    monkeypatch.setattr(
        imagenet,
        "IMAGENET_WDS_MANIFEST_SHA256",
        imagenet.hashlib.sha256(raw_manifest).hexdigest(),
    )
    return root


def _loader_generators(seed: int = 0) -> LoaderRandomGenerators:
    return LoaderRandomGenerators(
        train=torch.Generator().manual_seed(seed),
        validation=torch.Generator().manual_seed(seed + 1),
    )


class _WorkerRandomDataset(Dataset[torch.Tensor]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, _index: int) -> torch.Tensor:
        import numpy as np

        return torch.tensor(
            (random.random(), float(np.random.random()), float(torch.rand(()))),
            dtype=torch.float64,
        )


def _worker_random_loader(generator: torch.Generator) -> DataLoader[torch.Tensor]:
    return DataLoader(
        _WorkerRandomDataset(),
        batch_size=2,
        num_workers=1,
        persistent_workers=False,
        multiprocessing_context="spawn",
        worker_init_fn=imagenet._seed_worker,
        generator=generator,
    )


def test_small_recipe_uses_plain_vit3_training(tmp_path: Path) -> None:
    run = load_run(_args(tmp_path))
    assert run.model == {
        "image_size": 224,
        "patch_size": 16,
        "num_classes": 1000,
        "mlp_ratio": 4.0,
        "architecture": "vit3_cpe_mean_swiglu_v2", "ffn": "swiglu",
        "layer_scale": False,
        "class_token": False,
        "pooling": "token_ln_mean",
        "norm_eps": 1.0e-6,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "rank": 32,
        "drop_path_rate": 0.1,
        "drop_path_schedule": "linear",
        "position_encoding": "cpe",
    }
    assert (run.train["epochs"], run.train["optimizer"], run.train["augmentation"]) == (
        300,
        "fused_adamw",
        "rand_augment",
    )
    assert not any(run.train[k] for k in ("repeated_aug", "bce_loss", "ema"))
    assert run.train["amp_dtype"] == "bfloat16"
    assert (
        run.train["batch_size"],
        run.train["effective_batch"],
    ) == (512, 1024)
    assert (run.train["train_workers"], run.train["val_workers"]) == (12, 4)
    assert run.operator == {

        "bias": True,
        "implementation": "cuda",
    }


def test_worker_counts_are_independently_overridable(tmp_path: Path) -> None:
    run = load_run(_args(tmp_path, "--train-workers", "7", "--val-workers", "0"))
    assert (run.train["train_workers"], run.train["val_workers"]) == (7, 0)
    assert {"train_workers", "val_workers"}.issubset(run.overrides)


def test_webdataset_manifest_requires_the_pinned_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_webdataset_root(tmp_path, monkeypatch)
    manifest = imagenet.load_imagenet_webdataset_manifest(root)
    assert manifest.train.contract() == {"samples": 8, "shards": 2}
    assert manifest.validation.contract() == {"samples": 8, "shards": 2}
    assert manifest.train.paths[0].name == "imagenet1k-train-0000.tar"
    assert manifest.validation.paths[-1].name == "imagenet1k-validation-01.tar"
    imagenet.verify_imagenet_webdataset_payloads(manifest)

    invalid = root / "imagenet1k-train-0001.tar"
    invalid.write_bytes(b"not a tar archive")
    with pytest.raises(ValueError, match="not a readable tar archive"):
        imagenet.verify_imagenet_webdataset_payloads(manifest)

    invalid.unlink()
    with pytest.raises(FileNotFoundError, match="missing non-empty shards"):
        imagenet.load_imagenet_webdataset_manifest(root)


def test_webdataset_preflight_rejects_duplicate_and_noncontiguous_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_webdataset_root(tmp_path, monkeypatch)
    shard = root / "imagenet1k-train-0001.tar"

    with tarfile.open(shard, "a") as archive:
        for suffix, value in (("jpg", b"image"), ("cls", b"0"), ("json", b"{}")):
            member = tarfile.TarInfo(f"train-01-00.{suffix}")
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    with pytest.raises(ValueError, match="repeats source key"):
        imagenet._verify_webdataset_shard(shard, expected_samples=4)

    with tarfile.open(shard, "w") as archive:
        members = (
            ("alpha.jpg", b"image"),
            ("beta.jpg", b"image"),
            ("alpha.cls", b"0"),
            ("alpha.json", b"{}"),
            ("beta.cls", b"0"),
            ("beta.json", b"{}"),
            ("gamma.jpg", b"image"),
            ("gamma.cls", b"0"),
            ("gamma.json", b"{}"),
            ("delta.jpg", b"image"),
            ("delta.cls", b"0"),
            ("delta.json", b"{}"),
        )
        for name, value in members:
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    with pytest.raises(ValueError, match="does not contain exactly one"):
        imagenet._verify_webdataset_shard(shard, expected_samples=4)


def test_webdataset_train_repeats_undecoded_groups_and_replays_an_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("webdataset")
    root = _write_webdataset_root(tmp_path, monkeypatch)
    manifest = imagenet.load_imagenet_webdataset_manifest(root)
    calls: list[int] = []

    def counting_transform(_image: object) -> torch.Tensor:
        calls.append(len(calls))
        return torch.tensor([calls[-1]], dtype=torch.int64)

    dataset = imagenet._ImageNetWebDataset(
        manifest.train,
        transform=counting_transform,
        state=_cpu_state(),
        seed=17,
        num_classes=3,
        training=True,
        source_views=3,
        physical_batch_size=4,
        microbatches_per_epoch=2,
        worker_count=1,
    )
    dataset.set_epoch(4)
    batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0, drop_last=True)))
    assert batch[0][:, 0].tolist() == [0, 1, 2, 3]
    assert calls == [0, 1, 2, 3]

    def stable_transform(image: object) -> torch.Tensor:
        return torch.tensor(image.getpixel((0, 0)), dtype=torch.int64)  # type: ignore[union-attr]

    replayable = imagenet._ImageNetWebDataset(
        manifest.train,
        transform=stable_transform,
        state=_cpu_state(),
        seed=17,
        num_classes=3,
        training=True,
        source_views=3,
        physical_batch_size=4,
        microbatches_per_epoch=2,
        worker_count=1,
    )
    replayable.set_epoch(7)
    first = next(iter(DataLoader(replayable, batch_size=4, num_workers=0, drop_last=True)))
    replayable.set_epoch(7)
    second = next(iter(DataLoader(replayable, batch_size=4, num_workers=0, drop_last=True)))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_webdataset_rank_partition_and_validation_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("webdataset")
    root = _write_webdataset_root(tmp_path, monkeypatch, train_shards=4)
    manifest = imagenet.load_imagenet_webdataset_manifest(root)
    rank_zero = imagenet._ImageNetWebDataset(
        manifest.train,
        transform=lambda _image: torch.zeros(1),
        state=_cpu_state(world_size=2, rank=0),
        seed=3,
        num_classes=3,
        training=True,
        source_views=1,
        physical_batch_size=2,
        microbatches_per_epoch=2,
        worker_count=2,
    )
    rank_one = imagenet._ImageNetWebDataset(
        manifest.train,
        transform=lambda _image: torch.zeros(1),
        state=_cpu_state(world_size=2, rank=1),
        seed=3,
        num_classes=3,
        training=True,
        source_views=1,
        physical_batch_size=2,
        microbatches_per_epoch=2,
        worker_count=2,
    )
    zero_shards = rank_zero._rank_shards(epoch=2)
    one_shards = rank_one._rank_shards(epoch=2)
    assert {path for path, _ in zero_shards}.isdisjoint(
        {path for path, _ in one_shards}
    )
    assert {path for path, _ in zero_shards} | {
        path for path, _ in one_shards
    } == set(manifest.train.paths)
    worker_zero = rank_zero._training_slices(
        worker_id=0,
        worker_count=2,
        epoch=2,
    )
    worker_one = rank_zero._training_slices(
        worker_id=1,
        worker_count=2,
        epoch=2,
    )
    worker_zero_records = {
        (slice_.path, index)
        for slice_ in worker_zero
        for index in range(slice_.start, slice_.start + slice_.count)
    }
    worker_one_records = {
        (slice_.path, index)
        for slice_ in worker_one
        for index in range(slice_.start, slice_.start + slice_.count)
    }
    assert worker_zero_records.isdisjoint(worker_one_records)
    assert len(worker_zero_records | worker_one_records) == 4

    transform = lambda _image: torch.ones(1)
    validation_zero = imagenet._ImageNetWebDataset(
        manifest.validation,
        transform=transform,
        state=_cpu_state(world_size=2, rank=0),
        seed=3,
        num_classes=3,
        training=False,
    )
    validation_one = imagenet._ImageNetWebDataset(
        manifest.validation,
        transform=transform,
        state=_cpu_state(world_size=2, rank=1),
        seed=3,
        num_classes=3,
        training=False,
    )
    labels_zero = [label for _, label in validation_zero]
    labels_one = [label for _, label in validation_one]
    assert labels_zero == labels_one
    assert len(labels_zero) == manifest.validation.num_samples


def test_webdataset_multiple_workers_preserve_repeat_groups_and_replay_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("webdataset")
    root = _write_webdataset_root(tmp_path, monkeypatch, train_shards=4)
    manifest = imagenet.load_imagenet_webdataset_manifest(root)
    dataset = imagenet._ImageNetWebDataset(
        manifest.train,
        transform=_pixel_code_transform,
        state=_cpu_state(),
        seed=29,
        num_classes=3,
        training=True,
        source_views=3,
        physical_batch_size=4,
        microbatches_per_epoch=12,
        worker_count=2,
    )

    def collect() -> list[torch.Tensor]:
        loader = DataLoader(
            dataset,
            batch_size=4,
            drop_last=True,
            num_workers=2,
            persistent_workers=False,
            prefetch_factor=1,
        )
        return [images.clone() for _, (images, _targets) in zip(range(12), loader)]

    dataset.set_epoch(5)
    first = collect()
    dataset.set_epoch(5)
    second = collect()
    assert len(first) == 12
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))

    source_groups = [
        tuple(tuple(pixel.tolist()) for pixel in images[offset : offset + 2])
        for images in first
        for offset in range(0, images.shape[0], 2)
    ]
    assert all(len(group) == 2 for group in source_groups)
    assert all(source_groups.count(group) == 3 for group in source_groups)
    assert all(
        len({group for group in source_groups[index : index + 2]}) == 2
        for index in range(0, len(source_groups), 2)
    )


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4, 8))
def test_repeated_augmentation_blocks_keep_physical_batches_source_unique(batch_size) -> None:
    dataset = imagenet._ImageNetWebDataset(
        imagenet._WebDatasetSplit("train", (), (), 1),
        transform=lambda _image: torch.zeros(1), state=_cpu_state(), seed=0,
        num_classes=1, training=True, source_views=3,
        physical_batch_size=batch_size, microbatches_per_epoch=17, worker_count=1,
    )
    remaining = 17
    next_source = 0
    emitted = []
    while remaining:
        source_count, output_count = dataset._repeated_augmentation_block(remaining)
        sources = [list(range(i * batch_size, (i + 1) * batch_size))
                   for i in range(next_source, next_source + source_count)]
        next_source += source_count
        emitted.extend((sources * dataset.source_views)[:output_count])
        remaining -= output_count
    assert next_source == dataset._source_batch_count(17)
    assert len(emitted) == 17
    assert all(len(set(batch)) == batch_size for batch in emitted)
    counts = Counter(sample for batch in emitted for sample in batch)
    assert max(counts.values()) <= dataset.source_views


def test_webdataset_unique_source_quotas_never_cycle_a_short_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("webdataset")
    root = _write_webdataset_root(
        tmp_path,
        monkeypatch,
        train_shards=3,
        samples_per_shard=4,
    )
    manifest = imagenet.load_imagenet_webdataset_manifest(root)
    dataset = imagenet._ImageNetWebDataset(
        manifest.train,
        transform=_pixel_code_transform,
        state=_cpu_state(),
        seed=41,
        num_classes=3,
        training=True,
        source_views=1,
        physical_batch_size=4,
        microbatches_per_epoch=3,
        worker_count=2,
    )
    dataset.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=4,
        drop_last=True,
        num_workers=2,
        persistent_workers=False,
        prefetch_factor=1,
    )
    images = torch.cat([batch for batch, _ in loader])
    assert images.shape[0] == 12
    assert torch.unique(images, dim=0).shape[0] == 12

    short = imagenet._ImageNetWebDataset(
        manifest.train,
        transform=_pixel_code_transform,
        state=_cpu_state(),
        seed=41,
        num_classes=3,
        training=True,
        source_views=1,
        physical_batch_size=4,
        microbatches_per_epoch=4,
        worker_count=1,
    )
    with pytest.raises(RuntimeError, match="cannot cover the planned epoch"):
        next(iter(DataLoader(short, batch_size=4, num_workers=0, drop_last=True)))


def test_webdataset_persistent_workers_observe_epochs_and_replay_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("webdataset")
    root = _write_webdataset_root(tmp_path, monkeypatch, train_shards=4)
    split = imagenet.load_imagenet_webdataset_manifest(root).train

    def make_loader():
        dataset = imagenet._ImageNetWebDataset(
            split, transform=_random_pixel_transform, state=_cpu_state(),
            seed=29, num_classes=3, training=True,
            physical_batch_size=4, microbatches_per_epoch=4, worker_count=2,
        )
        loader = DataLoader(dataset, batch_size=4, drop_last=True,
                            **imagenet._loader_kwargs(
                                workers=2, generator=torch.Generator().manual_seed(97)))
        return dataset, loader

    dataset, loader = make_loader()
    resumed = None
    try:
        assert loader.persistent_workers
        dataset.set_epoch(0)
        first = list(loader)
        pids = [worker.pid for worker in loader._iterator._workers]
        dataset.set_epoch(1)
        second = list(loader)
        assert pids == [worker.pid for worker in loader._iterator._workers]
        assert len(first) == len(second) == 4
        assert not torch.equal(torch.cat([x for x, _ in first]),
                               torch.cat([x for x, _ in second]))
        # Fresh dataset/workers starting directly at epoch 1 must replay both
        # the sample order and all three augmentation RNG streams exactly.
        restored_dataset, resumed = make_loader()
        restored_dataset.set_epoch(1)
        replay = list(resumed)
        for (x, y), (rx, ry) in zip(second, replay, strict=True):
            assert torch.equal(x, rx) and torch.equal(y, ry)
        dataset.set_epoch(0)
        repeated = list(loader)
        for (x, y), (rx, ry) in zip(first, repeated, strict=True):
            assert torch.equal(x, rx) and torch.equal(y, ry)
    finally:
        for current in (loader, resumed):
            if current is not None and current._iterator is not None:
                current._iterator._shutdown_workers()


def test_webdataset_zero_worker_loaders_keep_independent_generators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("webdataset")
    root = _write_webdataset_root(tmp_path, monkeypatch)
    plan = BatchingPlan(
        world_size=1,
        physical_batch_size=4,
        effective_batch_size=8,
        grad_accum=2,
        samples_per_epoch=8,
        updates_per_epoch=1,
    )
    monkeypatch.setattr(imagenet, "resolve_batching_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        imagenet,
        "build_train_transform",
        lambda _run: lambda _image: torch.zeros(3, 4, 4),
    )
    monkeypatch.setattr(
        imagenet,
        "build_eval_transform",
        lambda _run: lambda _image: torch.zeros(3, 4, 4),
    )
    train_loader, val_loader, train_dataset, received_plan, generators, data = build_loaders(
        load_run(_args(tmp_path, "--train-workers", "0", "--val-workers", "0")),
        root,
        _cpu_state(),
        requested_grad_accum=None,
    )
    assert received_plan is plan
    assert not train_loader.persistent_workers
    assert not val_loader.persistent_workers
    assert train_loader.generator is generators.train
    assert val_loader.generator is generators.validation
    assert not torch.equal(generators.train.get_state(), generators.validation.get_state())
    assert train_dataset.training
    assert data["streaming"]["source_views"] == 1


def test_source_revision_records_a_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    def check_output(command: tuple[str, ...], **_kwargs: object) -> str:
        if command == ("git", "rev-parse", "HEAD"):
            return "abc123\n"
        if command == ("git", "status", "--porcelain"):
            return " M experiments/imagenet.py\n"
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(imagenet.subprocess, "check_output", check_output)
    assert imagenet._source_revision() == {
        "git_commit": "abc123",
        "git_dirty": True,
    }


@pytest.mark.parametrize(
    ("tier", "expected"),
    (
        ("base", (768, 12, 12, 48, 224, 0.4)),
        ("tiny", (192, 12, 6, 16, 224, 0.0)),
    ),
)
def test_base_and_tiny_pretraining_recipes(tmp_path: Path, tier: str, expected: tuple[int, ...]) -> None:
    args = _args(tmp_path, "--tier", tier)
    run = load_run(args)
    assert (
        run.model["embed_dim"],
        run.model["depth"],
        run.model["num_heads"],
        run.model["rank"],
        run.model["image_size"],
        run.model["drop_path_rate"],
    ) == expected
    assert run.train["optimizer"] == "fused_adamw"


@pytest.mark.parametrize("world_size,batch,accum", [(1, 512, 2), (2, 512, 1), (8, 128, 1), (2, 64, 8)])
def test_batching_plan_preserves_selected_global_batch(tmp_path, world_size, batch, accum):
    run = load_run(_args(tmp_path, "--batch-size", str(batch)))
    plan = resolve_batching_plan(run, _cpu_state(world_size=world_size),
                                dataset_size=1_281_167, requested_grad_accum=None)
    assert plan.effective_batch_size == 1024
    assert plan.grad_accum == accum
    assert plan.updates_per_epoch == 1251
    assert plan.samples_per_epoch == 1_281_024


def test_batching_plan_rejects_non_equivalent_physical_schedules(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="physical batch_size"):
        resolve_batching_plan(
            load_run(_args(tmp_path, "--batch-size", "96")),
            _cpu_state(),
            dataset_size=1_281_167,
            requested_grad_accum=None,
        )
    with pytest.raises(ValueError, match="must equal effective_batch"):
        resolve_batching_plan(
            load_run(_args(tmp_path, "--batch-size", "512", "--grad-accum", "3")),
            _cpu_state(),
            dataset_size=1_281_167,
            requested_grad_accum=3,
        )


def test_webdataset_contract_rejects_a_changed_manifest_or_streaming_policy() -> None:
    data = _data_contract()
    imagenet._validate_webdataset_contract(data)
    changed_manifest = copy.deepcopy(data)
    changed_manifest["manifest_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="manifest digest"):
        imagenet._validate_webdataset_contract(changed_manifest)

    changed_streaming = copy.deepcopy(data)
    changed_streaming["streaming"]["source_views"] = 2
    with pytest.raises(ValueError, match="source view"):
        imagenet._validate_webdataset_contract(changed_streaming)


def test_scheduler_uses_optimizer_updates_and_restores_exactly(tmp_path):
    pytest.importorskip("timm")
    run = load_run(_args(tmp_path))
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=.001)
    scheduler = build_scheduler(optimizer, run, 1251)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    scheduler.step_update(1251)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6 + (.001 - 1e-6) / 20)
    scheduler.step_update(20 * 1251)
    expected = 1e-5 + (.001 - 1e-5) * (1 + __import__('math').cos(__import__('math').pi * 20 / 300)) / 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    restored_opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=.001)
    restored = build_scheduler(restored_opt, run, 1251)
    restored_opt.load_state_dict(optimizer.state_dict())
    restored.load_state_dict(scheduler.state_dict())
    assert restored_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for update in (20 * 1251 + 1, 150 * 1251, 300 * 1251):
        scheduler.step_update(update)
        restored.step_update(update)
        assert restored_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_recipe_fidelity_marks_diagnostic_duration(tmp_path):
    for extra, expected in [((), "vit3-derived"), (("--epochs", "30"), "explicitly-modified")]:
        run = load_run(_args(tmp_path, *extra))
        assert _recipe_fidelity(run, batching_plan=_checkpoint_batching_plan(),
                               resolved_optimizer="torch.adamw.fused") == expected


def test_train_epoch_mixup_receives_complete_physical_batch(tmp_path, monkeypatch) -> None:
    class RecordingMixup:
        def __init__(self):
            self.calls = []

        def __call__(self, images, targets):
            self.calls.append(targets.tolist())
            # Pair across the former 128-sample boundary.
            return images, targets.flip(0)

    class Scheduler:
        def step_update(self, index):
            assert index == 0

    torch.manual_seed(8)
    batch_size = 256
    dataset = TensorDataset(torch.randn(batch_size, 2), torch.arange(batch_size))
    model = torch.nn.Linear(2, batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    mixup = RecordingMixup()
    run = ImageNetRun(config_path=tmp_path / "test.toml", tier="test", phase="test",
                      model={}, operator={}, train={"bce_loss": False, "clip_grad": 5.0},
                      overrides=())
    plan = BatchingPlan(world_size=1, physical_batch_size=batch_size,
                        effective_batch_size=batch_size, grad_accum=1,
                        samples_per_epoch=batch_size, updates_per_epoch=1)
    monkeypatch.setattr(imagenet, "_autocast", lambda: nullcontext())
    train_epoch(model, DataLoader(dataset, batch_size=batch_size), SequentialSampler(dataset),
                torch.nn.CrossEntropyLoss(), optimizer, mixup, Scheduler(),
                epoch=0, state=_cpu_state(), run=run, batching_plan=plan, print_freq=0)
    assert mixup.calls == [list(range(batch_size))]


def test_gradient_accumulation_matches_one_effective_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingScheduler:
        def __init__(self) -> None:
            self.updates = 0

        def step_update(self, index: int) -> None:
            assert index == 0
            self.updates += 1

    features = torch.tensor(
        ((1.0, -1.0), (0.5, 1.0), (-1.0, 0.25), (1.5, 0.5))
    )
    targets = torch.tensor((0, 1, 1, 0))
    dataset = TensorDataset(features, targets)
    model = torch.nn.Linear(2, 2, bias=False)
    reference = copy.deepcopy(model)
    criterion = torch.nn.CrossEntropyLoss()

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    criterion(reference(features), targets).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
    reference_optimizer.step()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RecordingScheduler()
    run = ImageNetRun(
        config_path=tmp_path / "test.toml",
        tier="test",
        phase="test",
        model={},
        operator={},
        train={"bce_loss": False, "clip_grad": 0.1},
        overrides=(),
    )
    batching_plan = BatchingPlan(
        world_size=1,
        physical_batch_size=2,
        effective_batch_size=4,
        grad_accum=2,
        samples_per_epoch=4,
        updates_per_epoch=1,
    )
    monkeypatch.setattr(imagenet, "_autocast", lambda: nullcontext())
    train_epoch(
        model,
        DataLoader(dataset, batch_size=2),
        SequentialSampler(dataset),
        criterion,
        optimizer,
        None,
        scheduler,
        epoch=0,
        state=_cpu_state(),
        run=run,
        batching_plan=batching_plan,
        print_freq=0,
    )
    torch.testing.assert_close(model.weight, reference.weight)
    assert scheduler.updates == 1


def test_checkpoint_model_state_preserves_ridgon_contract_entries() -> None:
    from ridgon import Ridgon, RidgonConfig

    source = Ridgon(RidgonConfig(16, 2, rank=4))
    restored = _checkpoint_model_state({"model": source.state_dict()})
    assert isinstance(restored["_extra_state"], dict)

    target = Ridgon(RidgonConfig(16, 2, rank=4))
    target.load_state_dict(restored, strict=True)
    torch.testing.assert_close(target.core_delta, source.core_delta)

    with pytest.raises(ValueError, match="Ridgon _extra_state"):
        _checkpoint_model_state({"model": {"unexpected": {}}})


def test_evaluate_preserves_weighted_metrics_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    features = torch.tensor(((4.0, 0.0), (0.0, 4.0), (-4.0, 0.0)))
    targets = torch.tensor((0, 1, 1))
    monkeypatch.setattr(imagenet, "_autocast", lambda: nullcontext())

    loss, accuracy1, accuracy5 = imagenet.evaluate(
        model,
        DataLoader(TensorDataset(features, targets), batch_size=2),
        state=_cpu_state(),
    )

    expected_loss = torch.nn.functional.cross_entropy(model(features), targets).item()
    assert loss == pytest.approx(expected_loss)
    assert accuracy1 == 100.0
    assert accuracy5 == 100.0


def test_checkpoint_round_trip_has_no_ema(
    tmp_path: Path,
) -> None:
    pytest.importorskip("timm")

    torch.manual_seed(3)
    source = torch.nn.Linear(2, 2)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=3)
    source_optimizer.step()
    source_scheduler.step()
    run = load_run(_args(tmp_path))
    batching_plan = _checkpoint_batching_plan()
    source_generators = _loader_generators(23)
    payload = _checkpoint(
        epoch=4,
        model=source,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        run=run,
        batching_plan=batching_plan,
        data_contract=_data_contract(),
        best_acc1=73.5,
        rng=_capture_resume_rng_state(_cpu_state(), source_generators),
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    target = torch.nn.Linear(2, 2)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=3)
    target_generators = _loader_generators(47)
    start_epoch, best_acc1 = _load_resume(
        path,
        model=target,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        run=run,
        batching_plan=batching_plan,
        data_contract=_data_contract(),
        state=_cpu_state(),
        generators=target_generators,
    )
    assert (start_epoch, best_acc1) == (5, 73.5)
    for source_parameter, target_parameter in zip(
        source.parameters(),
        target.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(target_parameter, source_parameter)
    assert "model_ema" not in payload
    assert "scaler" not in payload
    assert target_scheduler.state_dict() == source_scheduler.state_dict()

    changed_physical_plan = BatchingPlan(
        world_size=1,
        physical_batch_size=512,
        effective_batch_size=2048,
        grad_accum=4,
        samples_per_epoch=1_280_000,
        updates_per_epoch=625,
    )
    with pytest.raises(ValueError, match="does not match"):
        _load_resume(
            path,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            run=run,
            batching_plan=changed_physical_plan,
            data_contract=_data_contract(),
            state=_cpu_state(),
            generators=target_generators,
        )

    changed_data = _data_contract()
    changed_data["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        _load_resume(
            path,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            run=run,
            batching_plan=batching_plan,
            data_contract=changed_data,
            state=_cpu_state(),
            generators=target_generators,
        )


def test_resume_rng_state_replays_python_numpy_torch_and_loader_generators() -> None:
    numpy = pytest.importorskip("numpy")
    state = _cpu_state()
    source_generators = _loader_generators(71)
    random.seed(13)
    numpy.random.seed(17)
    torch.manual_seed(19)
    saved = _capture_resume_rng_state(state, source_generators)

    expected_python = random.random()
    expected_numpy = numpy.random.random(5)
    expected_torch = torch.rand(5)
    expected_train_generator = torch.rand(5, generator=source_generators.train)
    expected_validation_generator = torch.rand(
        5, generator=source_generators.validation
    )

    random.random()
    numpy.random.random(5)
    torch.rand(5)
    torch.rand(5, generator=source_generators.train)
    torch.rand(5, generator=source_generators.validation)

    restored_generators = _loader_generators(101)
    _restore_resume_rng_state(
        saved,
        state=state,
        generators=restored_generators,
    )
    assert random.random() == expected_python
    numpy.testing.assert_array_equal(numpy.random.random(5), expected_numpy)
    assert torch.equal(torch.rand(5), expected_torch)
    assert torch.equal(
        torch.rand(5, generator=restored_generators.train),
        expected_train_generator,
    )
    assert torch.equal(
        torch.rand(5, generator=restored_generators.validation),
        expected_validation_generator,
    )


def test_nonpersistent_worker_rng_replays_from_its_loader_generator() -> None:
    pytest.importorskip("numpy")
    state = _cpu_state()
    source_generators = _loader_generators(113)
    loader = _worker_random_loader(source_generators.train)
    list(loader)
    saved = _capture_resume_rng_state(state, source_generators)

    expected = [batch.clone() for batch in loader]
    restored_generators = _loader_generators(127)
    _restore_resume_rng_state(
        saved,
        state=state,
        generators=restored_generators,
    )
    actual = [batch.clone() for batch in _worker_random_loader(restored_generators.train)]

    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        assert torch.equal(actual_batch, expected_batch)


def test_resume_rejects_missing_rng_before_loading_model(tmp_path: Path) -> None:
    run = load_run(_args(tmp_path))
    batching_plan = _checkpoint_batching_plan()
    source = torch.nn.Linear(2, 2)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=3)
    payload = _checkpoint(
        epoch=0,
        model=source,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        run=run,
        batching_plan=batching_plan,
        data_contract=_data_contract(),
        best_acc1=0.0,
        rng=_capture_resume_rng_state(_cpu_state(), _loader_generators(131)),
    )
    payload.pop("rng")
    path = tmp_path / "missing_rng.pt"
    torch.save(payload, path)

    target = torch.nn.Linear(2, 2)
    target_before = copy.deepcopy(target.state_dict())
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=3)
    with pytest.raises(ValueError, match="RNG state"):
        _load_resume(
            path,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            run=run,
            batching_plan=batching_plan,
            data_contract=_data_contract(),
            state=_cpu_state(),
            generators=_loader_generators(137),
        )
    for name, parameter in target.state_dict().items():
        assert torch.equal(parameter, target_before[name])


def test_atomic_checkpoint_write_preserves_the_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    previous = {"value": torch.tensor((1, 2, 3))}
    torch.save(previous, path)

    def interrupted_save(_value: object, handle: object) -> None:
        handle.write(b"partial")  # type: ignore[union-attr]
        raise OSError("simulated preemption")

    monkeypatch.setattr(imagenet.torch, "save", interrupted_save)
    with pytest.raises(OSError, match="simulated preemption"):
        _atomic_torch_save({"value": torch.tensor((4, 5, 6))}, path)

    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(restored["value"], previous["value"])
    assert not list(tmp_path.glob(".checkpoint.pt.*.tmp"))


def test_atomic_checkpoint_write_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    expected = {"value": torch.tensor((4, 5, 6))}
    _atomic_torch_save(expected, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(restored["value"], expected["value"])


def test_adamw_parameters_and_weight_decay_exclusions(tmp_path):
    pytest.importorskip("timm")
    model = torch.nn.Linear(4, 2)
    optimizer, name = build_optimizer(model, load_run(_args(tmp_path)))
    assert name == "torch.adamw.fused"
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["fused"] is True
    assert optimizer.defaults["betas"] == (.9, .999)
    for group in optimizer.param_groups:
        assert group["lr"] == .001
        for parameter in group["params"]:
            assert group["weight_decay"] == (0 if parameter is model.bias else .05)



def test_imagenet_optimizer_uses_plain_fused_adamw_with_delta_weight_decay(tmp_path):
    from ridgon import Ridgon, RidgonConfig
    model = Ridgon(RidgonConfig(16, 2, rank=4))
    optimizer, _ = build_optimizer(model, load_run(_args(tmp_path)))
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.defaults["fused"] is True
    assert not optimizer._optimizer_step_pre_hooks
    assert not optimizer._optimizer_step_post_hooks
    delta_groups = [group for group in optimizer.param_groups
                    if any(p is model.core_delta for p in group["params"])]
    assert len(delta_groups) == 1
    assert delta_groups[0]["weight_decay"] == .05
    assert torch.count_nonzero(model.core_delta) == 0



def test_custom_recipe_is_not_mislabeled_as_vit3(tmp_path):
    run = load_run(_args(tmp_path))
    run.train["lr"] = .002
    assert _recipe_fidelity(run, batching_plan=_checkpoint_batching_plan(),
                            resolved_optimizer="torch.adamw.fused") == "explicitly-modified"


def test_vit3_transforms_and_soft_targets(tmp_path):
    pytest.importorskip("timm")
    from PIL import Image
    run = load_run(_args(tmp_path))
    transform = imagenet.build_train_transform(run)
    validation = imagenet.build_eval_transform(run)
    assert validation.transforms[0].size == 256
    image = Image.new("RGB", (300, 280), (100, 150, 200))
    assert transform(image).shape == (3, 224, 224)
    assert validation(image).shape == (3, 224, 224)
    mixup, loss = imagenet.build_mixup_and_loss(run)
    images, targets = mixup(torch.zeros(2, 3, 224, 224), torch.tensor([0, 1]))
    torch.testing.assert_close(targets.sum(1), torch.ones(2))
    assert (targets > 0).all()  # smoothing, not binary multi-label targets
    assert torch.isfinite(loss(torch.zeros(2, 1000), targets))


def test_graph_step_rejects_gradient_accumulation():
    with pytest.raises(ValueError, match="grad_accum=1"):
        imagenet.ImageNetGraphStep(torch.nn.Linear(2, 2), torch.nn.CrossEntropyLoss(), grad_accum=2)


@pytest.mark.cuda
def test_graph_step_preserves_warmup_rng_updates_inputs_and_reuses_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(916)
    eager = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(16, 5)).cuda()
    graphed = copy.deepcopy(eager)
    criterion = torch.nn.CrossEntropyLoss()
    engine = imagenet.ImageNetGraphStep(graphed, criterion)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True) for m in (eager, graphed)]
    for step in range(3):
        images = torch.randn(16, 8, device="cuda") + step * .1
        targets = (torch.arange(16, device="cuda") + step) % 5
        rng = torch.cuda.get_rng_state()
        eager.zero_grad(set_to_none=True)
        with imagenet._autocast():
            expected_logits = eager(images)
            expected_loss = criterion(expected_logits, targets)
        expected_loss.backward()
        expected_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
        graphed.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in graphed.parameters()]
        logits, loss = engine(images, targets)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), expected_rng, rtol=0, atol=0)
        for a, b, old in zip(graphed.parameters(), eager.parameters(), before):
            torch.testing.assert_close(a, old, rtol=0, atol=0)
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        for model, optimizer in zip((eager, graphed), optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.param_groups[0]['lr'] = .001 / (step + 1)
            optimizer.step()
        for a, b in zip(graphed.parameters(), eager.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        graphed.eval()
        with torch.no_grad():
            graphed(images)
        graphed.train()
    with pytest.raises(ValueError, match="shape/dtype/device"):
        engine(images[:8], targets[:8])
