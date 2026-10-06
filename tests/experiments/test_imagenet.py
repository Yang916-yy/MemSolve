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
    VIT3_CONFIG,
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
            run_name="memsolve_direct_launcher_test",
        )
    finally:
        sys.path[:] = original_path
    assert callable(namespace["main"])


def _args(tmp_path: Path, *extra: str):
    return parse_args(
        [
            "--config",
            str(VIT3_CONFIG),
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


def _data_contract(*, source_views: int = 1, group_size: int | None = None) -> dict[str, object]:
    result = {
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
    if group_size is not None:
        result["streaming"].update(
            shard_order="global-sample-permutation-then-virtual-group-rank-stride",
            sample_shuffle={"algorithm": "torch.randperm", "seed_policy": "seed-epoch-v1"},
            repeated_augmentation_placement="global-virtual-group-repeat-then-rank-stride",
            augmentation_group_size=group_size,
        )
    return result


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
        "architecture": "vit3_qkconv_rope2d_mean_swiglu_v4", "ffn": "swiglu",
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
        "position_encoding": "rope_2d_axial",
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
        "qk_conv_kernel_size": 3,
        "output_gate_rank": 32,
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
    pytest.importorskip("wids")
    root = _write_webdataset_root(tmp_path, monkeypatch)
    manifest = imagenet.load_imagenet_webdataset_manifest(root)
    calls: list[int] = []

    def counting_transform(_image: object) -> torch.Tensor:
        calls.append(len(calls))
        return torch.tensor([calls[-1]], dtype=torch.int64)

    dataset = imagenet._ImageNetIndexedDataset(
        manifest.train,
        transform=counting_transform,
        state=_cpu_state(),
        seed=17,
        num_classes=3,
    )
    sampler = imagenet.VirtualGroupSampler(dataset, samples_per_rank=12, group_size=4)
    sampler.set_epoch(4)
    batches = iter(DataLoader(dataset, sampler=sampler, batch_size=4, num_workers=0, drop_last=True))
    batch = next(batches)
    assert batch[0][:, 0].tolist() == [0, 1, 2, 3]
    assert calls == [0, 1, 2, 3]
    next_view = next(batches)
    assert next_view[0][:, 0].tolist() == [4, 5, 6, 7]
    assert torch.equal(batch[1], next_view[1])

    def stable_transform(image: object) -> torch.Tensor:
        return torch.tensor(image.getpixel((0, 0)), dtype=torch.int64)  # type: ignore[union-attr]

    replayable = imagenet._ImageNetIndexedDataset(
        manifest.train,
        transform=stable_transform,
        state=_cpu_state(),
        seed=17,
        num_classes=3,
    )
    sampler = imagenet.VirtualGroupSampler(replayable, samples_per_rank=12, group_size=4)
    sampler.set_epoch(7)
    first = next(iter(DataLoader(replayable, sampler=sampler, batch_size=4, num_workers=0, drop_last=True)))
    sampler.set_epoch(7)
    second = next(iter(DataLoader(replayable, sampler=sampler, batch_size=4, num_workers=0, drop_last=True)))
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
    pytest.importorskip("wids")
    root = _write_webdataset_root(tmp_path, monkeypatch, train_shards=4)
    manifest = imagenet.load_imagenet_webdataset_manifest(root)
    dataset = imagenet._ImageNetIndexedDataset(
        manifest.train,
        transform=_pixel_code_transform,
        state=_cpu_state(),
        seed=29,
        num_classes=3,
    )
    sampler = imagenet.VirtualGroupSampler(dataset, samples_per_rank=48, group_size=2)

    def collect() -> list[torch.Tensor]:
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=4,
            drop_last=True,
            num_workers=2,
            persistent_workers=False,
            prefetch_factor=1,
        )
        return [images.clone() for _, (images, _targets) in zip(range(12), loader)]

    sampler.set_epoch(5)
    first = collect()
    sampler.set_epoch(5)
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
    assert all(len(set(group)) == 2 for group in source_groups)
    # Repeated groups can share a physical batch; members within each virtual
    # group remain unique, and sample order is independent of loader workers.
    reference = DataLoader(dataset, sampler=sampler, batch_size=4, num_workers=0)
    assert all(torch.equal(a, b) for a, (b, _) in zip(first, reference, strict=True))


@pytest.mark.parametrize("world_size", (1, 2, 3))
@pytest.mark.parametrize("group_size", (1, 2, 4, 8))
def test_virtual_ra_repeats_globally_before_rank_split(world_size, group_size):
    split = imagenet._WebDatasetSplit("train", (), (), 256)
    local_groups = []
    for rank in range(world_size):
        dataset = imagenet._ImageNetIndexedDataset(split, transform=None,
            state=_cpu_state(world_size=world_size, rank=rank), seed=0, num_classes=1)
        sampler = imagenet.VirtualGroupSampler(dataset, samples_per_rank=17*group_size,
                                               group_size=group_size, shuffle=False)
        indices = list(sampler)
        assert len(indices) == 17*group_size
        local_groups.append([tuple(indices[i:i+group_size]) for i in range(0, len(indices), group_size)])
    global_groups = [local_groups[rank][i] for i in range(17) for rank in range(world_size)]
    for index, group in enumerate(global_groups):
        source = index // 3
        assert group == tuple(range(source*group_size, (source+1)*group_size))
        assert len(set(group)) == group_size
    assert max(Counter(sample for group in global_groups for sample in group).values()) <= 3


def test_virtual_ra_views_stay_in_the_same_or_adjacent_optimizer_update():
    split = imagenet._WebDatasetSplit("train", (), (), 32)
    ranks = []
    for rank in (0, 1):
        dataset = imagenet._ImageNetIndexedDataset(split, transform=None,
            state=_cpu_state(world_size=2, rank=rank), seed=0, num_classes=1)
        ranks.append(list(imagenet.VirtualGroupSampler(
            dataset, samples_per_rank=16, group_size=2, shuffle=False)))
    source_updates = {}
    for update in range(2):
        batch = [index for rank in ranks for index in rank[update*8:(update+1)*8]]
        for source in batch:
            source_updates.setdefault(source, set()).add(update)
        if update == 0:
            assert Counter(batch) == Counter({0: 3, 1: 3, 2: 3, 3: 3, 4: 2, 5: 2})
    assert all(max(updates)-min(updates) <= 1 for updates in source_updates.values())
    # No padding or cycling can silently fill an insufficient source quota.
    short = imagenet._ImageNetIndexedDataset(imagenet._WebDatasetSplit("train", (), (), 3),
        transform=None, state=_cpu_state(), seed=0, num_classes=1)
    with pytest.raises(ValueError, match="unique-source"):
        imagenet.VirtualGroupSampler(short, samples_per_rank=12, group_size=2)


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
        physical_batch_size=4,
        microbatches_per_epoch=4,
        worker_count=1,
    )
    with pytest.raises(RuntimeError, match="cannot cover the planned epoch"):
        next(iter(DataLoader(short, batch_size=4, num_workers=0, drop_last=True)))


@pytest.mark.parametrize("virtual_groups", (False, True))
def test_webdataset_persistent_workers_observe_epochs_and_replay_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, virtual_groups: bool,
) -> None:
    pytest.importorskip("webdataset")
    if virtual_groups:
        pytest.importorskip("wids")
    root = _write_webdataset_root(tmp_path, monkeypatch, train_shards=4)
    split = imagenet.load_imagenet_webdataset_manifest(root).train

    def make_loader():
        if virtual_groups:
            dataset = imagenet._ImageNetIndexedDataset(
                split, transform=_random_pixel_transform, state=_cpu_state(), seed=29, num_classes=3)
            controller = imagenet.VirtualGroupSampler(dataset, samples_per_rank=16, group_size=2)
        else:
            dataset = imagenet._ImageNetWebDataset(
                split, transform=_random_pixel_transform, state=_cpu_state(),
                seed=29, num_classes=3, training=True,
                physical_batch_size=4, microbatches_per_epoch=4, worker_count=2,
            )
            controller = dataset
        loader = DataLoader(dataset, batch_size=4, drop_last=True,
                            **({"sampler": controller} if virtual_groups else {}),
                            **imagenet._loader_kwargs(
                                workers=2, generator=torch.Generator().manual_seed(97)))
        return controller, loader

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


@pytest.mark.parametrize("group_size", (None, 128))
def test_train_epoch_mixup_grouping_keeps_one_physical_forward(tmp_path, monkeypatch, group_size) -> None:
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
    forwards = []
    model.register_forward_pre_hook(lambda _model, args: forwards.append(args[0].shape[0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    mixup = RecordingMixup()
    run = ImageNetRun(config_path=tmp_path / "test.toml", tier="test", phase="test",
                      model={}, operator={}, train={"bce_loss": False, "clip_grad": 5.0, "optimizer": "sgd"},
                      overrides=())
    plan = BatchingPlan(world_size=1, physical_batch_size=batch_size,
                        effective_batch_size=batch_size, grad_accum=1,
                        samples_per_epoch=batch_size, updates_per_epoch=1,
                        augmentation_group_size=group_size)
    monkeypatch.setattr(imagenet, "_autocast", lambda: nullcontext())
    train_epoch(model, DataLoader(dataset, batch_size=batch_size), SequentialSampler(dataset),
                torch.nn.CrossEntropyLoss(), optimizer, mixup, Scheduler(),
                epoch=0, state=_cpu_state(), run=run, batching_plan=plan, print_freq=0)
    assert mixup.calls == ([list(range(batch_size))] if group_size is None
                          else [list(range(start, start+group_size)) for start in range(0, batch_size, group_size)])
    assert forwards == [batch_size]


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
        train={"bce_loss": False, "clip_grad": 0.1, "optimizer": "sgd",
               "ema": True, "ema_decay": 0.99996},
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
    model_ema = imagenet.build_model_ema(model, run)
    initial_weight = model.weight.detach().clone()
    ema_updates = []
    original_update = model_ema.update
    def record_update(source):
        ema_updates.append(1)
        original_update(source)
    monkeypatch.setattr(model_ema, "update", record_update)
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
        model_ema=model_ema,
    )
    torch.testing.assert_close(model.weight, reference.weight)
    assert scheduler.updates == 1
    assert len(ema_updates) == 1
    torch.testing.assert_close(model_ema.module.weight,
        initial_weight.lerp(model.weight.detach(), 1 - .99996), rtol=0, atol=0)


def test_checkpoint_model_state_preserves_memsolve_contract_entries() -> None:
    from memsolve import MemSolve, MemSolveConfig

    source = MemSolve(MemSolveConfig(16, 2, rank=4))
    restored = _checkpoint_model_state({"model": source.state_dict()})
    assert isinstance(restored["_extra_state"], dict)

    target = MemSolve(MemSolveConfig(16, 2, rank=4))
    target.load_state_dict(restored, strict=True)
    torch.testing.assert_close(target.core_delta, source.core_delta)

    with pytest.raises(ValueError, match="MemSolve _extra_state"):
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


def test_dual_evaluation_uses_distinct_weights_and_one_data_pass(monkeypatch):
    from types import SimpleNamespace
    model = torch.nn.Linear(2, 2, bias=False)
    shadow = copy.deepcopy(model)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
        shadow.weight.copy_(-torch.eye(2))
    features = torch.tensor(((4., 0.), (0., 4.), (-4., 0.)))
    targets = torch.tensor((0, 1, 1))
    seen = []
    def batches():
        for start in (0, 2):
            seen.append(start)
            yield features[start:start+2], targets[start:start+2]
    monkeypatch.setattr(imagenet, '_autocast', lambda: nullcontext())
    results = imagenet.evaluate_models(model, batches(), state=_cpu_state(),
                                       model_ema=SimpleNamespace(module=shadow))
    assert seen == [0, 2]
    assert results['val_acc1'] == 100. and results['ema_val_acc1'] == 0.
    assert results['val_acc5'] == results['ema_val_acc5'] == 100.
    for key, variant in [('val', model), ('ema_val', shadow)]:
        assert results[f'{key}_loss'] == pytest.approx(
            torch.nn.functional.cross_entropy(variant(features), targets).item())


def test_main_selects_independent_raw_and_ema_best(tmp_path, monkeypatch):
    state = _cpu_state()
    output = tmp_path / 'run'
    args = ['--tier', 'tiny', '--data-root', str(tmp_path), '--output', str(output),
            '--epochs', '3', '--execution', 'eager']
    run = load_run(parse_args(args))
    plan = resolve_batching_plan(run, state, dataset_size=1_281_167, requested_grad_accum=None)
    monkeypatch.setattr(imagenet, 'initialize_distributed', lambda: state)
    monkeypatch.setattr(imagenet, 'prepare_operator_backend', lambda *args: None)
    monkeypatch.setattr(imagenet, 'build_loaders', lambda *args, **kwargs:
        ([], [], None, plan, _loader_generators(), _data_contract(source_views=3, group_size=256)))
    monkeypatch.setattr(imagenet, 'build_model', lambda run: torch.nn.Linear(2, 2))
    def prepare(model, *args):
        # Simulate compile's bound forward closure. EMA must be copied before it.
        original = model.forward
        model.forward = lambda x: original(x)
        return model
    monkeypatch.setattr(imagenet, 'prepare_training_model', prepare)
    monkeypatch.setattr(imagenet, 'build_optimizer', lambda model, run:
        (torch.optim.SGD(model.parameters(), lr=.1), 'apex.lamb.fused'))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'test-cpu')
    def train(model, *args, epoch, model_ema, **kwargs):
        with torch.no_grad():
            model.weight.fill_(epoch+1)
            model_ema.module.weight.fill_(10*(epoch+1))
        return 1., 50., 100.
    def validate(model, *args, model_ema, **kwargs):
        assert model_ema.module.forward.__self__ is model_ema.module
        epoch = int(model.weight[0, 0].item()) - 1
        assert model_ema.module.weight[0, 0].item() == 10*(epoch+1)
        return {'val_acc1': [80., 79., 81.][epoch], 'val_loss': 1., 'val_acc5': 95.,
                'ema_val_acc1': [75., 82., 80.][epoch], 'ema_val_loss': .9, 'ema_val_acc5': 96.}
    monkeypatch.setattr(imagenet, 'train_epoch', train)
    monkeypatch.setattr(imagenet, 'evaluate_models', validate)
    imagenet.main(args)
    raw = torch.load(output / 'checkpoint_best.pt', weights_only=False)
    ema = torch.load(output / 'checkpoint_best_ema.pt', weights_only=False)
    last = torch.load(output / 'checkpoint_last.pt', weights_only=False)
    assert (raw['epoch'], ema['epoch'], last['epoch']) == (2, 1, 2)
    assert raw['selected_weights'] == 'model' and ema['selected_weights'] == 'model_ema'
    assert raw['model']['weight'][0, 0] == 3 and ema['model_ema']['weight'][0, 0] == 20
    assert (last['best_acc1'], last['best_ema_acc1']) == (81., 82.)
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert records[-1]['best_val_acc1'] == 81. and records[-1]['best_ema_val_acc1'] == 82.
    assert records[-1]['ema_val_acc1'] == 80.
    # Eval-only reports both loaded weight sets, without stepping the optimizer.
    imagenet.main([*args, '--resume', str(output / 'checkpoint_last.pt'), '--eval'])
    record = json.loads((output / 'metrics.jsonl').read_text().splitlines()[-1])
    assert record['event'] == 'evaluation'
    assert record['val_acc1'] == 81. and record['ema_val_acc1'] == 80.


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
    start_epoch, best_acc1, best_ema_acc1 = _load_resume(
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
    assert best_ema_acc1 is None
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


def test_ropevit_ema_preserves_contract_and_restores_shadow(tmp_path):
    from integrations.timm import create_memsolve_vit

    run = load_run(parse_args(['--tier', 'base', '--data-root', str(tmp_path),
                              '--output', str(tmp_path / 'run')]))
    assert run.train['ema'] is True and run.train['ema_decay'] == .99996
    model = create_memsolve_vit(image_size=16, num_classes=2, embed_dim=32,
        depth=1, num_heads=2, rank=16, mlp_ratio=4., bias=True)
    initial = copy.deepcopy(model)
    ema = imagenet.build_model_ema(model, run)
    assert not ema.module.training
    assert not any(p.requires_grad for p in ema.module.parameters())
    assert not list(model.buffers())  # No running statistics to average.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.125)
    ema.update(model)
    for shadow, before, current in zip(ema.module.parameters(), initial.parameters(), model.parameters()):
        torch.testing.assert_close(shadow, before.detach().lerp(current.detach(), 1 - .99996),
                                   rtol=0, atol=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
    state, generators = _cpu_state(), _loader_generators(23)
    plan = resolve_batching_plan(run, state, dataset_size=1_281_167, requested_grad_accum=None)
    data = _data_contract(source_views=3, group_size=256)
    payload = _checkpoint(epoch=4, model=model, optimizer=optimizer, scheduler=scheduler,
        run=run, batching_plan=plan, data_contract=data, best_acc1=73.5,
        rng=_capture_resume_rng_state(state, generators), model_ema=ema, best_ema_acc1=72.0)
    path = tmp_path / 'checkpoint.pt'
    torch.save(payload, path)
    target = copy.deepcopy(initial)
    target_ema = imagenet.build_model_ema(target, run)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=.1)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=3)
    kwargs = dict(model=target, optimizer=target_optimizer, scheduler=target_scheduler,
        run=run, batching_plan=plan, data_contract=data, state=state,
        generators=generators, model_ema=target_ema)
    assert _load_resume(path, **kwargs) == (5, 73.5, 72.0)
    for key, value in ema.module.state_dict().items():
        restored = target_ema.module.state_dict()[key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(restored, value, rtol=0, atol=0)
        else:
            assert restored == value  # Non-tensor operator/architecture contracts.
    for a, b in zip(target.parameters(), model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Finetuning starts EMA from the loaded ordinary weights, not the old shadow.
    ft = load_run(parse_args(['--tier', 'base', '--phase', 'finetune', '--finetune', str(path),
                             '--data-root', str(tmp_path), '--output', str(tmp_path / 'ft')]))
    fresh = copy.deepcopy(initial)
    imagenet._load_finetune(path, model=fresh, run=ft)
    fresh_ema = imagenet.build_model_ema(fresh, ft)
    for shadow, current in zip(fresh_ema.module.parameters(), model.parameters()):
        torch.testing.assert_close(shadow, current, rtol=0, atol=0)
    payload['selected_weights'] = 'model_ema'
    torch.save(payload, path)
    source = imagenet._load_finetune(path, model=fresh, run=ft)
    assert source['source_weights'] == 'model_ema'
    for loaded, shadow in zip(fresh.parameters(), ema.module.parameters()):
        torch.testing.assert_close(loaded, shadow, rtol=0, atol=0)
    # Resume always restores raw optimizer-owned weights, even from EMA best.
    assert _load_resume(path, **kwargs) == (5, 73.5, 72.0)
    for loaded, raw in zip(target.parameters(), model.parameters()):
        torch.testing.assert_close(loaded, raw, rtol=0, atol=0)
    del payload['model_ema']
    torch.save(payload, path)
    with pytest.raises(ValueError, match='missing model_ema'):
        _load_resume(path, **kwargs)


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
    from memsolve import MemSolve, MemSolveConfig
    model = MemSolve(MemSolveConfig(16, 2, rank=4))
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


@pytest.mark.parametrize("accum", [0, -1, True])
def test_graph_step_rejects_invalid_accumulation(accum):
    with pytest.raises(ValueError, match="positive integer"):
        imagenet.ImageNetGraphStep(torch.nn.Linear(2, 2), torch.nn.CrossEntropyLoss(), grad_accum=accum)


@pytest.mark.parametrize('recipe,epochs,decay,droppath', [
    ('deit3_400', 400, .02, .1), ('deit3_800', 800, .05, .2),
])
def test_deit3_and_finetune_batching(tmp_path, recipe, epochs, decay, droppath):
    def selected(phase):
        flags = ['--finetune', str(tmp_path / 'pretrain.pt')] if phase == 'finetune' else []
        return load_run(parse_args(['--config', str(imagenet.DEIT3_CONFIGS[recipe]), '--tier', 'base',
            '--phase', phase, '--data-root', str(tmp_path), '--output', str(tmp_path / phase), *flags]))
    pretrain, finetune = selected('pretrain'), selected('finetune')
    state = _cpu_state(world_size=2)
    p = resolve_batching_plan(pretrain, state, dataset_size=1_281_167, requested_grad_accum=None)
    f = resolve_batching_plan(finetune, state, dataset_size=1_281_167, requested_grad_accum=None)
    assert (p.physical_batch_size, p.grad_accum, p.effective_batch_size, p.updates_per_epoch) == (512, 2, 2048, 625)
    assert (f.physical_batch_size, f.grad_accum, f.effective_batch_size, f.updates_per_epoch) == (256, 1, 512, 2502)
    assert (p.augmentation_group_size, f.augmentation_group_size) == (256, 64)
    assert (pretrain.model['image_size'], pretrain.train['epochs'], pretrain.train['weight_decay']) == (192, epochs, decay)
    assert (finetune.model['image_size'], finetune.train['epochs'], finetune.train['weight_decay']) == (224, 20, .1)
    assert pretrain.model['drop_path_schedule'] == finetune.model['drop_path_schedule'] == 'constant'
    assert pretrain.model['drop_path_rate'] == finetune.model['drop_path_rate'] == droppath
    assert pretrain.train['bce_loss'] and not finetune.train['bce_loss']
    assert _recipe_fidelity(pretrain, batching_plan=p, resolved_optimizer='apex.lamb.fused') == 'deit3-derived'
    assert _recipe_fidelity(finetune, batching_plan=f, resolved_optimizer='torch.adamw.fused') == 'deit3-derived'


@pytest.mark.parametrize('tier', ['tiny', 'small'])
def test_default_ts_lamb_recipe_batches_augments_and_records_provenance(tmp_path, tier):
    from PIL import Image
    run = load_run(parse_args(['--tier', tier, '--data-root', str(tmp_path),
                              '--output', str(tmp_path / tier)]))
    plan = resolve_batching_plan(run, _cpu_state(world_size=2),
                                dataset_size=1_281_167, requested_grad_accum=None)
    assert run.config_path == imagenet.ROPEVIT_CONFIG
    assert run.train['ema'] is True and run.train['ema_decay'] == .99996
    assert (run.train['epochs'], run.train['lr'], run.train['weight_decay']) == (400, .004, .03)
    assert run.model['drop_path_rate'] == 0 and run.model['drop_path_schedule'] == 'constant'
    assert (plan.physical_batch_size, plan.effective_batch_size, plan.grad_accum) == (512, 2048, 2)
    assert (plan.augmentation_group_size, plan.updates_per_epoch) == (256, 625)
    assert _recipe_fidelity(run, batching_plan=plan, resolved_optimizer='apex.lamb.fused') == 'ropevit-derived'
    assert _recipe_fidelity(run, batching_plan=plan, resolved_optimizer='torch.adamw.fused') == 'explicitly-modified'
    transform = imagenet.build_train_transform(run)
    assert transform(Image.new('RGB', (300, 280))).shape == (3, 224, 224)
    mixup, loss = imagenet.build_mixup_and_loss(run)
    _, targets = mixup(torch.zeros(256, 3, 4, 4), torch.arange(256))
    assert (targets > 0).sum(1).max() <= 2  # No smoothing before membership BCE.
    assert isinstance(loss, imagenet.DeiTBinaryCrossEntropy)
    assert torch.isfinite(loss(torch.zeros(256, 1000), targets))
    metadata = run.as_dict(plan, _data_contract(source_views=3, group_size=256))
    assert metadata['official_ropevit_recipe'] == imagenet.OFFICIAL_ROPEVIT_URL
    assert 'official_vit3_recipe' not in metadata and 'official_deit3_recipe' not in metadata
    run.train['epochs'] = 30
    assert _recipe_fidelity(run, batching_plan=plan, resolved_optimizer='apex.lamb.fused') == 'explicitly-modified'


@pytest.mark.parametrize('phase', ['pretrain', 'finetune'])
def test_default_base_ropevit_recipe_changes_resolution_and_regularization(tmp_path, phase):
    flags = ['--finetune', str(tmp_path / 'pretrain.pt')] if phase == 'finetune' else []
    run = load_run(parse_args(['--tier', 'base', '--phase', phase, '--data-root', str(tmp_path),
                              '--output', str(tmp_path / phase), *flags]))
    plan = resolve_batching_plan(run, _cpu_state(world_size=2),
                                dataset_size=1_281_167, requested_grad_accum=None)
    assert run.config_path == imagenet.ROPEVIT_CONFIG
    if phase == 'pretrain':
        assert (run.model['image_size'], run.train['epochs'], run.train['lr']) == (192, 400, .003)
        assert (run.model['drop_path_rate'], run.train['weight_decay']) == (.1, .03)
        assert (plan.physical_batch_size, plan.grad_accum, plan.effective_batch_size) == (512, 2, 2048)
        assert (plan.augmentation_group_size, plan.updates_per_epoch) == (256, 625)
        assert run.train['bce_loss'] and run.train['label_smoothing'] == 0
        assert run.train['repeated_aug'] and run.train['augmentation'] == 'three_augment'
        optimizer = 'apex.lamb.fused'
    else:
        assert (run.model['image_size'], run.train['epochs'], run.train['lr']) == (224, 20, 1e-5)
        assert (run.model['drop_path_rate'], run.train['weight_decay']) == (.2, .1)
        assert (plan.physical_batch_size, plan.grad_accum, plan.effective_batch_size) == (256, 1, 512)
        assert (plan.augmentation_group_size, plan.updates_per_epoch) == (64, 2502)
        assert not run.train['bce_loss'] and run.train['label_smoothing'] == .1
        assert not run.train['repeated_aug'] and run.train['augmentation'] == 'rand_augment'
        assert 'drop_path_rate' not in run.train  # The phase override belongs to the model.
        optimizer = 'torch.adamw.fused'
    assert run.model['drop_path_schedule'] == 'constant'
    assert _recipe_fidelity(run, batching_plan=plan, resolved_optimizer=optimizer) == 'ropevit-derived'
    run.model['drop_path_rate'] = .3
    assert _recipe_fidelity(run, batching_plan=plan, resolved_optimizer=optimizer) == 'explicitly-modified'


def test_virtual_augmentation_checkpoint_contract_rejects_inconsistent_groups(tmp_path):
    run = load_run(parse_args(['--config', str(imagenet.DEIT3_CONFIG), '--tier', 'base',
                              '--data-root', str(tmp_path), '--output', str(tmp_path / 'run')]))
    plan = resolve_batching_plan(run, _cpu_state(world_size=2),
                                dataset_size=1_281_167, requested_grad_accum=None)
    contract = run.checkpoint_contract(plan, _data_contract(source_views=3, group_size=256))
    def envelope(value):
        return {'format_version': imagenet.IMAGENET_CHECKPOINT_FORMAT, 'contract': value,
                'contract_digest': checkpoint_contract_digest(value)}
    assert imagenet.validate_checkpoint_contract(envelope(contract)) == contract
    for section in ('train', 'batching'):
        changed = copy.deepcopy(contract)
        changed[section]['augmentation_group_size'] = 128
        with pytest.raises(ValueError, match='inconsistent'):
            imagenet.validate_checkpoint_contract(envelope(changed))
    changed = copy.deepcopy(contract)
    changed['data']['streaming']['augmentation_group_size'] = 128
    with pytest.raises(ValueError, match='inconsistent'):
        imagenet.validate_checkpoint_contract(envelope(changed))


@pytest.mark.cuda
@pytest.mark.parametrize('tier,config', [('base', imagenet.DEIT3_CONFIG), ('small', imagenet.ROPEVIT_CONFIG)])
def test_fused_lamb_matches_mature_reference_and_updates_zero_delta(tmp_path, tier, config):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('apex.optimizers')
    from timm.optim import param_groups_weight_decay
    from timm.optim.lamb import Lamb
    torch.manual_seed(17)
    actual = torch.nn.Module()
    actual.core_delta = torch.nn.Parameter(torch.zeros(2, 4, 4, device='cuda'))
    actual.linear = torch.nn.Linear(8, 16, device='cuda')
    reference = copy.deepcopy(actual)
    run = load_run(parse_args(['--config', str(config), '--tier', tier,
        '--data-root', str(tmp_path), '--output', str(tmp_path / 'out')]))
    optimizer, name = build_optimizer(actual, run)
    expected = Lamb(param_groups_weight_decay(reference, weight_decay=run.train['weight_decay']),
                    lr=run.train['lr'], eps=1e-8, max_grad_norm=1., decoupled_decay=False)
    assert name == 'apex.lamb.fused'
    for _ in range(3):
        for a, b in zip(actual.parameters(), reference.parameters()):
            gradient = torch.randn_like(a)
            a.grad, b.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        expected.step()
        for a, b in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)
    assert actual.core_delta.abs().sum() > 0


@pytest.mark.cuda
@pytest.mark.parametrize("bce", [False, True])
def test_graph_accumulation_matches_eager_updates_and_rng(bce):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(814)
    eager = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Dropout(.2), torch.nn.Linear(16, 5)).cuda()
    graphed = copy.deepcopy(eager)
    criterion = imagenet.DeiTBinaryCrossEntropy() if bce else torch.nn.CrossEntropyLoss()
    engine = imagenet.ImageNetGraphStep(graphed, criterion, grad_accum=2)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True) for m in (eager, graphed)]
    try:
        for step in range(3):
            images = [torch.randn(16, 8, device='cuda') + step for _ in range(2)]
            targets = [(torch.arange(16, device='cuda') + micro + step) % 5 for micro in range(2)]
            if bce:
                targets = [.3 * torch.nn.functional.one_hot(t, 5).float()
                    + .7 * torch.nn.functional.one_hot((t + 1) % 5, 5).float() for t in targets]
            rng = torch.cuda.get_rng_state()
            eager.zero_grad(set_to_none=True)
            logits, losses = [], []
            for image, target in zip(images, targets):
                with imagenet._autocast():
                    output = eager(image)
                    loss = criterion(output, target)
                (loss / 2).backward()
                logits.append(output)
                losses.append(loss)
            expected_rng = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng)
            graphed.zero_grad(set_to_none=True)
            output, loss = engine(images, targets)
            torch.testing.assert_close(output, torch.cat(logits), rtol=0, atol=0)
            torch.testing.assert_close(loss, torch.stack(losses).mean(), rtol=0, atol=0)
            torch.testing.assert_close(torch.cuda.get_rng_state(), expected_rng, rtol=0, atol=0)
            for a, b in zip(graphed.parameters(), eager.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            for optimizer in optimizers:
                optimizer.step()
            for a, b in zip(graphed.parameters(), eager.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        with pytest.raises(ValueError, match='complete grad_accum group'):
            engine(images[:1], targets[:1])
    finally:
        engine.close()


def test_deit_bce_matches_upstream_targets_loss_and_gradients(tmp_path):
    run = load_run(_args(tmp_path))
    run.train.update(bce_loss=True, label_smoothing=0.)
    _, criterion = imagenet.build_mixup_and_loss(run)
    targets = torch.tensor([[.3, .7, 0.], [0., 0., 1.]])
    logits = torch.tensor([[.5, -1., 2.], [-.5, 1., 0.]], requires_grad=True)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.gt(0).float())
    actual = criterion(logits, targets)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0],
                               torch.autograd.grad(expected, logits)[0])
    labels = torch.tensor([0, 2])
    torch.testing.assert_close(criterion(logits, labels),
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.nn.functional.one_hot(labels, 3).float()))
    run.train["label_smoothing"] = .1
    with pytest.raises(ValueError, match="label_smoothing = 0"):
        imagenet.build_mixup_and_loss(run)


@pytest.mark.cuda
@pytest.mark.parametrize("bce", [False, True])
def test_graph_step_preserves_warmup_rng_updates_inputs_and_reuses_gradients(bce):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(916)
    eager = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(16, 5)).cuda()
    graphed = copy.deepcopy(eager)
    criterion = imagenet.DeiTBinaryCrossEntropy() if bce else torch.nn.CrossEntropyLoss()
    engine = imagenet.ImageNetGraphStep(graphed, criterion)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True) for m in (eager, graphed)]
    for step in range(3):
        images = torch.randn(16, 8, device="cuda") + step * .1
        targets = (torch.arange(16, device="cuda") + step) % 5
        if bce:
            targets = .3 * torch.nn.functional.one_hot(targets, 5).float() + .7 * torch.nn.functional.one_hot((targets + 1) % 5, 5).float()
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
