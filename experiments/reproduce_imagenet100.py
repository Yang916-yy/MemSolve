"""Reproduce the published 100-epoch ImageNet-100 Small ablations."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import concurrent.futures
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tarfile
import time

import numpy as np
from PIL import Image
import torch
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset, get_worker_info

from experiments import imagenet as im
from memsolve import MemSolve
from memsolve.ball.reference import accretive_generator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/imagenet100.toml"
PUBLISHED_METADATA = ROOT / "results/imagenet100/metadata.json"

VARIANTS: dict[str, tuple[str, int, bool]] = {
    "equilibrium_r32": ("equilibrium", 32, True),
    "complement_r32": ("complement", 32, True),
    "no_omega_r32": ("equilibrium", 32, False),
    "equilibrium_r16": ("equilibrium", 16, True),
}


class LowerTriangular(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tril(value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_all() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)


def _load_published_contract() -> dict:
    data = json.loads(PUBLISHED_METADATA.read_text(encoding="utf-8"))
    return data["variants"]["equilibrium_r32"]["recorded"]


def _scan_shard(job: tuple[str, str, int, Path, dict[int, int]]) -> tuple[str, list]:
    split, name, expected_count, data_root, keep_map = job
    rows = []
    seen = 0
    label = None
    key = None
    with tarfile.open(data_root / name, "r:") as tar:
        for member in tar:
            if member.name.endswith(".cls"):
                raw_cls = int(tar.extractfile(member).read())
                seen += 1
                key = member.name[:-4]
                label = keep_map.get(raw_cls)
            elif member.name.endswith(".jpg") and label is not None:
                assert member.name[:-4] == key
                rows.append([name, member.offset_data, member.size, label, key])
                label = None
    if seen != expected_count:
        raise ValueError(f"shard {name} count mismatch: {seen} != {expected_count}")
    return split, rows


def ensure_subset_indices(data_root: Path, cache_dir: Path) -> dict:
    """Verify or build the ImageNet-100 WebDataset byte-offset index."""
    contract = _load_published_contract()["dataset"]
    keep_map = {int(k): int(v) for k, v in contract["class_map_original_to_subset"].items()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_idx = cache_dir / "train_index.json"
    val_idx = cache_dir / "validation_index.json"

    if not (train_idx.is_file() and val_idx.is_file()):
        info_path = data_root / "_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"missing {info_path}; provide ImageNet-1K WebDataset shards in --data-root"
            )
        spec = json.loads(info_path.read_text(encoding="utf-8"))
        jobs = []
        for split in ("train", "validation"):
            s = spec["splits"][split]
            jobs.extend(
                (split, name, count, data_root, keep_map)
                for name, count in zip(s["filenames"], s["shard_lengths"])
            )
        out: dict[str, list] = {"train": [], "validation": []}
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 4)) as pool:
            for split, rows in pool.map(_scan_shard, jobs):
                out[split].extend(rows)
        for split, path in (("train", train_idx), ("validation", val_idx)):
            path.write_text(json.dumps(out[split]), encoding="utf-8")

    keys: dict[str, set[str]] = {}
    for split, path in (("train", train_idx), ("validation", val_idx)):
        expected_split = contract["splits"][split]
        actual_sha = _digest(path)
        if actual_sha != expected_split["sha256"]:
            raise ValueError(
                f"{split}_index.json sha256 mismatch: got {actual_sha}, "
                f"expected {expected_split['sha256']}"
            )
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != expected_split["samples"]:
            raise ValueError(f"{split} sample count mismatch: {len(rows)}")
        counts = Counter(row[3] for row in rows)
        if set(counts) != set(range(100)):
            raise ValueError(f"{split} does not cover all 100 classes")
        if split == "validation" and set(counts.values()) != {50}:
            raise ValueError("validation split must have 50 images per class")
        keys[split] = {row[4] for row in rows}
        if len(keys[split]) != expected_split["samples"]:
            raise ValueError(f"duplicate sample keys in {split}")
        ends: dict[str, int] = {}
        for name, offset, size, _, _ in rows:
            assert offset >= 0 and size > 0
            ends[name] = max(ends.get(name, 0), offset + size)
        for name, end in ends.items():
            shard = data_root / name
            if not shard.is_file() or shard.stat().st_size < end:
                raise FileNotFoundError(f"shard missing or truncated: {shard}")
    if keys["train"] & keys["validation"]:
        raise ValueError("train and validation keys overlap")
    return contract


class SubsetTarDataset(Dataset):
    def __init__(self, index_path: Path, data_root: Path, transform) -> None:
        self.rows = json.loads(index_path.read_text(encoding="utf-8"))
        self.data_root = data_root
        self.transform = transform
        self.handles: OrderedDict[str, io.BufferedReader] = OrderedDict()
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.seeded_epoch: int | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(epoch)

    def __getitem__(self, index: int):
        worker = get_worker_info()
        epoch = int(self.epoch)
        if worker is not None and self.seeded_epoch != epoch:
            seed = epoch * 10000 + worker.id + 123
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.seeded_epoch = epoch
        name, offset, size, label, key = self.rows[index]
        handle = self.handles.pop(name, None)
        if handle is None:
            handle = (self.data_root / name).open("rb")
        self.handles[name] = handle
        if len(self.handles) > 32:
            self.handles.popitem(last=False)[1].close()
        handle.seek(offset)
        data = handle.read(size)
        assert len(data) == size, key
        with Image.open(io.BytesIO(data)) as picture:
            image = self.transform(picture.convert("RGB"))
        return image, label


class EpochController:
    def __init__(self, dataset: SubsetTarDataset, generator: torch.Generator) -> None:
        self.dataset = dataset
        self.generator = generator

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        self.generator.manual_seed(20261010 + epoch)


def build_run_config(variant: str, data_root: Path, output: Path, config_path: Path = DEFAULT_CONFIG):
    if variant == "equilibrium_r64":
        mode, rank = "equilibrium", 64
    else:
        mode, rank, _ = VARIANTS[variant]
    args = im.parse_args([
        "--config", str(config_path),
        "--tier", "small",
        "--data-root", str(data_root),
        "--output", str(output),
    ])
    run = im.load_run(args)
    return replace(
        run,
        operator={**run.operator, "core_mode": mode},
        model={**run.model, "rank": rank},
    )


def ensure_common_init(data_root: Path, cache_dir: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    init_path = cache_dir / "common_init.pt"
    if not init_path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        master = im.build_model(build_run_config("equilibrium_r64", data_root, cache_dir, config_path))
        torch.save(master.state_dict(), init_path)
    expected_sha = _load_published_contract()["initial_weights_sha256"]
    actual_sha = _digest(init_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"common_init.pt sha256 mismatch: got {actual_sha}, expected {expected_sha}"
        )
    return init_path


def make_model(variant: str, data_root: Path, output: Path, cache_dir: Path, device: torch.device | str):
    run = build_run_config(variant, data_root, output)
    init_path = ensure_common_init(data_root, cache_dir)
    torch.manual_seed(0)
    model = im.build_model(run)
    common = torch.load(init_path, map_location="cpu", weights_only=True)
    own = model.state_dict()
    rank, heads = run.model["rank"], run.model["num_heads"]
    paired = {}
    for name, value in own.items():
        if not isinstance(value, torch.Tensor):
            paired[name] = value
        elif name.endswith(("w_bc.weight", "w_bc.bias")):
            master = common[name]
            relation = master[: heads * 64].reshape(heads, 64, *master.shape[1:])[:, :rank]
            paired[name] = torch.cat(
                (relation.reshape(heads * rank, *master.shape[1:]), master[heads * 64 :]),
                dim=0,
            )
        elif name.endswith("core_base_raw"):
            paired[name] = torch.zeros_like(value)
        else:
            assert common[name].shape == value.shape, name
            paired[name] = common[name]
    model.load_state_dict(paired, strict=True)
    if not VARIANTS[variant][2]:
        for module in model.modules():
            if isinstance(module, MemSolve):
                parametrize.register_parametrization(
                    module, "core_base_raw", LowerTriangular()
                )
    return model.to(device), run


def verify_all_variants_cpu(data_root: Path, cache_dir: Path) -> dict:
    ensure_subset_indices(data_root, cache_dir)
    ensure_common_init(data_root, cache_dir)
    published = json.loads(PUBLISHED_METADATA.read_text(encoding="utf-8"))["variants"]
    report = {}
    shared = None
    for variant, (mode, rank, skew) in VARIANTS.items():
        model, run = make_model(variant, data_root, cache_dir / variant, cache_dir, "cpu")
        params = dict(model.named_parameters())
        if shared is None:
            shared = {
                n: p.detach().clone()
                for n, p in params.items()
                if not n.endswith(("w_bc.weight", "w_bc.bias", "core_base_raw"))
            }
        for name, value in shared.items():
            if name in params:
                assert torch.equal(value, params[name]), (variant, name)
        assert run.train["epochs"] == 100
        assert not run.train["ema"] and not run.train["repeated_aug"]
        if not skew:
            module = next(m for m in model.modules() if isinstance(m, MemSolve))
            with torch.no_grad():
                module.parametrizations.core_base_raw.original.normal_(std=0.1)
            raw = module.core_base_raw
            actual = accretive_generator(raw)
            expected = accretive_generator(
                module.parametrizations.core_base_raw.original, skew_coupling=False
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.sum().backward()
            assert (
                torch.count_nonzero(
                    torch.triu(module.parametrizations.core_base_raw.original.grad, 1)
                )
                == 0
            )
        num_params = sum(p.numel() for p in params.values())
        assert num_params == published[variant]["recorded"]["parameters"], variant
        report[variant] = {
            "core_mode": mode,
            "rank": rank,
            "skew_coupling": skew,
            "parameters": num_params,
            "shared_weights_equal": True,
        }
        del model
    return report


def train_variant(variant: str, data_root: Path, output: Path, cache_dir: Path) -> None:
    dataset_meta = ensure_subset_indices(data_root, cache_dir)
    init_path = ensure_common_init(data_root, cache_dir)
    device = torch.device("cuda", 0)
    state = im.DistributedState(0, 1, 0, device)
    model, run = make_model(variant, data_root, output, cache_dir, device)
    im.prepare_operator_backend(run, device)
    output.mkdir(parents=True, exist_ok=False)
    trainset = SubsetTarDataset(
        cache_dir / "train_index.json", data_root, im.build_train_transform(run)
    )
    valset = SubsetTarDataset(
        cache_dir / "validation_index.json", data_root, im.build_eval_transform(run)
    )
    generator = torch.Generator().manual_seed(20261010)
    controller = EpochController(trainset, generator)
    kwargs = dict(
        batch_size=512,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True,
        multiprocessing_context="spawn",
    )
    loader = DataLoader(
        trainset,
        shuffle=True,
        drop_last=True,
        num_workers=run.train["train_workers"],
        generator=generator,
        **kwargs,
    )
    valloader = DataLoader(
        valset,
        shuffle=False,
        num_workers=run.train["val_workers"],
        generator=torch.Generator().manual_seed(551),
        **kwargs,
    )
    steps = len(loader)
    plan = im.BatchingPlan(1, 512, 512, 1, steps * 512, steps)
    optimizer, optimizer_name = im.build_optimizer(model, run)
    scheduler = im.build_scheduler(optimizer, run, steps)
    mixup, criterion = im.build_mixup_and_loss(run)
    _seed_all()
    graph = im.ImageNetGraphStep(model, criterion)
    mode, rank, skew = VARIANTS[variant]
    metadata = {
        "variant": variant,
        "ablation": {
            "core_mode": mode,
            "rank": rank,
            "skew_coupling": skew,
            "cpe": True,
        },
        "epochs": 100,
        "seed": 0,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "model": run.model,
        "operator": run.operator,
        "train": run.train,
        "optimizer": optimizer_name,
        "batching": plan.as_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "dataset": dataset_meta,
        "initial_weights_sha256": _digest(init_path),
        "config_sha256": _digest(DEFAULT_CONFIG),
        "script_sha256": _digest(Path(__file__)),
        "note": (
            "Fresh 100-epoch cosine schedule, 5-epoch update warmup; no EMA. "
            "Augmented training loss is not directly comparable to clean validation CE."
        ),
    }
    im._atomic_json(output / "metadata.json", metadata)
    best = -1.0
    try:
        for epoch in range(100):
            started = time.monotonic()
            lr = optimizer.param_groups[0]["lr"]
            loss, top1, top5 = im.train_epoch(
                model,
                loader,
                controller,
                criterion,
                optimizer,
                mixup,
                scheduler,
                epoch=epoch,
                state=state,
                run=run,
                batching_plan=plan,
                print_freq=50,
                graph_step=graph,
            )
            val_loss, val_top1, val_top5 = im.evaluate(model, valloader, state=state)
            row = {
                "epoch": epoch + 1,
                "train_loss": loss,
                "train_acc1": top1,
                "val_loss": val_loss,
                "val_acc1": val_top1,
                "val_acc5": val_top5,
                "val_error1": 100 - val_top1,
                "best_val_acc1": max(best, val_top1),
                "lr_start": lr,
                "lr_end": optimizer.param_groups[0]["lr"],
                "seconds": time.monotonic() - started,
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "eta_per_layer": [
                    m.complement().detach().cpu().tolist()
                    for m in model.modules()
                    if isinstance(m, MemSolve)
                ],
            }
            improved = val_top1 > best
            best = max(best, val_top1)
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc1": best,
                "metadata": metadata,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            }
            im._atomic_torch_save(payload, output / "checkpoint_last.pt")
            if improved:
                im._atomic_torch_save(payload, output / "checkpoint_best.pt")
            im._append_jsonl(output / "metrics.jsonl", row)
            print(json.dumps({"event": "epoch", **row}), flush=True)
        im._atomic_json(output / "complete.json", {"epochs": 100, "best_val_acc1": best})
    finally:
        graph.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=list(VARIANTS))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for shared subset indices and common_init.pt (defaults to --output parent)",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    cache_dir = (args.cache_dir or output.parent).resolve()
    if args.prepare_only:
        report = verify_all_variants_cpu(data_root, cache_dir)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return
    train_variant(args.variant, data_root, output, cache_dir)


if __name__ == "__main__":
    main()
