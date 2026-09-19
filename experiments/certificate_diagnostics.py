"""Measure LSSO's realized certificates across depth and sequence length."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

from lsso import LSSO
from lsso.ball import cuda as cuda_backend

from experiments.sequence_data import prepare_lra
from experiments.train_transformers import SequenceClassifier, build_model


DEFAULT_LENGTHS = (1024, 2048, 4096, 8192)
METRICS = (
    "q",
    "contraction_slack",
    "mu",
    "state_ratio",
    "adjoint_ratio",
    "state_bound_usage",
    "adjoint_bound_usage",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="Formal Text best.pt; repeat for additional seeds.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--probe-seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint.with_name("config.json")
    if not config_path.is_file():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_text_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[SequenceClassifier, Any, dict[str, Any]]:
    config = _checkpoint_config(checkpoint)
    resolved = config["resolved_arguments"]
    model_config = config["model"]
    if resolved["suite"] != "lra" or resolved["task"] != "text":
        raise ValueError(f"expected an LRA Text checkpoint, got {checkpoint}")
    if model_config["mixer"] != "lsso" or model_config["core_mode"] != "dynamic":
        raise ValueError(f"expected a dynamic LSSO checkpoint, got {checkpoint}")

    bundle = prepare_lra(
        "text",
        data_root=Path(resolved["data_root"]),
        cache_root=Path(resolved["cache_root"]),
        max_length=int(resolved["max_length"]),
        validation_fraction=float(resolved["validation_fraction"]),
        split_seed=int(resolved["split_seed"]),
        pathfinder_resolution=None,
        allow_download=False,
        revision=None,
        formal=True,
    )
    build_args = SimpleNamespace(
        suite="lra",
        task="text",
        implementation="cuda",
        dim=int(model_config["dim"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        rank=int(model_config["rank"]),
        mixer="lsso",
        core_mode="dynamic",
        mlp_ratio=float(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
        bias=bool(model_config["bias"]),
        pooling=str(model_config["pooling"]),
    )
    model = build_model(build_args, bundle)
    if not isinstance(model, SequenceClassifier):
        raise TypeError("LRA Text must construct a SequenceClassifier")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), bundle, config


def _validation_token_stream(dataset: Any, required: int) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    total = 0
    for index in range(len(dataset)):
        tokens, _label = dataset[index]
        pieces.append(tokens.to(dtype=torch.long, device="cpu"))
        total += int(tokens.numel())
        if total >= required:
            break
    if total < required:
        raise RuntimeError(
            f"validation split has only {total} tokens, need {required}"
        )
    return torch.cat(pieces)[:required]


def _probe_tokens(
    stream: torch.Tensor,
    *,
    probes: int,
    max_length: int,
) -> torch.Tensor:
    required = probes * max_length
    if stream.numel() < required:
        raise ValueError(f"token stream has {stream.numel()} entries, need {required}")
    return stream[:required].view(probes, max_length)


def _adjoint_probe(
    *,
    batch_indices: torch.Tensor,
    layer: int,
    heads: int,
    rank: int,
    head_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for probe_index in batch_indices.tolist():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1_000_003 * layer + 97_409 * probe_index)
        value = torch.randint(
            0,
            2,
            (heads, rank, head_dim),
            generator=generator,
            dtype=torch.int64,
        ).to(dtype=torch.float32)
        value = 2.0 * value - 1.0
        value = value / torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True)
        values.append(value)
    return torch.stack(values).to(device=device)


def _initial_features(
    model: SequenceClassifier,
    tokens: torch.Tensor,
) -> torch.Tensor:
    encoder = model.encoder
    assert isinstance(encoder.input_embedding, torch.nn.Embedding)
    assert encoder.position_embedding is not None
    training_length = encoder.position_embedding.num_embeddings
    positions = torch.arange(tokens.shape[1], device=tokens.device) % training_length
    return (
        encoder.input_embedding(tokens)
        + encoder.position_embedding(positions)[None]
    ).to(dtype=torch.float16)


def _record_batch(
    model: SequenceClassifier,
    tokens: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    length: int,
    checkpoint_index: int,
    checkpoint_seed: int,
    probe_seed: int,
) -> list[dict[str, float | int]]:
    encoder = model.encoder
    mask = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
    x = _initial_features(model, tokens)
    records: list[dict[str, float | int]] = []
    for layer_index, block in enumerate(encoder.blocks):
        if not isinstance(block.mixer, LSSO):
            raise TypeError("certificate diagnostics require LSSO blocks")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            normalized = block.norm1(x)
        config = block.mixer.config
        rhs = _adjoint_probe(
            batch_indices=batch_indices,
            layer=layer_index,
            heads=config.num_heads,
            rank=config.rank,
            head_dim=config.head_dim,
            seed=probe_seed,
            device=x.device,
        )
        diagnostics = block.mixer.diagnostics(
            normalized,
            valid_mask=mask,
            adjoint_rhs=rhs,
        )
        for batch_offset, probe_index in enumerate(batch_indices.tolist()):
            for head in range(config.num_heads):
                record: dict[str, float | int] = {
                    "checkpoint": checkpoint_index,
                    "seed": checkpoint_seed,
                    "probe": probe_index,
                    "length": length,
                    "layer": layer_index + 1,
                    "head": head,
                }
                for name in METRICS:
                    record[name] = float(diagnostics[name][batch_offset, head].cpu())
                records.append(record)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            x = block(x, mask)
    return records


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def summarize(records: list[dict[str, float | int]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[int, int], list[dict[str, float | int]]] = {}
    for record in records:
        groups.setdefault((int(record["length"]), int(record["layer"])), []).append(record)
    rows: list[dict[str, float | int | str]] = []
    for (length, layer), group in sorted(groups.items()):
        for metric in METRICS:
            row: dict[str, float | int | str] = {
                "length": length,
                "layer": layer,
                "metric": metric,
                "count": len(group),
            }
            row.update(_percentiles(float(item[metric]) for item in group))
            rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    plotted = (
        ("q", r"Realized gain $q(X)$"),
        ("mu", r"Monotonicity margin $\mu$"),
        ("state_ratio", r"$\|U^\star\|_F/\|Z\|_F$"),
        ("adjoint_ratio", r"$\|A_K^{-T}G\|_F/\|G\|_F$"),
    )
    lengths = sorted({int(row["length"]) for row in summary})
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(lengths)))
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, plotted):
        for length, color in zip(lengths, colors):
            rows = [
                row
                for row in summary
                if row["metric"] == metric and int(row["length"]) == length
            ]
            layers = np.asarray([int(row["layer"]) for row in rows])
            medians = np.asarray([float(row["median"]) for row in rows])
            low = np.asarray([float(row["p10"]) for row in rows])
            high = np.asarray([float(row["p90"]) for row in rows])
            axis.plot(layers, medians, marker="o", linewidth=1.6, color=color, label=f"{length // 1024}K")
            axis.fill_between(layers, low, high, color=color, alpha=0.14)
        axis.set_xlabel("Layer")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted({int(row["layer"]) for row in summary}))
        axis.grid(alpha=0.22, linewidth=0.6)
        if metric == "q":
            axis.text(
                0.98,
                0.04,
                "all length curves overlap",
                ha="right",
                va="bottom",
                transform=axis.transAxes,
                fontsize=8,
            )
    axes[0, 0].legend(title="Length", ncol=2, frameon=False)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.probes <= 0 or args.batch_size <= 0:
        raise ValueError("probes and batch-size must be positive")
    if any(length <= 0 for length in args.lengths):
        raise ValueError("lengths must be positive")
    lengths = tuple(sorted(set(args.lengths)))
    max_length = max(lengths)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("certificate stress diagnostics require CUDA")
    cuda_backend.load(device=device)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int]] = []
    metadata: dict[str, Any] = {
        "schema": 1,
        "protocol": "validation-text-concatenated-prefix-operator-stress-v1",
        "scope": "operator-level stress test; not an 8K task-accuracy evaluation",
        "lengths": lengths,
        "probes": args.probes,
        "batch_size": args.batch_size,
        "probe_seed": args.probe_seed,
        "adjoint_rhs": "unit-Frobenius Rademacher; shared by length for each probe/layer",
        "position_embedding": "training-range learned absolute table repeated modulo 4096",
        "checkpoint_paths": [str(path.resolve()) for path in args.checkpoint],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
    }

    shared_probes: torch.Tensor | None = None
    for checkpoint_index, checkpoint in enumerate(args.checkpoint):
        model, bundle, config = _load_text_model(checkpoint.resolve(), device)
        checkpoint_seed = int(config["resolved_arguments"]["seed"])
        if shared_probes is None:
            stream = _validation_token_stream(bundle.validation, args.probes * max_length)
            shared_probes = _probe_tokens(
                stream,
                probes=args.probes,
                max_length=max_length,
            )
            metadata["dataset"] = config["dataset"]
        for length in lengths:
            for start in range(0, args.probes, args.batch_size):
                stop = min(start + args.batch_size, args.probes)
                indices = torch.arange(start, stop, dtype=torch.int64)
                tokens = shared_probes[start:stop, :length].to(device)
                records.extend(
                    _record_batch(
                        model,
                        tokens,
                        indices,
                        length=length,
                        checkpoint_index=checkpoint_index,
                        checkpoint_seed=checkpoint_seed,
                        probe_seed=args.probe_seed,
                    )
                )
            print(
                json.dumps(
                    {
                        "checkpoint": checkpoint_index,
                        "seed": checkpoint_seed,
                        "length": length,
                        "records": len(records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    all_finite = all(
        math.isfinite(float(record[metric]))
        for record in records
        for metric in METRICS
    )
    if not all_finite:
        raise RuntimeError("certificate diagnostics produced a non-finite value")

    q_max = max(float(record["q"]) for record in records)
    state_usage_max = max(float(record["state_bound_usage"]) for record in records)
    adjoint_usage_max = max(float(record["adjoint_bound_usage"]) for record in records)
    tolerance = 5e-10
    if q_max >= 1.0 + tolerance:
        raise RuntimeError(f"observed q above one: {q_max}")
    if state_usage_max > 1.0 + tolerance:
        raise RuntimeError(f"state certificate violation: {state_usage_max}")
    if adjoint_usage_max > 1.0 + tolerance:
        raise RuntimeError(f"adjoint certificate violation: {adjoint_usage_max}")

    summary = summarize(records)
    with (output / "observations.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    _write_csv(summary, output / "summary.csv")
    metadata["observations"] = len(records)
    metadata["checks"] = {
        "q_max": q_max,
        "state_bound_usage_max": state_usage_max,
        "adjoint_bound_usage_max": adjoint_usage_max,
        "all_finite": all_finite,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(summary, output / "certificate_length_depth.png")
    print(json.dumps(metadata["checks"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["summarize"]
