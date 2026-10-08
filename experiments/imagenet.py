"""Distributed ImageNet-1K training with RoPE-ViT, ViT³ and DeiT III recipes.

The model itself intentionally lives outside this entrypoint.  The runner calls
``integrations.timm.create_memsolve_vit`` so classification, detection, and
segmentation share one backbone implementation rather than recreating model
math in every experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import subprocess
import tarfile
import tempfile
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 is supported by the package.
    import tomli as tomllib
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info


ROOT = Path(__file__).resolve().parents[1]
VIT3_CONFIG = ROOT / "experiments" / "configs" / "imagenet_vit3.toml"
ROPEVIT_CONFIG = ROOT / "experiments" / "configs" / "imagenet_ropevit_400.toml"
DEFAULT_CONFIG = ROPEVIT_CONFIG
DEIT3_CONFIG = ROOT / "experiments" / "configs" / "imagenet_deit3_400.toml"
DEIT3_CONFIGS = {
    "deit3_400": DEIT3_CONFIG,
    "deit3_800": ROOT / "experiments" / "configs" / "imagenet_deit3_800.toml",
}
RECIPE_CONFIGS = {"vit3": VIT3_CONFIG, "ropevit_400": ROPEVIT_CONFIG, **DEIT3_CONFIGS}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OFFICIAL_VIT3_URL = (
    "https://github.com/LeapLabTHU/ViTTT/tree/"
    "e3477587d099e6b9e83e9e7c80b1b999e0989a20/vittt"
)
OFFICIAL_DEIT3_URL = "https://github.com/facebookresearch/deit/blob/main/README_revenge.md"
OFFICIAL_ROPEVIT_URL = "https://github.com/naver-ai/rope-vit/blob/main/deit/README.md"
IMAGENET_CHECKPOINT_FORMAT = 13
IMAGENET_WDS_SOURCE = "timm/imagenet-1k-wds"
IMAGENET_WDS_MANIFEST_SHA256 = (
    "092ba12f49692720b20ebcf241e416ceab77d6f079b382d5c1bb3e1227e9834f"
)
IMAGENET_TRAIN_SAMPLES = 1_281_167
IMAGENET_VALIDATION_SAMPLES = 50_000
IMAGENET_TRAIN_SHARDS = 1_024
IMAGENET_VALIDATION_SHARDS = 64
WDS_SAMPLE_SHUFFLE_SIZE = 8_192
WDS_SAMPLE_SHUFFLE_INITIAL = 2_048


def checkpoint_contract_digest(contract: Mapping[str, Any]) -> str:
    """Return the canonical digest stored with every ImageNet checkpoint."""

    try:
        encoded = json.dumps(
            contract,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("ImageNet checkpoint contract is not JSON-serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint_contract(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current checkpoint envelope before any tensor is loaded."""

    if checkpoint.get("format_version") != IMAGENET_CHECKPOINT_FORMAT:
        raise ValueError("checkpoint does not use the current ImageNet contract format")
    contract = checkpoint.get("contract")
    if not isinstance(contract, dict) or not all(
        key in contract
        for key in ("tier", "phase", "model", "operator", "train", "batching", "data")
    ):
        raise ValueError("checkpoint is missing its complete ImageNet contract")
    _validate_batching_contract(contract["batching"])
    _validate_webdataset_contract(contract["data"])
    group_size = contract["batching"].get("augmentation_group_size")
    if group_size != contract["train"].get("augmentation_group_size") or group_size != contract["data"]["streaming"].get("augmentation_group_size"):
        raise ValueError("checkpoint virtual augmentation group is inconsistent across train, batching and data")
    digest = checkpoint.get("contract_digest")
    expected = checkpoint_contract_digest(contract)
    if not isinstance(digest, str) or digest != expected:
        raise ValueError("checkpoint ImageNet contract digest does not match its content")
    return contract


@dataclass(frozen=True)
class DistributedState:
    """Process-local state established by torchrun."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class BatchingPlan:
    """Resolved physical, optional virtual-group and optimizer-update batches."""

    world_size: int
    physical_batch_size: int
    effective_batch_size: int
    grad_accum: int
    samples_per_epoch: int
    updates_per_epoch: int
    augmentation_group_size: int | None = None

    @property
    def samples_per_rank(self) -> int:
        return self.samples_per_epoch // self.world_size

    @property
    def microbatches_per_epoch(self) -> int:
        return self.updates_per_epoch * self.grad_accum

    def as_dict(self) -> dict[str, int]:
        return {
            "world_size": self.world_size,
            "physical_batch_size": self.physical_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "grad_accum": self.grad_accum,
            "samples_per_epoch": self.samples_per_epoch,
            "updates_per_epoch": self.updates_per_epoch,
            **({"augmentation_group_size": self.augmentation_group_size}
               if self.augmentation_group_size is not None else {}),
        }


@dataclass(frozen=True)
class LoaderRandomGenerators:
    """Independent generators for replayable train and validation workers."""

    train: torch.Generator
    validation: torch.Generator


@dataclass(frozen=True)
class ImageNetRun:
    """Fully resolved run contract, including the selected official recipe."""

    config_path: Path
    tier: str
    phase: str
    model: dict[str, Any]
    operator: dict[str, Any]
    train: dict[str, Any]
    overrides: tuple[str, ...]

    def checkpoint_contract(
        self,
        batching_plan: BatchingPlan,
        data_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        contract: dict[str, Any] = {
            "tier": self.tier,
            "phase": self.phase,
            "model": self.model,
            "operator": self.operator,
            "train": self.train,
            "data": dict(data_contract),
        }
        contract["batching"] = batching_plan.as_dict()
        return contract

    def checkpoint_contract_digest(
        self,
        batching_plan: BatchingPlan,
        data_contract: Mapping[str, Any],
    ) -> str:
        return checkpoint_contract_digest(
            self.checkpoint_contract(batching_plan, data_contract)
        )

    def as_dict(
        self,
        batching_plan: BatchingPlan,
        data_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "tier": self.tier,
            "phase": self.phase,
            "model": self.model,
            "operator": self.operator,
            "train": self.train,
            "overrides": list(self.overrides),
            **({"official_ropevit_recipe": OFFICIAL_ROPEVIT_URL}
               if self.train.get("recipe") == "ropevit_400"
               else {"official_deit3_recipe": OFFICIAL_DEIT3_URL}
               if self.train.get("recipe") in DEIT3_CONFIGS
               else {"official_vit3_recipe": OFFICIAL_VIT3_URL}),
            "batching": batching_plan.as_dict(),
            "data": dict(data_contract),
            "checkpoint_contract_digest": self.checkpoint_contract_digest(
                batching_plan,
                data_contract,
            ),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MemSolve with RoPE-ViT, ViT³ or DeiT III recipes on ImageNet-1K with torchrun."
    )
    parser.add_argument("--config", type=Path,
                        default=DEFAULT_CONFIG,
                        help="defaults to RoPE-ViT-derived 400e training for T/S/B")
    parser.add_argument("--tier", choices=("tiny", "small", "base"), required=True)
    parser.add_argument(
        "--phase",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "ModelScope timm/imagenet-1k-wds root containing _info.json and "
            "the downloaded tar shards."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=("cuda", "reference"),
        help="Override the configured MemSolve implementation for a diagnostic run.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an epoch-boundary checkpoint with its captured RNG state.",
    )
    parser.add_argument(
        "--finetune", type=Path,
        help="Initialize the 224px finetuning phase from a 192px checkpoint; reset optimizer, scheduler and RNG.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the official duration for a diagnostic run; recorded as non-canonical.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the physical per-GPU batch size.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        help="Require this many physical batches per optimizer update.",
    )
    parser.add_argument(
        "--train-workers",
        type=int,
        help="Override streaming WebDataset workers per rank.",
    )
    parser.add_argument(
        "--val-workers",
        type=int,
        help="Override validation WebDataset workers per rank.",
    )
    parser.add_argument("--seed", type=int, help="Override the official seed.")
    parser.add_argument("--save-every", type=int, help="Checkpoint interval in epochs.")
    parser.add_argument("--execution", choices=("eager", "graph", "compile-graph"),
                        help="Training execution; Graph captures a complete accumulated optimizer update.")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval", action="store_true", help="Evaluate --resume without training.")
    return parser.parse_args(argv)


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _require_keys(values: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing)}")


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = _positive_float(value, name) if value != 0 else 0.0
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def load_run(args: argparse.Namespace) -> ImageNetRun:
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    defaults = dict(_as_mapping(raw.get("defaults"), "[defaults]"))
    operator = dict(_as_mapping(raw.get("operator"), "[operator]"))
    operator.setdefault("qk_conv_kernel_size", 3)
    operator.setdefault("output_gate_rank", 32)
    tiers = _as_mapping(raw.get("tiers"), "[tiers]")
    tier_values = dict(_as_mapping(tiers.get(args.tier), f"[tiers.{args.tier}]"))
    phases = {
        key: value for key, value in tier_values.items() if isinstance(value, dict)
    }
    phase_values = phases.get(args.phase)
    if phase_values is None:
        available = sorted(phases)
        raise ValueError(
            f"tier {args.tier!r} does not define phase {args.phase!r}; "
            f"available phases: {', '.join(available) or 'none'}"
        )
    phase_values = dict(_as_mapping(phase_values, f"[tiers.{args.tier}.{args.phase}]"))
    tier_values = {
        key: value for key, value in tier_values.items() if not isinstance(value, dict)
    }
    # Resolution fine-tuning may change regularization without changing weights.
    drop_path_rate = phase_values.pop("drop_path_rate", tier_values["drop_path_rate"])

    model = {
        "image_size": phase_values["input_size"],
        "patch_size": defaults["patch_size"],
        "num_classes": defaults["num_classes"],
        "mlp_ratio": defaults["mlp_ratio"],
        "norm_eps": defaults["norm_eps"],
        "architecture": "vit3_qkconv_lpe2d_mean_swiglu_v5", "ffn": "swiglu",
        "layer_scale": False,
        "class_token": False,
        "pooling": "token_ln_mean",
        "drop_path_schedule": defaults.get("drop_path_schedule", "linear"),
        "position_encoding": defaults["position_encoding"],
        **tier_values,
        "drop_path_rate": drop_path_rate,
    }
    train = {**defaults, **phase_values}
    train.pop("input_size")
    overrides: list[str] = []
    for argument, key in (
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.train_workers, "train_workers"),
        (args.val_workers, "val_workers"),
        (args.seed, "seed"),
        (args.save_every, "save_every"),
        (args.execution, "execution"),
    ):
        if argument is not None:
            train[key] = argument
            overrides.append(key)
    if args.grad_accum is not None:
        _positive_int(args.grad_accum, "--grad-accum")
        overrides.append("grad_accum")
    if args.implementation is not None:
        operator["implementation"] = args.implementation
        overrides.append("implementation")

    _validate_run(args.tier, args.phase, model, operator, train)
    if args.eval and args.resume is None:
        raise ValueError("--eval requires --resume")
    if args.resume is not None and args.finetune is not None:
        raise ValueError("--resume and --finetune are mutually exclusive")
    if args.finetune is not None and args.phase != "finetune":
        raise ValueError("--finetune requires --phase finetune")
    if args.phase == "finetune" and args.resume is None and args.finetune is None:
        raise ValueError("finetuning requires --finetune or --resume")

    return ImageNetRun(
        config_path=config_path,
        tier=args.tier,
        phase=args.phase,
        model=model,
        operator=operator,
        train=train,
        overrides=tuple(overrides),
    )


def _validate_run(
    tier: str,
    phase: str,
    model: Mapping[str, Any],
    operator: Mapping[str, Any],
    train: Mapping[str, Any],
) -> None:
    recipe = train.get("recipe", "vit3")
    if recipe not in RECIPE_CONFIGS:
        raise ValueError(f"recipe must be one of {', '.join(RECIPE_CONFIGS)}")
    _require_keys(
        model,
        (
            "image_size",
            "patch_size",
            "num_classes",
            "embed_dim",
            "depth",
            "num_heads",
            "rank",
            "mlp_ratio",
            "norm_eps",
            "drop_path_rate",
        ),
        f"model contract for {tier}",
    )
    _require_keys(
        operator,
        ("bias", "implementation"),
        "[operator]",
    )
    _require_keys(
        train,
        (
            "epochs",
            "batch_size",
            "effective_batch",
            "lr",
            "min_lr",
            "warmup_lr",
            "warmup_epochs",
            "weight_decay",
            "train_workers",
            "val_workers",
            "eval_crop_ratio",
            "mixup",
            "cutmix",
            "mixup_prob",
            "mixup_switch_prob",
            "mixup_mode",
            "ema",
            "clip_grad",
            "amp_dtype",
            "save_every",
            "optimizer",
            "augmentation",
            "color_jitter",
            "auto_augment",
            "repeated_aug",
            "bce_loss",
            "label_smoothing",
        ),
        f"training contract for {tier}/{phase}",
    )

    image_size = _positive_int(model["image_size"], "image_size")
    patch_size = _positive_int(model["patch_size"], "patch_size")
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")
    embed_dim = _positive_int(model["embed_dim"], "embed_dim")
    heads = _positive_int(model["num_heads"], "num_heads")
    if embed_dim % heads:
        raise ValueError("embed_dim must be divisible by num_heads")
    _positive_int(model["depth"], "depth")
    _positive_int(model["rank"], "rank")
    _positive_int(model["num_classes"], "num_classes")
    _positive_float(model["mlp_ratio"], "mlp_ratio")
    _positive_float(model["norm_eps"], "norm_eps")
    _probability(model["drop_path_rate"], "drop_path_rate")

    for key, expected in {
        "architecture": "vit3_qkconv_lpe2d_mean_swiglu_v5", "ffn": "swiglu", "layer_scale": False, "class_token": False,
        "pooling": "token_ln_mean", "position_encoding": "learned_2d",
    }.items():
        if model.get(key) != expected:
            raise ValueError(f"the vision scaffold requires {key}={expected!r}")
    if "layer_scale_init_value" in model or "layer_scale_init_value" in train:
        raise ValueError("LayerScale is not part of the current vision scaffold")

    if "core_mode" in operator:
        raise ValueError("core_mode is not part of the QKV equilibrium contract")
    from memsolve import MemSolveConfig
    MemSolveConfig(
        dim=embed_dim, num_heads=heads, rank=model["rank"], qk_conv_dim=2,
        qk_conv_kernel_size=operator["qk_conv_kernel_size"],
        output_gate_rank=operator["output_gate_rank"],
    )
    if operator["bias"] is not True:
        raise ValueError("the ViT³ scaffold requires qkv/projection bias")
    if operator["implementation"] not in {"cuda", "reference"}:
        raise ValueError("operator.implementation must be 'cuda' or 'reference'")

    if train.get("execution", "eager") not in {"eager", "graph", "compile-graph"}:
        raise ValueError("unsupported training execution")
    _positive_int(train["epochs"], "epochs")
    _positive_int(train["batch_size"], "batch_size")
    _positive_int(train["effective_batch"], "effective_batch")
    _nonnegative_int(train["train_workers"], "train_workers")
    _nonnegative_int(train["val_workers"], "val_workers")
    _nonnegative_int(train["warmup_epochs"], "warmup_epochs")
    _positive_int(train["save_every"], "save_every")
    _positive_float(train["lr"], "lr")
    _positive_float(train["warmup_lr"], "warmup_lr")
    _positive_float(train["min_lr"], "min_lr")
    _positive_float(train["weight_decay"], "weight_decay")
    _probability(train["eval_crop_ratio"], "eval_crop_ratio")
    _probability(train["mixup"], "mixup")
    _probability(train["cutmix"], "cutmix")
    _probability(train["mixup_prob"], "mixup_prob")
    _probability(train["mixup_switch_prob"], "mixup_switch_prob")
    _probability(train["label_smoothing"], "label_smoothing")
    if train["clip_grad"] != 0:
        _positive_float(train["clip_grad"], "clip_grad")
    if train["mixup_mode"] != "batch":
        raise ValueError("the ImageNet recipes use batch-mode Mixup/CutMix")
    if train["amp_dtype"] != "bfloat16":
        raise ValueError("the current ImageNet contract requires train.amp_dtype = 'bfloat16'")
    if not isinstance(train["ema"], bool):
        raise ValueError("ema must be a boolean")
    if train["ema"]:
        decay = _probability(train.get("ema_decay"), "ema_decay")
        if decay >= 1:
            raise ValueError("ema_decay must be less than one")
    if recipe == "vit3":
        if train["optimizer"] != "fused_adamw":
            raise ValueError("plain ViT³ uses fused AdamW")
        if train["augmentation"] != "rand_augment":
            raise ValueError("plain ViT³ uses RandAugment")
        if model["drop_path_schedule"] != "linear":
            raise ValueError("plain ViT³ requires linear DropPath")
        for key in ("repeated_aug", "ema"):
            if train[key] is not False:
                raise ValueError(f"plain ViT³ requires {key} = false")
    else:
        family = "RoPE-ViT" if recipe == "ropevit_400" else "DeiT III"
        group_size = _positive_int(train.get("augmentation_group_size"), "augmentation_group_size")
        if group_size % 2 or int(train["batch_size"]) % group_size:
            raise ValueError(f"{family} physical batch_size must be divisible by an even augmentation_group_size")
        if model["drop_path_schedule"] != "constant":
            raise ValueError(f"{family} requires constant DropPath across layers")
        expected_optimizer = "fused_lamb" if phase == "pretrain" else "fused_adamw"
        expected_augmentation = "three_augment" if phase == "pretrain" else "rand_augment"
        if train["optimizer"] != expected_optimizer or train["augmentation"] != expected_augmentation:
            raise ValueError(f"{family} {phase} requires {expected_optimizer} and {expected_augmentation}")
        if train["repeated_aug"] is not (phase == "pretrain"):
            raise ValueError(f"{family} repeats augmentation only in pretraining")
        if recipe in DEIT3_CONFIGS and train["ema"] is not False:
            raise ValueError("the explicit DeiT III adaptations disable EMA")
    if not isinstance(train["bce_loss"], bool):
        raise ValueError("bce_loss must be a boolean")
    if train["bce_loss"] and float(train["label_smoothing"]) != 0:
        raise ValueError("DeiT III multi-label BCE requires label_smoothing = 0")

    expected = {
        "small": (384, 12, 6, 32),
        "base": (768, 12, 12, 48),
        "tiny": (192, 12, 6, 16),
    }[tier]
    actual = (model["embed_dim"], model["depth"], model["num_heads"], model["rank"])
    if actual != expected:
        raise ValueError(f"{tier} geometry must be {expected}, got {actual}")
    low_resolution_pretrain = phase == "pretrain" and (
        recipe in DEIT3_CONFIGS or (recipe == "ropevit_400" and tier == "base")
    )
    expected_size = 192 if low_resolution_pretrain else 224
    if image_size != expected_size or (recipe == "vit3" and phase != "pretrain"):
        raise ValueError(f"{recipe}/{phase} requires input size {expected_size}")


def _validate_batching_contract(value: object) -> None:
    batching = _as_mapping(value, "checkpoint batching contract")
    _require_keys(
        batching,
        (
            "world_size",
            "physical_batch_size",
            "effective_batch_size",
            "grad_accum",
            "samples_per_epoch",
            "updates_per_epoch",
        ),
        "checkpoint batching contract",
    )
    world_size = _positive_int(batching["world_size"], "batching.world_size")
    physical_batch_size = _positive_int(
        batching["physical_batch_size"],
        "batching.physical_batch_size",
    )
    effective_batch_size = _positive_int(
        batching["effective_batch_size"],
        "batching.effective_batch_size",
    )
    grad_accum = _positive_int(batching["grad_accum"], "batching.grad_accum")
    samples_per_epoch = _positive_int(
        batching["samples_per_epoch"],
        "batching.samples_per_epoch",
    )
    updates_per_epoch = _positive_int(
        batching["updates_per_epoch"],
        "batching.updates_per_epoch",
    )
    if effective_batch_size != world_size * physical_batch_size * grad_accum:
        raise ValueError(
            "checkpoint effective_batch_size does not match world_size, "
            "physical_batch_size, and grad_accum"
        )
    if samples_per_epoch % effective_batch_size:
        raise ValueError("checkpoint samples_per_epoch is not a whole effective batch")
    if updates_per_epoch != samples_per_epoch // effective_batch_size:
        raise ValueError("checkpoint updates_per_epoch does not match samples_per_epoch")
    if "augmentation_group_size" in batching:
        group = _positive_int(batching["augmentation_group_size"], "batching.augmentation_group_size")
        if group % 2 or physical_batch_size % group:
            raise ValueError("checkpoint physical batch is not divisible by its even augmentation group")


def resolve_batching_plan(
    run: ImageNetRun,
    state: DistributedState,
    *,
    dataset_size: int,
    requested_grad_accum: int | None,
) -> BatchingPlan:
    """Resolve one exact optimizer-update schedule for the current launcher."""

    if dataset_size < 1:
        raise ValueError("ImageNet train dataset must contain at least one sample")
    physical_batch_size = int(run.train["batch_size"])
    effective_batch_size = int(run.train["effective_batch"])

    global_physical_batch = state.world_size * physical_batch_size
    if requested_grad_accum is None:
        if effective_batch_size % global_physical_batch:
            raise ValueError(
                "effective_batch must be divisible by world_size * physical batch_size"
            )
        grad_accum = effective_batch_size // global_physical_batch
    else:
        grad_accum = _positive_int(requested_grad_accum, "--grad-accum")
    if global_physical_batch * grad_accum != effective_batch_size:
        raise ValueError(
            "world_size * physical batch_size * grad_accum must equal effective_batch"
        )

    updates_per_epoch = dataset_size // effective_batch_size
    if updates_per_epoch < 1:
        raise ValueError(
            "ImageNet train dataset is smaller than one configured effective batch"
        )
    plan = BatchingPlan(
        world_size=state.world_size,
        physical_batch_size=physical_batch_size,
        effective_batch_size=effective_batch_size,
        grad_accum=grad_accum,
        samples_per_epoch=updates_per_epoch * effective_batch_size,
        updates_per_epoch=updates_per_epoch,
        augmentation_group_size=run.train.get("augmentation_group_size"),
    )
    _validate_batching_contract(plan.as_dict())
    return plan


def initialize_distributed() -> DistributedState:
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    present = [key for key in keys if key in os.environ]
    if present and len(present) != len(keys):
        missing = sorted(set(keys) - set(present))
        raise RuntimeError(
            "incomplete torchrun environment; missing " + ", ".join(missing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet training requires a CUDA device")

    if not present:
        return DistributedState(
            rank=0,
            world_size=1,
            local_rank=torch.cuda.current_device(),
            device=torch.device("cuda", torch.cuda.current_device()),
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 1 or not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError("torchrun rank/world-size/device configuration is invalid")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
    )


def finalize_distributed(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(state: DistributedState) -> None:
    if state.enabled:
        dist.barrier()


def seed_everything(seed: int, state: DistributedState) -> None:
    process_seed = int(seed) + state.rank
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    try:
        import numpy as np

        np.random.seed(process_seed)
    except ImportError:
        pass


def _seed_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _local_resume_rng_state(
    state: DistributedState,
    generators: LoaderRandomGenerators,
) -> dict[str, Any]:
    """Capture one rank's replayable epoch-boundary random state."""

    try:
        import numpy as np

        numpy_state: Any | None = np.random.get_state()
    except ImportError:
        numpy_state = None
    cuda_state = (
        torch.cuda.get_rng_state(state.device)
        if state.device.type == "cuda"
        else None
    )
    return {
        "rank": state.rank,
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
        "train_generator": generators.train.get_state(),
        "validation_generator": generators.validation.get_state(),
    }


def _capture_resume_rng_state(
    state: DistributedState,
    generators: LoaderRandomGenerators,
) -> dict[str, Any]:
    """Collect all rank-local random states for an exact epoch-boundary resume."""

    local_state = _local_resume_rng_state(state, generators)
    states: list[object]
    if state.enabled:
        states = [None] * state.world_size
        dist.all_gather_object(states, local_state)
    else:
        states = [local_state]
    return {
        "world_size": state.world_size,
        "device_type": state.device.type,
        "states": states,
    }


def _resume_rng_state_for_rank(
    saved: Any,
    *,
    state: DistributedState,
) -> Mapping[str, Any]:
    """Validate and select the saved random state for the current rank."""

    if not isinstance(saved, dict):
        raise ValueError("resume checkpoint is missing its epoch-boundary RNG state")
    if saved.get("world_size") != state.world_size:
        raise ValueError("resume checkpoint RNG world size does not match this run")
    if saved.get("device_type") != state.device.type:
        raise ValueError("resume checkpoint RNG device type does not match this run")
    states = saved.get("states")
    if not isinstance(states, list) or len(states) != state.world_size:
        raise ValueError("resume checkpoint has an invalid per-rank RNG state")
    rank_state = states[state.rank]
    if not isinstance(rank_state, dict) or rank_state.get("rank") != state.rank:
        raise ValueError("resume checkpoint RNG state does not match this rank")
    required = (
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "train_generator",
        "validation_generator",
    )
    if any(key not in rank_state for key in required):
        raise ValueError("resume checkpoint has an incomplete RNG state")
    if not all(
        isinstance(rank_state[key], torch.Tensor)
        for key in ("torch_cpu", "train_generator", "validation_generator")
    ):
        raise ValueError("resume checkpoint has an invalid CPU or loader RNG state")
    cuda_state = rank_state["torch_cuda"]
    if state.device.type == "cuda":
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError("resume checkpoint has an invalid CUDA RNG state")
    elif cuda_state is not None:
        raise ValueError("resume checkpoint unexpectedly contains CUDA RNG state")
    return rank_state


def _restore_resume_rng_state(
    saved: Any,
    *,
    state: DistributedState,
    generators: LoaderRandomGenerators,
) -> None:
    """Restore one rank's saved random streams after model state is loaded."""

    rank_state = _resume_rng_state_for_rank(saved, state=state)
    try:
        random.setstate(rank_state["python"])
        numpy_state = rank_state["numpy"]
        if numpy_state is not None:
            try:
                import numpy as np
            except ImportError as error:
                raise ValueError(
                    "resume checkpoint requires NumPy to restore its RNG state"
                ) from error
            np.random.set_state(numpy_state)
        torch.set_rng_state(rank_state["torch_cpu"])
        if state.device.type == "cuda":
            torch.cuda.set_rng_state(rank_state["torch_cuda"], state.device)
        generators.train.set_state(rank_state["train_generator"])
        generators.validation.set_state(rank_state["validation_generator"])
    except (IndexError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("resume checkpoint has an invalid RNG state") from error


class DeiTGaussianBlur:
    """PIL radius sampling from facebookresearch/deit augment.py (Apache-2.0)."""

    def __call__(self, image: Any) -> Any:
        from PIL import ImageFilter

        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))


def build_train_transform(run: ImageNetRun) -> Any:
    train = run.train
    image_size = int(run.model["image_size"])
    if train["augmentation"] == "three_augment":
        from torchvision import transforms
        from timm.data.transforms import RandomResizedCropAndInterpolation

        # Official DeiT III augment.py: RRC, flip, one of grayscale /
        # solarization / PIL blur, then color jitter and ImageNet normalization.
        return transforms.Compose([
            RandomResizedCropAndInterpolation(image_size, scale=(0.08, 1.0), interpolation="bicubic"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                transforms.Grayscale(3), transforms.RandomSolarize(128, p=1.0), DeiTGaussianBlur(),
            ]),
            transforms.ColorJitter(*([float(train["color_jitter"])] * 3)),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    from timm.data import create_transform

    return create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=(None if float(train["color_jitter"]) == 0 else float(train["color_jitter"])),
        auto_augment=str(train["auto_augment"]),
        interpolation="bicubic",
        re_prob=float(train["reprob"]),
        re_mode="pixel",
        re_count=1,
    )


def build_eval_transform(run: ImageNetRun) -> Any:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image_size = int(run.model["image_size"])
    resize_size = int(image_size / float(run.train["eval_crop_ratio"]))
    return transforms.Compose(
        (
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )


@dataclass(frozen=True)
class _WebDatasetSplit:
    """One validated split from ModelScope's ImageNet-1K WebDataset manifest."""

    name: str
    paths: tuple[Path, ...]
    shard_lengths: tuple[int, ...]
    num_samples: int

    def contract(self) -> dict[str, int]:
        return {"samples": self.num_samples, "shards": len(self.paths)}


@dataclass(frozen=True)
class ImageNetWebDatasetManifest:
    """The fixed ModelScope ImageNet-1K WebDataset layout used by this runner."""

    root: Path
    sha256: str
    train: _WebDatasetSplit
    validation: _WebDatasetSplit

    def data_contract(
        self, *, repeated_augmentation: bool, augmentation_group_size: int | None = None,
    ) -> dict[str, Any]:
        indexed = augmentation_group_size is not None
        return {
            "format": "webdataset-v1",
            "source": IMAGENET_WDS_SOURCE,
            "manifest_sha256": self.sha256,
            "train": self.train.contract(),
            "validation": self.validation.contract(),
            "streaming": {
                "shard_order": (
                    "global-sample-permutation-then-virtual-group-rank-stride" if indexed
                    else "global-epoch-permutation-then-rank-stride-worker-quota"
                ),
                "sample_shuffle": {"algorithm": "torch.randperm", "seed_policy": "seed-epoch-v1"} if indexed else {
                    "buffer_size": WDS_SAMPLE_SHUFFLE_SIZE,
                    "initial_size": WDS_SAMPLE_SHUFFLE_INITIAL,
                },
                "source_views": 3 if repeated_augmentation else 1,
                "repeated_augmentation_placement": (
                    "global-virtual-group-repeat-then-rank-stride" if indexed else "rank-local-physical-batch-interleave"
                ),
                **({"augmentation_group_size": augmentation_group_size} if indexed else {}),
                "validation_partition": "worker-stride-full-per-rank",
                "worker_rng": "epoch-rank-worker-v1",
            },
        }


def _expected_wds_filenames(split: str, count: int, width: int) -> tuple[str, ...]:
    return tuple(
        f"imagenet1k-{split}-{index:0{width}d}.tar" for index in range(count)
    )


def _load_webdataset_split(
    value: object,
    *,
    root: Path,
    split: str,
    expected_count: int,
    expected_samples: int,
    index_width: int,
) -> _WebDatasetSplit:
    raw = _as_mapping(value, f"WebDataset manifest split {split!r}")
    _require_keys(
        raw,
        ("name", "filenames", "shard_lengths", "num_samples"),
        f"WebDataset manifest split {split!r}",
    )
    if raw["name"] != split:
        raise ValueError(f"WebDataset manifest split {split!r} has a mismatched name")
    filenames_value = raw["filenames"]
    if not isinstance(filenames_value, list) or not all(
        isinstance(name, str) for name in filenames_value
    ):
        raise ValueError(f"WebDataset manifest split {split!r} has invalid filenames")
    filenames = tuple(filenames_value)
    expected_filenames = _expected_wds_filenames(split, expected_count, index_width)
    if filenames != expected_filenames:
        raise ValueError(
            f"WebDataset manifest split {split!r} does not match the official shard list"
        )
    lengths_value = raw["shard_lengths"]
    if not isinstance(lengths_value, list) or len(lengths_value) != expected_count:
        raise ValueError(f"WebDataset manifest split {split!r} has invalid shard lengths")
    shard_lengths = tuple(
        _positive_int(length, f"WebDataset manifest {split}.shard_lengths")
        for length in lengths_value
    )
    num_samples = _positive_int(
        raw["num_samples"],
        f"WebDataset manifest {split}.num_samples",
    )
    if num_samples != expected_samples or sum(shard_lengths) != num_samples:
        raise ValueError(
            f"WebDataset manifest split {split!r} has an unexpected sample count"
        )
    paths = tuple(root / filename for filename in filenames)
    missing = [path.name for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise FileNotFoundError(
            f"WebDataset split {split!r} is missing non-empty shards: {preview}{suffix}"
        )
    return _WebDatasetSplit(
        name=split,
        paths=paths,
        shard_lengths=shard_lengths,
        num_samples=num_samples,
    )


def load_imagenet_webdataset_manifest(
    data_root: Path,
    *,
    expected_sha256: str | None = None,
) -> ImageNetWebDatasetManifest:
    """Validate the exact ModelScope ``timm/imagenet-1k-wds`` release layout."""

    root = data_root.resolve()
    manifest_path = root / "_info.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"expected ModelScope ImageNet WebDataset manifest {manifest_path}"
        )
    raw_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha256 = (
        IMAGENET_WDS_MANIFEST_SHA256
        if expected_sha256 is None
        else expected_sha256
    )
    if digest != expected_sha256:
        raise ValueError(
            "_info.json does not match the pinned timm/imagenet-1k-wds manifest; "
            "redownload the ModelScope dataset revision used by this repository"
        )
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("ImageNet WebDataset _info.json is not valid JSON") from error
    if not isinstance(raw, dict) or raw.get("name") != "imagenet1k":
        raise ValueError("_info.json is not the expected ImageNet-1K WebDataset manifest")
    splits = _as_mapping(raw.get("splits"), "WebDataset manifest splits")
    _require_keys(splits, ("train", "validation"), "WebDataset manifest splits")
    return ImageNetWebDatasetManifest(
        root=root,
        sha256=digest,
        train=_load_webdataset_split(
            splits["train"],
            root=root,
            split="train",
            expected_count=IMAGENET_TRAIN_SHARDS,
            expected_samples=IMAGENET_TRAIN_SAMPLES,
            index_width=4,
        ),
        validation=_load_webdataset_split(
            splits["validation"],
            root=root,
            split="validation",
            expected_count=IMAGENET_VALIDATION_SHARDS,
            expected_samples=IMAGENET_VALIDATION_SAMPLES,
            index_width=2,
        ),
    )


def _verify_webdataset_shard(path: Path, *, expected_samples: int) -> None:
    expected_fields = {"jpg", "cls", "json"}
    completed_keys: set[str] = set()
    current_key: str | None = None
    current_fields: set[str] = set()
    record_count = 0

    def finish_record() -> None:
        nonlocal current_key, current_fields, record_count
        if current_key is None:
            return
        if current_key in completed_keys:
            raise ValueError(
                f"WebDataset shard {path} repeats source key {current_key!r}"
            )
        if current_fields != expected_fields:
            raise ValueError(
                f"WebDataset shard {path} record {current_key!r} does not contain "
                "exactly one .jpg/.cls/.json triple"
            )
        completed_keys.add(current_key)
        record_count += 1
        current_key = None
        current_fields = set()

    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                # Match WebDataset's base_plus_ext grouping rule so the preflight
                # validates the same logical records the streaming reader will see.
                match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", member.name)
                if match is None:
                    continue
                key, suffix = match.group(1), match.group(2).lower()
                if current_key is not None and key != current_key:
                    finish_record()
                if current_key is None:
                    current_key = key
                if suffix in current_fields:
                    raise ValueError(
                        f"WebDataset shard {path} repeats field {suffix!r} for "
                        f"source key {key!r}"
                    )
                current_fields.add(suffix)
            finish_record()
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"WebDataset shard {path} is not a readable tar archive") from error
    if record_count != expected_samples:
        raise ValueError(
            f"WebDataset shard {path} has {record_count} records, expected {expected_samples}"
        )


def verify_imagenet_webdataset_payloads(manifest: ImageNetWebDatasetManifest) -> None:
    """Validate tar readability and one jpg/cls/json record triple per manifest sample."""

    for split in (manifest.train, manifest.validation):
        for path, expected_samples in zip(
            split.paths,
            split.shard_lengths,
            strict=True,
        ):
            _verify_webdataset_shard(path, expected_samples=expected_samples)


def _verify_webdataset_payloads_for_state(
    manifest: ImageNetWebDatasetManifest,
    state: DistributedState,
) -> None:
    if not state.enabled:
        verify_imagenet_webdataset_payloads(manifest)
        return
    failure: str | None = None
    if state.is_main:
        try:
            verify_imagenet_webdataset_payloads(manifest)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
    result: list[object] = [failure]
    dist.broadcast_object_list(result, src=0, device=state.device)
    if result[0] is not None:
        raise RuntimeError(f"ImageNet WebDataset payload validation failed: {result[0]}")


def _validate_webdataset_contract(value: object) -> None:
    data = _as_mapping(value, "checkpoint data contract")
    _require_keys(
        data,
        ("format", "source", "manifest_sha256", "train", "validation", "streaming"),
        "checkpoint data contract",
    )
    if data["format"] != "webdataset-v1" or data["source"] != IMAGENET_WDS_SOURCE:
        raise ValueError("checkpoint does not use the current ImageNet WebDataset contract")
    digest = data["manifest_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("checkpoint WebDataset manifest digest is invalid")
    for name, expected_samples, expected_shards in (
        ("train", IMAGENET_TRAIN_SAMPLES, IMAGENET_TRAIN_SHARDS),
        ("validation", IMAGENET_VALIDATION_SAMPLES, IMAGENET_VALIDATION_SHARDS),
    ):
        split = _as_mapping(data[name], f"checkpoint WebDataset {name} split")
        if split != {"samples": expected_samples, "shards": expected_shards}:
            raise ValueError(f"checkpoint WebDataset {name} split is invalid")
    streaming = _as_mapping(data["streaming"], "checkpoint WebDataset streaming contract")
    _require_keys(
        streaming,
        (
            "shard_order",
            "sample_shuffle",
            "source_views",
            "repeated_augmentation_placement",
            "validation_partition",
            "worker_rng",
        ),
        "checkpoint WebDataset streaming contract",
    )
    indexed = streaming["shard_order"] == "global-sample-permutation-then-virtual-group-rank-stride"
    if not indexed and streaming["shard_order"] != "global-epoch-permutation-then-rank-stride-worker-quota":
        raise ValueError("checkpoint WebDataset shard ordering is invalid")
    sample_shuffle = _as_mapping(
        streaming["sample_shuffle"],
        "checkpoint WebDataset sample shuffle contract",
    )
    expected_shuffle = {"algorithm": "torch.randperm", "seed_policy": "seed-epoch-v1"} if indexed else {
        "buffer_size": WDS_SAMPLE_SHUFFLE_SIZE,
        "initial_size": WDS_SAMPLE_SHUFFLE_INITIAL,
    }
    if sample_shuffle != expected_shuffle:
        raise ValueError("checkpoint WebDataset sample shuffle contract is invalid")
    if streaming["source_views"] not in (1, 3):
        raise ValueError("checkpoint WebDataset source view count is invalid")
    if (
        streaming["repeated_augmentation_placement"]
        != ("global-virtual-group-repeat-then-rank-stride" if indexed else "rank-local-physical-batch-interleave")
    ):
        raise ValueError("checkpoint WebDataset repeated-augmentation placement is invalid")
    if indexed:
        group_size = _positive_int(streaming.get("augmentation_group_size"), "data.augmentation_group_size")
        if group_size % 2:
            raise ValueError("checkpoint augmentation group size must be even")
    if streaming["validation_partition"] != "worker-stride-full-per-rank":
        raise ValueError("checkpoint WebDataset validation partition is invalid")
    if streaming["worker_rng"] != "epoch-rank-worker-v1":
        raise ValueError("checkpoint WebDataset worker RNG policy is invalid")


def _webdataset_seed(*values: int) -> int:
    encoded = ":".join(str(int(value)) for value in values).encode("ascii")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")


def _require_webdataset() -> Any:
    try:
        import webdataset
    except ImportError as error:
        raise RuntimeError(
            "ImageNet WebDataset training requires webdataset>=1,<2; "
            "install the package with the vision extra."
        ) from error
    return webdataset


def _webdataset_local_url(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        # WebDataset's local opener expects a drive path without file URI's extra slash.
        return f"file:{resolved.as_posix()}"
    return resolved.as_uri()


def _decode_webdataset_sample(
    sample: Mapping[str, Any],
    *,
    transform: Any,
    num_classes: int,
) -> tuple[torch.Tensor, int]:
    image_bytes = sample.get("jpg")
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("WebDataset sample is missing JPEG bytes under the 'jpg' field")
    try:
        target = int(sample["cls"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("WebDataset sample is missing a valid class id under 'cls'") from error
    if not 0 <= target < num_classes:
        raise ValueError(f"WebDataset class id {target} is outside [0, {num_classes})")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(bytes(image_bytes))) as image:
            tensor = transform(image.convert("RGB"))
    except Exception as error:
        raise ValueError("failed to decode or transform a WebDataset JPEG sample") from error
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("ImageNet transform must return a torch.Tensor")
    return tensor, target


@dataclass(frozen=True)
class _WebDatasetShardSlice:
    path: Path
    start: int
    count: int


class _ImageNetIndexedDataset(Dataset[tuple[torch.Tensor, int]]):
    """Local tar access owned by WIDS; persistent workers observe a shared epoch."""

    def __init__(self, split: _WebDatasetSplit, *, transform: Any,
                 state: DistributedState, seed: int, num_classes: int) -> None:
        self.split, self.transform, self.state = split, transform, state
        self.seed, self.num_classes = int(seed), int(num_classes)
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._seeded_epoch = None
        self._reader = None

    def __len__(self) -> int:
        return self.split.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("ImageNet epoch must be non-negative")
        self._shared_epoch.fill_(epoch)

    def __getstate__(self) -> dict[str, Any]:
        # Spawned workers own their mmap handles; never serialize an open cache.
        return {**self.__dict__, "_reader": None, "_seeded_epoch": None}

    def _get_reader(self) -> Any:
        if self._reader is None:
            try:
                from wids import ShardListDataset
            except ImportError as error:
                raise RuntimeError("virtual-group ImageNet loading requires wids; install the vision extra") from error
            import resource
            import warnings

            # Uniform random sampling visits all local shards. Keep their small
            # mmap indexes, avoiding repeated tar-header scans. Only this loader
            # process raises its soft limit; no server-wide setting is changed.
            required = 3 * len(self.split.paths) + 256
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < required:
                if hard != resource.RLIM_INFINITY and hard < required:
                    raise RuntimeError("WIDS needs a higher per-process file-descriptor limit")
                resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
            shards = [{"url": str(path.resolve()), "nsamples": count}
                      for path, count in zip(self.split.paths, self.split.shard_lengths, strict=True)]
            with warnings.catch_warnings():
                # WIDS warns about >200 cached shards before checking the limit;
                # the bound above already reserves descriptors for every shard.
                warnings.filterwarnings("ignore", message="LRU size is very large.*")
                self._reader = ShardListDataset(
                    shards, transformations=[], localname=str, keep=True,
                    lru_size=len(shards),
                )
        return self._reader

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not 0 <= index < len(self):
            raise IndexError("ImageNet source index is outside its manifest")
        worker = get_worker_info()
        epoch = int(self._shared_epoch.item())
        if worker is not None and self._seeded_epoch != epoch:
            torch.manual_seed(_webdataset_seed(self.seed, epoch, self.state.rank, worker.id, 1))
            _seed_worker(worker.id)
            self._seeded_epoch = epoch
        # Use WIDS's indexed shard primitive directly: its public dataset
        # wrapper also adds unused metadata and a cold-cache miss heuristic
        # meant for small, shard-local LRUs. Our cache retains every shard.
        shard, inner_index, _ = self._get_reader().get_shard(index)
        sample = shard[inner_index]
        # WIDS returns BytesIO fields with dotted suffixes; the common decoder
        # retains class validation, RGB conversion and ImageNet transforms.
        return _decode_webdataset_sample(
            {"jpg": sample[".jpg"].getvalue(), "cls": sample[".cls"].getvalue()},
            transform=self.transform, num_classes=self.num_classes,
        )


class VirtualGroupSampler(Sampler[int]):
    """Repeat whole source groups globally, then stride by rank.

    Reuses the repository's earlier DeiT III virtual-device schedule
    (22c89c0), following Meta RASampler's repeat-before-rank-split ordering.
    Group members are unique; each repeated index receives a fresh transform.
    """

    def __init__(self, dataset: _ImageNetIndexedDataset, *, samples_per_rank: int,
                 group_size: int, num_repeats: int = 3, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.group_size = _positive_int(group_size, "augmentation_group_size")
        self.num_repeats = _positive_int(num_repeats, "num_repeats")
        self.num_samples = _positive_int(samples_per_rank, "samples_per_rank")
        if self.num_samples % self.group_size:
            raise ValueError("samples_per_rank must be divisible by augmentation_group_size")
        self.global_groups = self.num_samples // self.group_size * dataset.state.world_size
        self.source_samples = math.ceil(self.global_groups / self.num_repeats) * self.group_size
        if len(dataset) < self.source_samples:
            raise ValueError("ImageNet dataset cannot cover the unique-source virtual-group quota")
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(_webdataset_seed(self.dataset.seed, self.epoch))
            indices = torch.randperm(len(self.dataset), generator=generator)
        else:
            indices = torch.arange(len(self.dataset))
        groups = indices[:self.source_samples].reshape(-1, self.group_size)
        repeated = torch.repeat_interleave(groups, self.num_repeats, dim=0)[:self.global_groups]
        local = repeated[self.dataset.state.rank::self.dataset.state.world_size].reshape(-1)
        return iter(local.tolist())


class _ImageNetWebDataset(IterableDataset[tuple[torch.Tensor, int]]):
    """Finite validation and quota-controlled training ImageNet WebDataset reader."""

    def __init__(
        self,
        split: _WebDatasetSplit,
        *,
        transform: Any,
        state: DistributedState,
        seed: int,
        num_classes: int,
        training: bool,
        physical_batch_size: int | None = None,
        microbatches_per_epoch: int | None = None,
        worker_count: int | None = None,
    ) -> None:
        super().__init__()
        if training:
            if (
                physical_batch_size is None
                or microbatches_per_epoch is None
                or worker_count is None
            ):
                raise ValueError("training WebDataset requires a complete batching plan")
            if physical_batch_size < 1:
                raise ValueError("training WebDataset physical batch size must be positive")
            if microbatches_per_epoch < 1 or worker_count < 1:
                raise ValueError("training WebDataset requires positive batch and worker counts")
        self.split = split
        self.transform = transform
        self.state = state
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        self.training = training
        self.physical_batch_size = physical_batch_size
        self.microbatches_per_epoch = microbatches_per_epoch
        self.worker_count = worker_count
        # Dataset copies in persistent workers must observe the parent's epoch.
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def _epoch(self) -> int:
        return int(self._shared_epoch.item())

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("WebDataset epoch must be non-negative")
        self._shared_epoch.fill_(int(epoch))

    @staticmethod
    def _worker_state() -> tuple[int, int]:
        worker = get_worker_info()
        return (0, 1) if worker is None else (worker.id, worker.num_workers)

    def _rank_shards(self, epoch: int) -> tuple[tuple[Path, int], ...]:
        shards = list(zip(self.split.paths, self.split.shard_lengths, strict=True))
        if self.training:
            random.Random(_webdataset_seed(self.seed, epoch)).shuffle(shards)
            shards = shards[self.state.rank :: self.state.world_size]
        if not shards:
            raise RuntimeError(
                f"WebDataset {self.split.name!r} has no shards for rank {self.state.rank}"
            )
        return tuple(shards)

    @staticmethod
    def _slice_shards(
        shards: Sequence[tuple[Path, int]],
        *,
        offset: int,
        count: int,
    ) -> tuple[_WebDatasetShardSlice, ...]:
        if offset < 0 or count < 0:
            raise ValueError("WebDataset shard slices must be non-negative")
        remaining = count
        skipped = offset
        slices: list[_WebDatasetShardSlice] = []
        for path, shard_length in shards:
            if skipped >= shard_length:
                skipped -= shard_length
                continue
            start = skipped
            take = min(shard_length - start, remaining)
            if take:
                slices.append(_WebDatasetShardSlice(path=path, start=start, count=take))
            remaining -= take
            skipped = 0
            if not remaining:
                return tuple(slices)
        if remaining:
            raise RuntimeError("WebDataset shard assignment exceeded its source records")
        return tuple(slices)

    def _training_batch_counts(self) -> tuple[int, ...]:
        if (
            self.physical_batch_size is None
            or self.microbatches_per_epoch is None
            or self.worker_count is None
        ):
            raise RuntimeError("training WebDataset is missing its batching plan")
        total_batches = self.microbatches_per_epoch
        base, extra = divmod(total_batches, self.worker_count)
        return tuple(base + (worker_id < extra) for worker_id in range(self.worker_count))

    def _training_slices(
        self,
        *,
        worker_id: int,
        worker_count: int,
        epoch: int,
    ) -> tuple[_WebDatasetShardSlice, ...]:
        if worker_count != self.worker_count:
            raise RuntimeError(
                "training WebDataset worker count changed after its batching plan was resolved"
            )
        batches = self._training_batch_counts()
        if not 0 <= worker_id < len(batches):
            raise RuntimeError("training WebDataset reported an invalid worker id")
        source_counts = tuple(
            batch_count * self.physical_batch_size
            for batch_count in batches
        )
        shards = self._rank_shards(epoch)
        available = sum(shard_length for _, shard_length in shards)
        required = sum(source_counts)
        if required > available:
            raise RuntimeError(
                "WebDataset rank cannot cover the planned epoch with unique source records: "
                f"requires {required}, has {available}; reduce the effective batch or workers"
            )
        return self._slice_shards(
            shards,
            offset=sum(source_counts[:worker_id]),
            count=source_counts[worker_id],
        )

    def _validation_slices(
        self,
        *,
        worker_id: int,
        worker_count: int,
    ) -> tuple[_WebDatasetShardSlice, ...]:
        shards = self._rank_shards(epoch=0)
        assigned = shards[worker_id::worker_count]
        if not assigned:
            raise RuntimeError(
                f"WebDataset {self.split.name!r} has no shards for worker {worker_id}"
            )
        return tuple(
            _WebDatasetShardSlice(path=path, start=0, count=shard_length)
            for path, shard_length in assigned
        )

    def _raw_samples(
        self,
        shards: Sequence[_WebDatasetShardSlice],
        *,
        shuffle_seed: int | None,
    ) -> Iterator[Mapping[str, Any]]:
        wds = _require_webdataset()

        def records() -> Iterator[Mapping[str, Any]]:
            for shard in shards:
                dataset = wds.WebDataset(
                    [_webdataset_local_url(shard.path)],
                    handler=wds.handlers.reraise_exception,
                    shardshuffle=False,
                    nodesplitter=None,
                    workersplitter=None,
                )
                yield from itertools.islice(
                    dataset,
                    shard.start,
                    shard.start + shard.count,
                )

        source: Iterator[Mapping[str, Any]] = records()
        if shuffle_seed is not None:
            source = wds.filters.shuffle(
                WDS_SAMPLE_SHUFFLE_SIZE,
                initial=WDS_SAMPLE_SHUFFLE_INITIAL,
                seed=shuffle_seed,
            )(source)
        yield from source

    def _decode(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, int]:
        return _decode_webdataset_sample(
            sample,
            transform=self.transform,
            num_classes=self.num_classes,
        )

    def _iter_training(
        self,
        *,
        worker_id: int,
        worker_count: int,
    ) -> Iterator[tuple[torch.Tensor, int]]:
        batches = self._training_batch_counts()[worker_id]
        if not batches:
            return
        source = iter(
            self._raw_samples(
                self._training_slices(
                    worker_id=worker_id,
                    worker_count=worker_count,
                    epoch=self._epoch,
                ),
                shuffle_seed=_webdataset_seed(
                    self.seed,
                    self._epoch,
                    self.state.rank,
                    worker_id,
                ),
            )
        )
        produced = 0
        for sample in source:
            yield self._decode(sample)
            produced += 1
        if produced != batches * self.physical_batch_size:
            raise RuntimeError("WebDataset stream ended before its unique-source batch quota was produced")

    def _iter_validation(
        self,
        *,
        worker_id: int,
        worker_count: int,
    ) -> Iterator[tuple[torch.Tensor, int]]:
        for sample in self._raw_samples(
            self._validation_slices(
                worker_id=worker_id,
                worker_count=worker_count,
            ),
            shuffle_seed=None,
        ):
            yield self._decode(sample)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        worker_id, worker_count = self._worker_state()
        if get_worker_info() is not None:
            # A resumed loader creates fresh workers; persistent workers do not
            # rerun worker_init_fn. Seed each iterator independently of lifetime.
            seed = _webdataset_seed(
                self.seed, self._epoch if self.training else 0,
                self.state.rank, worker_id, int(self.training),
            )
            torch.manual_seed(seed)
            _seed_worker(worker_id)
        if self.training:
            yield from self._iter_training(
                worker_id=worker_id,
                worker_count=worker_count,
            )
        else:
            yield from self._iter_validation(
                worker_id=worker_id,
                worker_count=worker_count,
            )


def _loader_kwargs(
    *,
    workers: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "pin_memory": True,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if workers:
        # Workers start after CUDA/DDP initialization; avoid inheriting its state.
        kwargs["multiprocessing_context"] = "spawn"
        # One prefetched physical batch per worker bounds host-side image memory.
        kwargs["prefetch_factor"] = 1
    return kwargs


def build_loaders(
    run: ImageNetRun,
    data_root: Path,
    state: DistributedState,
    *,
    requested_grad_accum: int | None,
) -> tuple[
    DataLoader[Any],
    DataLoader[Any],
    _ImageNetWebDataset | VirtualGroupSampler,
    BatchingPlan,
    LoaderRandomGenerators,
    dict[str, Any],
]:
    manifest = load_imagenet_webdataset_manifest(data_root)
    _verify_webdataset_payloads_for_state(manifest, state)
    batching_plan = resolve_batching_plan(
        run,
        state,
        dataset_size=manifest.train.num_samples,
        requested_grad_accum=requested_grad_accum,
    )
    train_workers = int(run.train["train_workers"])
    val_workers = int(run.train["val_workers"])
    effective_train_workers = min(
        max(1, train_workers),
        batching_plan.microbatches_per_epoch,
    )
    group_size = batching_plan.augmentation_group_size
    if group_size is None and state.world_size * effective_train_workers > len(manifest.train.paths):
        raise ValueError("train worker count exceeds the available WebDataset shard partition")
    if val_workers > len(manifest.validation.paths):
        raise ValueError("validation worker count exceeds the 64 validation shards")

    source_views = 3 if bool(run.train["repeated_aug"]) else 1
    data_contract = manifest.data_contract(
        repeated_augmentation=bool(run.train["repeated_aug"]),
        augmentation_group_size=group_size,
    )
    if group_size is not None:
        train_dataset = _ImageNetIndexedDataset(
            manifest.train, transform=build_train_transform(run), state=state,
            seed=int(run.train["seed"]), num_classes=int(run.model["num_classes"]),
        )
        epoch_controller = VirtualGroupSampler(
            train_dataset, samples_per_rank=batching_plan.samples_per_rank,
            group_size=group_size, num_repeats=source_views,
        )
    else:
        train_dataset = _ImageNetWebDataset(
            manifest.train, transform=build_train_transform(run), state=state,
            seed=int(run.train["seed"]), num_classes=int(run.model["num_classes"]),
            training=True, physical_batch_size=batching_plan.physical_batch_size,
            microbatches_per_epoch=batching_plan.microbatches_per_epoch,
            worker_count=effective_train_workers,
        )
        epoch_controller = train_dataset
    # The public DeiT command does not enable --dist-eval, so every rank scans
    # all 50k validation examples and metric reduction preserves that protocol.
    val_dataset = _ImageNetWebDataset(
        manifest.validation,
        transform=build_eval_transform(run),
        state=state,
        seed=int(run.train["seed"]),
        num_classes=int(run.model["num_classes"]),
        training=False,
    )
    generators = LoaderRandomGenerators(
        train=torch.Generator().manual_seed(int(run.train["seed"]) + state.rank),
        validation=torch.Generator().manual_seed(
            int(run.train["seed"]) + state.world_size + state.rank
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batching_plan.physical_batch_size,
        **({"sampler": epoch_controller} if group_size is not None else {}),
        drop_last=True,
        **_loader_kwargs(
            workers=0 if train_workers == 0 else effective_train_workers,
            generator=generators.train,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(1.5 * int(run.train["batch_size"]))),
        drop_last=False,
        **_loader_kwargs(workers=val_workers, generator=generators.validation),
    )
    return (
        train_loader,
        val_loader,
        epoch_controller,
        batching_plan,
        generators,
        data_contract,
    )


def prepare_operator_backend(run: ImageNetRun, device: torch.device) -> None:
    if run.operator["implementation"] == "cuda":
        from memsolve.ball import cuda

        cuda.load(device=device)


def build_model(run: ImageNetRun) -> nn.Module:
    """Call the shared ViT³-style MemSolve backbone factory owned by integrations."""

    try:
        from integrations.timm import create_memsolve_vit
    except ImportError as error:
        raise RuntimeError(
            "ImageNet training requires integrations.timm.create_memsolve_vit; "
            "the shared ViT³-style backbone adapter is not available."
        ) from error

    model = run.model
    return create_memsolve_vit(
        image_size=int(model["image_size"]),
        patch_size=int(model["patch_size"]),
        num_classes=int(model["num_classes"]),
        embed_dim=int(model["embed_dim"]),
        depth=int(model["depth"]),
        num_heads=int(model["num_heads"]),
        rank=int(model["rank"]),
        qk_conv_kernel_size=run.operator["qk_conv_kernel_size"],
        output_gate_rank=run.operator["output_gate_rank"],
        mlp_ratio=float(model["mlp_ratio"]),
        bias=bool(run.operator["bias"]),
        implementation=str(run.operator["implementation"]),
        drop_path_rate=float(model["drop_path_rate"]),
        drop_path_schedule=str(model["drop_path_schedule"]),
        norm_eps=float(model["norm_eps"]),
    )


def build_model_ema(model: nn.Module, run: ImageNetRun) -> Any | None:
    """Create upstream's constant-decay shadow before compilation/wrapping.

    timm V3 provides foreach updates. Parameter iteration handles MemSolve's
    non-tensor extra state; buffers are copied (this backbone has no running
    statistics). Call update without a step to retain the constant decay from
    the first optimizer update, as in RoPE-ViT's ModelEma.
    """
    if not run.train["ema"]:
        return None
    from timm.utils import ModelEmaV3

    ema = ModelEmaV3(_unwrap_model(model), decay=float(run.train["ema_decay"]),
                     use_warmup=False, foreach=True, exclude_buffers=True)
    ema.module.requires_grad_(False)
    return ema


def _checkpoint_model_state(checkpoint: Mapping[str, Any], *, weights: str = "model") -> dict[str, Any]:
    if weights not in {"model", "model_ema"}:
        raise ValueError("checkpoint weights must be model or model_ema")
    state = checkpoint.get(weights)
    if not isinstance(state, dict) or not all(
        isinstance(key, str)
        and (
            isinstance(value, torch.Tensor)
            or (key.endswith("_extra_state") and isinstance(value, dict))
        )
        for key, value in state.items()
    ):
        raise ValueError(
            "checkpoint must contain tensor model state and MemSolve _extra_state contracts"
        )
    return copy.deepcopy(state)


def _position_extra_tokens(tokens: int) -> int:
    for extra in (0, 1, 2):
        side = math.isqrt(tokens - extra)
        if side * side == tokens - extra:
            return extra
    raise ValueError(f"position embedding length {tokens} does not encode a square patch grid")


def interpolate_position_embedding(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Bicubically resize a learned 2D patch table for downstream checkpoint transfer."""

    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != target.shape[0]:
        raise ValueError("position embeddings must have shape [batch, tokens, channels]")
    if source.shape[-1] != target.shape[-1]:
        raise ValueError("position embeddings must have the same channel dimension")
    source_extra = _position_extra_tokens(source.shape[1])
    target_extra = _position_extra_tokens(target.shape[1])
    if source_extra != target_extra:
        raise ValueError("source and target position embeddings disagree on extra tokens")
    source_side = math.isqrt(source.shape[1] - source_extra)
    target_side = math.isqrt(target.shape[1] - target_extra)
    extra = source[:, :source_extra]
    patch = source[:, source_extra:]
    patch = patch.reshape(
        source.shape[0], source_side, source_side, source.shape[-1]
    ).permute(0, 3, 1, 2)
    patch = functional.interpolate(
        patch.float(),
        size=(target_side, target_side),
        mode="bicubic",
        align_corners=False,
    )
    patch = patch.permute(0, 2, 3, 1).reshape(
        source.shape[0], target_side * target_side, -1
    )
    result = torch.cat((extra.float(), patch), dim=1).to(dtype=target.dtype)
    return result


def _no_weight_decay(model: nn.Module) -> set[str]:
    parameter_names = {name for name, _ in model.named_parameters()}
    candidate = getattr(model, "no_weight_decay", None)
    if candidate is None:
        # The shared wrapper delegates to timm's ViT encoder, so retain the
        # published no-decay treatment without teaching the experiment about
        # any other model parameters.
        return {
            name
            for name in parameter_names
            if name == "pos_embed"
            or name.endswith(".pos_embed")
            or name == "cls_token"
            or name.endswith(".cls_token")
        }
    names = candidate()
    if not isinstance(names, (set, frozenset)) or not all(
        isinstance(name, str) for name in names
    ):
        raise TypeError("model.no_weight_decay() must return a set of parameter names")
    resolved: set[str] = set()
    for name in names:
        if name in parameter_names:
            resolved.add(name)
            continue
        matches = [candidate for candidate in parameter_names if candidate.endswith(f".{name}")]
        if len(matches) != 1:
            raise ValueError(f"model.no_weight_decay() returned unknown name {name!r}")
        resolved.add(matches[0])
    return resolved


def build_optimizer(model: nn.Module, run: ImageNetRun) -> tuple[torch.optim.Optimizer, str]:
    from timm.optim import param_groups_weight_decay

    groups = param_groups_weight_decay(
        model, weight_decay=float(run.train["weight_decay"]),
        no_weight_decay_list=_no_weight_decay(model),
    )
    if run.train["optimizer"] == "fused_lamb":
        try:
            from apex.optimizers import FusedLAMB
        except ImportError as error:
            raise RuntimeError("fused_lamb requires NVIDIA Apex with CUDA extensions; see docs/IMAGENET_DEIT3.md") from error
        if any(p.device.type != "cuda" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("fused LAMB requires CUDA FP32 parameters; BF16 autocast activations are supported")
        return FusedLAMB(
            groups, lr=float(run.train["lr"]), betas=(0.9, 0.999), eps=1e-8,
            bias_correction=True, adam_w_mode=True, grad_averaging=True,
            max_grad_norm=float(run.train["clip_grad"]), use_nvlamb=False,
        ), "apex.lamb.fused"
    return (
        torch.optim.AdamW(
            groups, lr=float(run.train["lr"]), betas=(0.9, 0.999), eps=1e-8, fused=True,
        ),
        "torch.adamw.fused",
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    run: ImageNetRun,
    updates_per_epoch: int,
) -> Any:
    _positive_int(updates_per_epoch, "updates_per_epoch")
    from timm.scheduler import CosineLRScheduler

    return CosineLRScheduler(
        optimizer,
        t_initial=int(run.train["epochs"]) * updates_per_epoch,
        lr_min=float(run.train["min_lr"]),
        cycle_mul=1.0,
        cycle_decay=1.0,
        cycle_limit=1,
        warmup_t=int(run.train["warmup_epochs"]) * updates_per_epoch,
        warmup_lr_init=float(run.train["warmup_lr"]),
        warmup_prefix=False,
        t_in_epochs=False,
    )


class DeiTBinaryCrossEntropy(nn.BCEWithLogitsLoss):
    """DeiT III targets: positive membership after unsmoothed Mixup/CutMix.

    Matches facebookresearch/deit engine.py (targets.gt(0)) and the default
    BCEWithLogitsLoss reduction over both examples and classes.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=logits.shape[-1])
        targets = targets.gt(0).to(dtype=torch.float32)
        return super().forward(logits, targets)


def build_mixup_and_loss(run: ImageNetRun) -> tuple[Any | None, nn.Module]:
    from timm.data import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

    if run.train["bce_loss"] and float(run.train["label_smoothing"]) != 0:
        raise ValueError("DeiT III multi-label BCE requires label_smoothing = 0")
    mixup_active = (
        float(run.train["mixup"]) > 0
        or float(run.train["cutmix"]) > 0
    )
    mixup = None
    if mixup_active:
        mixup = Mixup(
            mixup_alpha=float(run.train["mixup"]),
            cutmix_alpha=float(run.train["cutmix"]),
            cutmix_minmax=None,
            prob=float(run.train["mixup_prob"]),
            switch_prob=float(run.train["mixup_switch_prob"]),
            mode=str(run.train["mixup_mode"]),
            label_smoothing=float(run.train["label_smoothing"]),
            num_classes=int(run.model["num_classes"]),
        )
    if run.train["bce_loss"]:
        return mixup, DeiTBinaryCrossEntropy()
    if mixup_active:
        return mixup, SoftTargetCrossEntropy()
    if float(run.train["label_smoothing"]) > 0:
        return mixup, LabelSmoothingCrossEntropy(
            smoothing=float(run.train["label_smoothing"])
        )
    return mixup, nn.CrossEntropyLoss()


def apply_virtual_group_mixup(
    images: torch.Tensor, targets: torch.Tensor, mixup: Any, group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent timm batch-mode draws, without splitting model forwards."""
    group_size = _positive_int(group_size, "augmentation_group_size")
    if group_size % 2:
        raise ValueError("Mixup/CutMix virtual groups must be even")
    if images.shape[0] != targets.shape[0] or images.shape[0] % group_size:
        raise ValueError("physical batch must consist of complete virtual augmentation groups")
    labels = []
    for start in range(0, images.shape[0], group_size):
        stop = start + group_size
        mixed_images, mixed_targets = mixup(images[start:stop], targets[start:stop])
        # timm mutates the image view. Keep the adapter correct for callables
        # returning a new tensor as well, without extra copies in the timm case.
        if mixed_images.data_ptr() != images[start:stop].data_ptr():
            images[start:stop].copy_(mixed_images)
        labels.append(mixed_targets)
    return images, torch.cat(labels, dim=0)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _reduce_metrics(
    packed: torch.Tensor, state: DistributedState
) -> tuple[float, float, float]:
    if packed.shape != (4,) or packed.dtype != torch.float64:
        raise ValueError("metrics must be a float64 tensor with shape [4]")
    if state.enabled:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    loss_sum, correct1, correct5, count = packed.tolist()
    if count == 0:
        raise RuntimeError("no samples were processed")
    return (
        loss_sum / count,
        100.0 * correct1 / count,
        100.0 * correct5 / count,
    )


def _autocast() -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _assert_finite_loss(loss: torch.Tensor, *, epoch: int, step: int) -> None:
    message = f"non-finite training loss at epoch {epoch + 1}, step {step + 1}"
    if loss.is_cuda:
        # This checks the default CUDA stream without forcing a host readback.
        torch._assert_async(torch.isfinite(loss), message)
    elif not torch.isfinite(loss):
        raise FloatingPointError(message)


def prepare_training_model(model: nn.Module, state: DistributedState, execution: str) -> nn.Module:
    """Compile surrounding vision blocks; retain the tested MemSolve CUDA boundary."""
    if execution not in {"eager", "graph", "compile-graph"}:
        raise ValueError(f"unsupported training execution: {execution}")
    if execution == "compile-graph":
        from memsolve import MemSolve
        import torch._dynamo.config as dynamo_config
        import torch._functorch.config as functorch_config

        # Forward uses BF16 autocast; backward runs outside it, as recommended
        # by PyTorch AMP. AOTAutograd must specialize for that same context.
        functorch_config.backward_pass_autocast = "off"

        # timm's blocks share Python code but specialize on their per-layer
        # DropPath probabilities. Reserve train/eval variants for each block
        # instead of hitting Dynamo's default eight-variant eager fallback.
        mixer_count = sum(isinstance(module, MemSolve) for module in model.modules())
        dynamo_config.recompile_limit = max(
            dynamo_config.recompile_limit, 2 * mixer_count + 4
        )
        for module in model.modules():
            if isinstance(module, MemSolve):
                module.forward = torch.compiler.disable(module.forward)
                # The CUDA mixer already fuses normalization and its gate.
        # One outer graph captures all compiled segments and CUDA MemSolve calls.
        # Disable Inductor's own graphs so graph pools and RNG have one owner.
        # Preserve eager BF16 rounding between fused pointwise operations.
        # This retains the numerical contract while removing intermediate IO.
        model.compile(backend="inductor", fullgraph=False, dynamic=False,
                      options={"triton.cudagraphs": False, "emulate_precision_casts": True})
    if state.enabled:
        if execution == "eager":
            return DistributedDataParallel(model, device_ids=[state.local_rank])
        stream = torch.cuda.Stream(device=state.device)
        stream.wait_stream(torch.cuda.current_stream(state.device))
        with torch.cuda.stream(stream):
            model = DistributedDataParallel(
                model, device_ids=[state.local_rank], static_graph=True,
                gradient_as_bucket_view=True,
            )
        torch.cuda.current_stream(state.device).wait_stream(stream)
        model._memsolve_graph_stream = stream
    return model


class ImageNetGraphStep:
    """Capture a complete fixed-shape accumulated update, including DDP sync.

    Optimizer, clipping, data augmentation and scheduling remain outside. Warmup
    never updates weights and restores RNG/buffers before the first real replay.
    This owner persists across epochs; checkpoint loading must happen before it.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, *, grad_accum: int = 1):
        self.grad_accum = _positive_int(grad_accum, "grad_accum")
        self.model, self.criterion = model, criterion
        self.graph = None
        self.parameters = tuple(model.parameters())

    def _capture(self, images: tuple[torch.Tensor, ...], targets: tuple[torch.Tensor, ...]) -> None:
        if any(not x.is_cuda for x in (*images, *targets)):
            raise ValueError("CUDA Graph requires CUDA images and targets")
        self.images = tuple(x.clone() for x in images)
        self.targets = tuple(x.clone() for x in targets)
        cpu_rng = torch.get_rng_state()
        device = images[0].device
        cuda_rng = torch.cuda.get_rng_state(device)
        buffers = [(b, b.clone()) for b in self.model.buffers()]
        stream = getattr(self.model, "_memsolve_graph_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        def forward_backward(*, synchronize_each: bool = False):
            logits_batches, losses = [], []
            for index, (image, target) in enumerate(zip(self.images, self.targets, strict=True)):
                context = (
                    self.model.no_sync()
                    if isinstance(self.model, DistributedDataParallel) and index + 1 < self.grad_accum and not synchronize_each
                    else nullcontext()
                )
                with context:
                    with _autocast():
                        logits = self.model(image)
                        loss = self.criterion(logits, target)
                        backward_loss = loss / self.grad_accum if self.grad_accum > 1 else loss
                    backward_loss.backward()
                logits_batches.append(logits)
                losses.append(loss)
            if self.grad_accum == 1:
                return logits_batches[0], losses[0]
            return torch.cat(logits_batches), torch.stack(losses).mean()
        with torch.cuda.stream(stream):
            # PyTorch requires >=11 eager DDP iterations before full capture.
            for warmup in range(11):
                self.model.zero_grad(set_to_none=True)
                # Static DDP must discover its graph on a synchronized first
                # backward before no_sync accumulation (pytorch/pytorch#143580).
                forward_backward(synchronize_each=(warmup == 0))
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.model.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.logits, self.loss = forward_backward()
        self.gradients = tuple(p.grad for p in self.parameters)
        torch.cuda.current_stream(device).wait_stream(stream)
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)

    def close(self) -> None:
        # NCCL retains captured communicators until the graph is destroyed.
        # Release the graph before destroy_process_group to avoid shutdown hangs.
        if self.graph is not None:
            torch.cuda.synchronize(self.images[0].device)
            self.graph.reset()
            self.graph = None
            self.model.zero_grad(set_to_none=True)
            self.gradients = ()
            self.images = self.targets = self.logits = self.loss = None

    def __call__(
        self, images: torch.Tensor | Sequence[torch.Tensor], targets: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.model.training:
            raise RuntimeError("training graph cannot replay in evaluation mode")
        images = (images,) if isinstance(images, torch.Tensor) else tuple(images)
        targets = (targets,) if isinstance(targets, torch.Tensor) else tuple(targets)
        if len(images) != self.grad_accum or len(targets) != self.grad_accum:
            raise ValueError("CUDA Graph requires one complete grad_accum group")
        if any(x.shape != images[0].shape for x in images) or any(x.shape != targets[0].shape for x in targets):
            raise ValueError("CUDA Graph microbatches must have equal shapes")
        if self.graph is None:
            self._capture(images, targets)
        for source, destination in zip((*images, *targets), (*self.images, *self.targets), strict=True):
            if source.shape != destination.shape or source.dtype != destination.dtype or source.device != destination.device:
                raise ValueError("CUDA Graph input shape/dtype/device changed")
            destination.copy_(source, non_blocking=True)
        # Restores graph-owned gradients after epoch/evaluation zero_grad calls.
        for parameter, gradient in zip(self.parameters, self.gradients):
            parameter.grad = gradient
        self.graph.replay()
        return self.logits, self.loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    epoch_controller: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    mixup: Any | None,
    scheduler: Any,
    *,
    epoch: int,
    state: DistributedState,
    run: ImageNetRun,
    batching_plan: BatchingPlan,
    print_freq: int,
    graph_step: ImageNetGraphStep | None = None,
    model_ema: Any | None = None,
) -> tuple[float, float, float]:
    if hasattr(epoch_controller, "set_epoch"):
        epoch_controller.set_epoch(epoch)
    model.train()
    metric_totals = torch.zeros(4, device=state.device, dtype=torch.float64)
    processed_examples = 0
    optimizer_updates = 0
    processed_steps = 0
    started = time.perf_counter()
    model.zero_grad(set_to_none=True)
    pending_images, pending_targets, pending_metric_targets = [], [], []
    for step, (images, targets) in zip(
        range(batching_plan.microbatches_per_epoch),
        loader,
        strict=False,
    ):
        images = images.to(state.device, non_blocking=True)
        targets = targets.to(state.device, non_blocking=True)
        metric_targets = targets
        if mixup is not None:
            if batching_plan.augmentation_group_size is None:
                images, targets = mixup(images, targets)
            else:
                images, targets = apply_virtual_group_mixup(
                    images, targets, mixup, batching_plan.augmentation_group_size,
                )

        update_boundary = (step + 1) % batching_plan.grad_accum == 0
        sync_context = nullcontext()
        if state.enabled and not update_boundary:
            if not isinstance(model, DistributedDataParallel):
                raise RuntimeError("distributed ImageNet training requires DistributedDataParallel")
            sync_context = model.no_sync()
        if graph_step is not None:
            pending_images.append(images)
            pending_targets.append(targets)
            pending_metric_targets.append(metric_targets)
            if not update_boundary:
                continue
            logits, data_loss = graph_step(pending_images, pending_targets)
            metric_targets = torch.cat(pending_metric_targets)
            pending_images.clear()
            pending_targets.clear()
            pending_metric_targets.clear()
            _assert_finite_loss(data_loss, epoch=epoch, step=step)
        else:
            with sync_context:
                with _autocast():
                    logits = model(images)
                    data_loss = criterion(logits, targets)
                    loss = data_loss / batching_plan.grad_accum
                _assert_finite_loss(data_loss, epoch=epoch, step=step)
                loss.backward()
        if update_boundary:
            # Fused LAMB clips once internally after accumulation and DDP sync.
            if run.train["optimizer"] != "fused_lamb" and float(run.train["clip_grad"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(run.train["clip_grad"]))
            optimizer.step()
            if model_ema is not None:
                model_ema.update(_unwrap_model(model))
            if graph_step is None:
                model.zero_grad(set_to_none=True)
            # Match upstream zero-based update indexing, once per accumulated batch.
            scheduler.step_update(epoch * batching_plan.updates_per_epoch + optimizer_updates)
            optimizer_updates += 1

        batch = logits.shape[0]
        metric_totals[0].add_(data_loss.detach().to(dtype=torch.float64), alpha=batch)
        if metric_targets.ndim == 1:
            correct1 = (logits.detach().argmax(dim=1) == metric_targets).sum()
            correct5 = (
                logits.detach().topk(k=min(5, logits.shape[1]), dim=1).indices.eq(
                    metric_targets[:, None]
                ).any(dim=1).sum().to(dtype=torch.float64)
            )
            metric_totals[1].add_(correct1.to(dtype=torch.float64))
            metric_totals[2].add_(correct5)
        metric_totals[3].add_(batch)
        processed_examples += batch
        processed_steps += batching_plan.grad_accum if graph_step is not None else 1
        if state.is_main and print_freq > 0 and (step + 1) % print_freq == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "steps": batching_plan.microbatches_per_epoch,
                        "optimizer_step": optimizer_updates,
                        "optimizer_steps": batching_plan.updates_per_epoch,
                        "loss": data_loss.detach().item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "images_per_second": processed_examples / max(elapsed, 1.0e-9),
                    }
                ),
                flush=True,
            )
    if processed_steps != batching_plan.microbatches_per_epoch:
        raise RuntimeError("ImageNet train loader ended before the resolved batching plan")
    if optimizer_updates != batching_plan.updates_per_epoch:
        raise RuntimeError("ImageNet epoch ended without complete optimizer updates")
    return _reduce_metrics(metric_totals, state)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    state: DistributedState,
) -> tuple[float, float, float]:
    metrics = evaluate_models(model, loader, state=state)
    return metrics["val_loss"], metrics["val_acc1"], metrics["val_acc5"]


@torch.no_grad()
def evaluate_models(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    state: DistributedState,
    model_ema: Any | None = None,
) -> dict[str, float]:
    """Evaluate both weight sets on the same decoded validation batches."""
    models = {"val": model}
    if model_ema is not None:
        models["ema_val"] = model_ema.module
    for variant in models.values():
        variant.eval()
    criterion = nn.CrossEntropyLoss()
    totals = {key: torch.zeros(4, device=state.device, dtype=torch.float64) for key in models}
    for images, targets in loader:
        images = images.to(state.device, non_blocking=True)
        targets = targets.to(state.device, non_blocking=True)
        for key, variant in models.items():
            with _autocast():
                logits = variant(images)
                loss = criterion(logits, targets)
            batch = images.shape[0]
            correct1 = (logits.argmax(dim=1) == targets).sum()
            correct5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(
                targets[:, None]).any(dim=1).sum()
            totals[key][0].add_(loss.detach().to(dtype=torch.float64), alpha=batch)
            totals[key][1].add_(correct1.to(dtype=torch.float64))
            totals[key][2].add_(correct5.to(dtype=torch.float64))
            totals[key][3].add_(batch)
    return {f"{key}_{name}": value for key, total in totals.items()
            for name, value in zip(("loss", "acc1", "acc5"), _reduce_metrics(total, state), strict=True)}


def _source_revision() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"),
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    """Replace a checkpoint only after its complete temporary file is durable."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _record_runtime_metadata(
    output: Path,
    *,
    parameter_count: int,
    resolved_optimizer: str,
    run: ImageNetRun,
    batching_plan: BatchingPlan,
) -> None:
    path = output / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    metadata["model"] = {"trainable_parameters": parameter_count}
    metadata["optimizer_resolved"] = resolved_optimizer
    metadata["validation_weights"] = ["model", "model_ema"] if run.train["ema"] else ["model"]
    if run.train.get("execution") == "compile-graph":
        metadata["compiler"] = {
            "backend": "inductor", "emulate_precision_casts": True,
            "backward_pass_autocast": "off", "inductor_cuda_graphs": False,
            "outer_cuda_graph": True,
        }
    metadata["recipe_fidelity"] = _recipe_fidelity(
        run,
        batching_plan=batching_plan,
        resolved_optimizer=resolved_optimizer,
    )
    _atomic_json(path, metadata)


def _recipe_fidelity(
    run: ImageNetRun,
    *,
    batching_plan: BatchingPlan,
    resolved_optimizer: str,
) -> str:
    non_semantic_overrides = {
        "batch_size",
        "grad_accum",
        "train_workers",
        "val_workers",
        "save_every",
        "execution",
    }
    # A custom TOML can alter hyperparameters without creating CLI overrides.
    recipe = run.train.get("recipe", "vit3")
    is_deit = recipe in DEIT3_CONFIGS
    with RECIPE_CONFIGS[recipe].open("rb") as handle:
        canonical = tomllib.load(handle)
    canonical_train = {
        **canonical["defaults"],
        **canonical["tiers"][run.tier][run.phase],
    }
    canonical_train.pop("input_size")
    phase_drop_path = canonical_train.pop("drop_path_rate", None)
    matches_recipe = all(
        run.train.get(key) == value
        for key, value in canonical_train.items()
        if key not in non_semantic_overrides
    )
    tier_model = {key: value for key, value in canonical["tiers"][run.tier].items() if not isinstance(value, dict)}
    if phase_drop_path is not None:
        tier_model["drop_path_rate"] = phase_drop_path
    matches_recipe = matches_recipe and all(run.model.get(key) == value for key, value in tier_model.items())
    matches_recipe = matches_recipe and run.model["image_size"] == canonical["tiers"][run.tier][run.phase]["input_size"]
    if (
        matches_recipe
        and not (set(run.overrides) - non_semantic_overrides)
        and batching_plan.effective_batch_size == int(run.train["effective_batch"])
        and resolved_optimizer == ("apex.lamb.fused" if canonical_train["optimizer"] == "fused_lamb" else "torch.adamw.fused")
    ):
        if recipe == "ropevit_400":
            return "ropevit-derived"
        return "deit3-derived" if is_deit else "vit3-derived"
    return "explicitly-modified"


def _prepare_output(
    output: Path,
    args: argparse.Namespace,
    run: ImageNetRun,
    state: DistributedState,
    batching_plan: BatchingPlan,
    data_contract: Mapping[str, Any],
) -> Path:
    output = output.resolve()
    failure: str | None = None
    source_revision = _source_revision() if state.is_main else None
    if state.is_main:
        try:
            output.mkdir(parents=True, exist_ok=True)
            if args.resume is None and any(output.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite a non-empty output directory: {output}"
                )
            metadata = {
                "event": "start",
                "run": run.as_dict(batching_plan, data_contract),
                "launcher": {
                    "world_size": state.world_size,
                    "rank": state.rank,
                    "local_rank": state.local_rank,
                },
                "environment": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(state.device),
                    "source_revision": source_revision,
                },
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }
            try:
                import timm

                metadata["environment"]["timm"] = timm.__version__
            except ImportError:
                metadata["environment"]["timm"] = None
            _atomic_json(output / "metadata.json", metadata)
        except Exception as error:  # Broadcast rank-zero setup failures to torchrun peers.
            failure = f"{type(error).__name__}: {error}"
    if state.enabled:
        payload: list[str | None] = [failure]
        dist.broadcast_object_list(payload, src=0)
        failure = payload[0]
    if failure is not None:
        raise RuntimeError(f"unable to initialize output directory: {failure}")
    _barrier(state)
    return output


def _checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    run: ImageNetRun,
    batching_plan: BatchingPlan,
    data_contract: Mapping[str, Any],
    best_acc1: float,
    rng: Mapping[str, Any],
    model_ema: Any | None = None,
    best_ema_acc1: float | None = None,
) -> dict[str, Any]:
    if (model_ema is not None) != run.train["ema"]:
        raise ValueError("EMA instance does not match the checkpoint training contract")
    if model_ema is not None and best_ema_acc1 is None:
        raise ValueError("EMA checkpoint requires best_ema_acc1")
    contract = run.checkpoint_contract(batching_plan, data_contract)
    return {
        "format_version": IMAGENET_CHECKPOINT_FORMAT,
        "epoch": epoch,
        "best_acc1": best_acc1,
        "best_ema_acc1": best_ema_acc1,
        "selected_weights": "model",
        "contract": contract,
        "contract_digest": checkpoint_contract_digest(contract),
        "model": _unwrap_model(model).state_dict(),
        **({"model_ema": model_ema.module.state_dict()} if model_ema is not None else {}),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": dict(rng),
    }


def _load_resume(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    run: ImageNetRun,
    batching_plan: BatchingPlan,
    data_contract: Mapping[str, Any],
    state: DistributedState,
    generators: LoaderRandomGenerators,
    model_ema: Any | None = None,
) -> tuple[int, float, float | None]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint must be a mapping")
    if validate_checkpoint_contract(checkpoint) != run.checkpoint_contract(
        batching_plan,
        data_contract,
    ):
        raise ValueError("resume checkpoint contract does not match the requested run")
    if (model_ema is not None) != run.train["ema"]:
        raise ValueError("EMA instance does not match the resume training contract")
    best_ema_acc1 = checkpoint.get("best_ema_acc1")
    if model_ema is not None:
        if not isinstance(checkpoint.get("model_ema"), dict):
            raise ValueError("resume checkpoint is missing model_ema")
        if not isinstance(best_ema_acc1, (float, int)) or not math.isfinite(best_ema_acc1):
            raise ValueError("resume checkpoint has an invalid best_ema_acc1")
        model_ema.module.load_state_dict(checkpoint["model_ema"], strict=True)
    rng_state = checkpoint.get("rng")
    _resume_rng_state_for_rank(rng_state, state=state)
    _unwrap_model(model).load_state_dict(_checkpoint_model_state(checkpoint), strict=True)
    for key, owner in (("optimizer", optimizer), ("scheduler", scheduler)):
        serialized_state = checkpoint.get(key)
        if not isinstance(serialized_state, dict):
            raise ValueError(f"resume checkpoint is missing {key}")
        owner.load_state_dict(serialized_state)
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("resume checkpoint has an invalid epoch")
    best_acc1 = checkpoint.get("best_acc1")
    if not isinstance(best_acc1, (float, int)):
        raise ValueError("resume checkpoint has an invalid best_acc1")
    _restore_resume_rng_state(rng_state, state=state, generators=generators)
    return epoch + 1, float(best_acc1), float(best_ema_acc1) if model_ema is not None else None


def _load_finetune(path: Path, *, model: nn.Module, run: ImageNetRun) -> dict[str, Any]:
    """Transfer compatible encoder weights, without training-state restoration."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("finetune checkpoint must be a mapping")
    contract = validate_checkpoint_contract(checkpoint)
    source_model = contract["model"]
    differences = {"image_size", "drop_path_rate", "drop_path_schedule"}
    if contract["tier"] != run.tier or any(
        source_model.get(key) != value for key, value in run.model.items() if key not in differences
    ) or contract["operator"] != run.operator:
        raise ValueError("finetune checkpoint architecture/operator does not match this model")
    weights = checkpoint.get("selected_weights", "model")
    target = _unwrap_model(model)
    state_dict = dict(_checkpoint_model_state(checkpoint, weights=weights))
    position_key = "encoder.pos_embed"
    source_position = state_dict.get(position_key)
    target_position = target.state_dict().get(position_key)
    if isinstance(source_position, torch.Tensor) and isinstance(target_position, torch.Tensor):
        if source_position.shape != target_position.shape:
            state_dict[position_key] = interpolate_position_embedding(source_position, target_position)
    target.load_state_dict(state_dict, strict=True)
    return {"checkpoint": str(path.resolve()), "source_phase": contract["phase"],
            "source_weights": weights,
            "source_image_size": source_model["image_size"], "source_epoch": checkpoint.get("epoch"),
            "source_contract_digest": checkpoint["contract_digest"]}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run = load_run(args)
    execution = str(run.train.get("execution", "eager"))
    if execution != "eager":
        # Only this launch process inherits the Graph-compatible NCCL policy.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    state = initialize_distributed()
    graph_step = None
    try:
        seed_everything(int(run.train["seed"]), state)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        prepare_operator_backend(run, state.device)
        (
            train_loader,
            val_loader,
            train_dataset,
            batching_plan,
            generators,
            data_contract,
        ) = build_loaders(
            run,
            args.data_root,
            state,
            requested_grad_accum=args.grad_accum,
        )
        model = build_model(run)
        model.to(state.device)
        model_without_ddp = model
        finetune_source = None
        if args.finetune is not None:
            finetune_source = _load_finetune(args.finetune, model=model, run=run)
        # Copy before compile installs bound forward wrappers: deepcopy of those
        # closures could leave the EMA mixer reading the ordinary model weights.
        model_ema = build_model_ema(model_without_ddp, run)
        model = prepare_training_model(model, state, execution)
        if model_ema is not None and state.enabled:
            # DDP has now synchronized the differently seeded rank initializations.
            model_ema.module.load_state_dict(model_without_ddp.state_dict(), strict=True)

        optimizer, resolved_optimizer = build_optimizer(
            model_without_ddp,
            run,
        )
        scheduler = build_scheduler(optimizer, run, batching_plan.updates_per_epoch)
        mixup, criterion = build_mixup_and_loss(run)
        criterion.to(state.device)
        start_epoch = 0
        best_acc1 = float("-inf")
        best_ema_acc1 = float("-inf") if model_ema is not None else None
        if args.resume is not None:
            start_epoch, best_acc1, best_ema_acc1 = _load_resume(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                run=run,
                batching_plan=batching_plan,
                data_contract=data_contract,
                state=state,
                generators=generators,
                model_ema=model_ema,
            )

        output = _prepare_output(
            args.output,
            args,
            run,
            state,
            batching_plan,
            data_contract,
        )
        parameter_count = sum(
            parameter.numel() for parameter in model_without_ddp.parameters() if parameter.requires_grad
        )
        if state.is_main:
            _record_runtime_metadata(
                output,
                parameter_count=parameter_count,
                resolved_optimizer=resolved_optimizer,
                run=run,
                batching_plan=batching_plan,
            )
            if finetune_source is not None:
                _append_jsonl(output / "metrics.jsonl", {"event": "finetune_initialization", **finetune_source})
            print(
                json.dumps(
                    {
                        "event": "start",
                        "tier": run.tier,
                        "phase": run.phase,
                        "parameters": parameter_count,
                        "train_samples": data_contract["train"]["samples"],
                        "scheduled_train_samples": batching_plan.samples_per_epoch,
                        "val_samples": data_contract["validation"]["samples"],
                        "steps_per_epoch": batching_plan.microbatches_per_epoch,
                        "optimizer_steps_per_epoch": batching_plan.updates_per_epoch,
                        "physical_batch_size": batching_plan.physical_batch_size,
                        "effective_batch": batching_plan.effective_batch_size,
                        "grad_accum": batching_plan.grad_accum,
                        **({"augmentation_group_size": batching_plan.augmentation_group_size,
                            "augmentation_draws_per_update": batching_plan.effective_batch_size // batching_plan.augmentation_group_size}
                           if batching_plan.augmentation_group_size is not None else {}),
                        "optimizer": resolved_optimizer,
                        "ema": model_ema is not None,
                        "ema_decay": run.train.get("ema_decay") if model_ema is not None else None,
                        "validation_weights": ["model", "model_ema"] if model_ema is not None else ["model"],
                        "world_size": state.world_size,
                    }
                ),
                flush=True,
            )
        if args.eval:
            validation = evaluate_models(model, val_loader, state=state, model_ema=model_ema)
            if state.is_main:
                record = {
                    "event": "evaluation",
                    "checkpoint": str(args.resume),
                    **validation,
                }
                _append_jsonl(output / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            return

        if start_epoch >= int(run.train["epochs"]):
            if state.is_main:
                print(json.dumps({"event": "complete", "already_complete": True}), flush=True)
            return
        graph_step = (
            ImageNetGraphStep(model, criterion, grad_accum=batching_plan.grad_accum)
            if execution != "eager" else None
        )
        for epoch in range(start_epoch, int(run.train["epochs"])):
            epoch_started = time.perf_counter()
            train_loss, train_acc1, train_acc5 = train_epoch(
                model,
                train_loader,
                train_dataset,
                criterion,
                optimizer,
                mixup,
                scheduler,
                epoch=epoch,
                state=state,
                run=run,
                batching_plan=batching_plan,
                print_freq=args.print_freq,
                graph_step=graph_step,
                model_ema=model_ema,
            )
            validation = evaluate_models(model, val_loader, state=state, model_ema=model_ema)
            record = {
                "event": "epoch",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "train_acc5": train_acc5,
                **validation,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - epoch_started,
            }
            is_best = False
            is_best_ema = False
            if state.is_main:
                is_best = validation["val_acc1"] > best_acc1
                if is_best:
                    best_acc1 = validation["val_acc1"]
                if best_ema_acc1 is not None:
                    is_best_ema = validation["ema_val_acc1"] > best_ema_acc1
                    if is_best_ema:
                        best_ema_acc1 = validation["ema_val_acc1"]
            if state.enabled:
                best_payload: list[Any] = [is_best, best_acc1, is_best_ema, best_ema_acc1]
                dist.broadcast_object_list(best_payload, src=0)
                is_best = bool(best_payload[0])
                best_acc1 = float(best_payload[1])
                is_best_ema = bool(best_payload[2])
                best_ema_acc1 = best_payload[3]
            save_last = (epoch + 1) % int(run.train["save_every"]) == 0
            rng_state = (
                _capture_resume_rng_state(state, generators)
                if save_last or is_best or is_best_ema
                else None
            )
            checkpoint_failure: str | None = None
            if state.is_main:
                if save_last or is_best or is_best_ema:
                    try:
                        if rng_state is None:
                            raise RuntimeError("checkpoint RNG state was not collected")
                        checkpoint = _checkpoint(
                            epoch=epoch,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            run=run,
                            batching_plan=batching_plan,
                            data_contract=data_contract,
                            best_acc1=best_acc1,
                            rng=rng_state,
                            model_ema=model_ema,
                            best_ema_acc1=best_ema_acc1,
                        )
                        if save_last:
                            _atomic_torch_save(checkpoint, output / "checkpoint_last.pt")
                        if is_best:
                            _atomic_torch_save(checkpoint, output / "checkpoint_best.pt")
                        if is_best_ema:
                            _atomic_torch_save({**checkpoint, "selected_weights": "model_ema"},
                                               output / "checkpoint_best_ema.pt")
                    except Exception as error:
                        checkpoint_failure = f"{type(error).__name__}: {error}"
            if state.enabled:
                failure_payload: list[str | None] = [checkpoint_failure]
                dist.broadcast_object_list(failure_payload, src=0)
                checkpoint_failure = failure_payload[0]
            if checkpoint_failure is not None:
                raise RuntimeError(f"unable to save ImageNet checkpoint: {checkpoint_failure}")
            if state.is_main:
                record["best_val_acc1"] = best_acc1
                if best_ema_acc1 is not None:
                    record["best_ema_val_acc1"] = best_ema_acc1
                _append_jsonl(output / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            _barrier(state)
        if state.is_main:
            print(json.dumps({"event": "complete", "best_val_acc1": best_acc1,
                              **({"best_ema_val_acc1": best_ema_acc1} if best_ema_acc1 is not None else {})}), flush=True)
    finally:
        if graph_step is not None:
            graph_step.close()
        finalize_distributed(state)


if __name__ == "__main__":
    main()
