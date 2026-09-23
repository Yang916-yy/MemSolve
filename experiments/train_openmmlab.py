"""Train or evaluate the repository's MMDetection and MMSegmentation recipes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_ROOT = ROOT / "runs" / "openmmlab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Ridgon OpenMMLab downstream experiment."
    )
    parser.add_argument("config", type=Path, help="MMDet/MMSeg config path")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="COCO or ADEChallengeData2016 root, depending on the config.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        help="Required ImageNet classification checkpoint for a new downstream run.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=Path,
        const="auto",
        help="Resume a detector/segmentor checkpoint, or auto-resume when omitted.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        metavar="CHECKPOINT",
        help="Evaluate a downstream checkpoint instead of training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the recipe's default seed.",
    )
    parser.add_argument(
        "--auto-scale-lr",
        action="store_true",
        help="Scale the documented global-batch learning rate to the actual batch.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action="append",
        metavar="KEY=VALUE",
        help="MMEngine configuration overrides; may be repeated.",
    )
    parser.add_argument(
        "--launcher",
        choices=("none", "pytorch", "slurm", "mpi"),
        default="none",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def _checkpoint_path(path: Path, *, option: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{option} does not name a file: {resolved}")
    return str(resolved)


def _config_options(raw_options: list[list[str]] | None) -> dict[str, Any]:
    if raw_options is None:
        return {}
    from mmengine.config import DictAction

    merged: dict[str, Any] = {}
    for options in raw_options:
        for option in options:
            if "=" not in option:
                raise SystemExit(f"--cfg-options requires KEY=VALUE, got {option!r}")
            key, value = option.split("=", maxsplit=1)
            merged[key] = DictAction._parse_iterable(value)
    return merged


def _set_data_root(cfg: Any, data_root: Path) -> None:
    root = str(data_root.expanduser().resolve())
    if cfg.default_scope == "mmdet":
        root = root.rstrip("/") + "/"
        for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            cfg[loader_name].dataset.data_root = root
        annotation = root + "annotations/instances_val2017.json"
        cfg.val_evaluator.ann_file = annotation
        cfg.test_evaluator.ann_file = annotation
        return
    if cfg.default_scope == "mmseg":
        for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            cfg[loader_name].dataset.data_root = root
        return
    raise SystemExit(
        "config.default_scope must be 'mmdet' or 'mmseg', "
        f"got {cfg.default_scope!r}"
    )


def _configure_checkpoint(cfg: Any, args: argparse.Namespace) -> bool:
    """Configure state restoration and return whether this is evaluation-only."""

    if args.test is not None:
        if args.resume is not None or args.backbone_checkpoint is not None:
            raise SystemExit("--test cannot be combined with --resume or --backbone-checkpoint")
        cfg.load_from = _checkpoint_path(args.test, option="--test")
        cfg.resume = False
        return True

    if args.resume is not None:
        if args.backbone_checkpoint is not None:
            raise SystemExit("--resume cannot be combined with --backbone-checkpoint")
        cfg.resume = True
        cfg.load_from = (
            None
            if args.resume == "auto"
            else _checkpoint_path(args.resume, option="--resume")
        )
        return False

    if args.backbone_checkpoint is None:
        raise SystemExit(
            "new downstream training requires --backbone-checkpoint; "
            "do not silently train a paper result from scratch"
        )
    cfg.model.backbone.checkpoint = _checkpoint_path(
        args.backbone_checkpoint,
        option="--backbone-checkpoint",
    )
    return False


def _load_cuda_backend(cfg: Any) -> None:
    implementation = cfg.model.backbone.get("implementation")
    if implementation != "cuda":
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("the configured Ridgon CUDA fast path requires a CUDA device")
    device_index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    from ridgon.ball import cuda

    cuda.load(device=device_index)


def _require_openmmlab_runtime(default_scope: str) -> None:
    """Fail before model construction when the required compiled stack is absent."""

    try:
        import integrations.openmmlab as bridge
        if default_scope == "mmdet":
            from mmdet.models.detectors import MaskRCNN  # noqa: F401

            registered = getattr(bridge, "RidgonMaskRCNN", None)
            framework = "MMDetection"
        elif default_scope == "mmseg":
            from mmseg.models.segmentors import EncoderDecoder  # noqa: F401

            registered = getattr(bridge, "RidgonEncoderDecoder", None)
            framework = "MMSegmentation"
        else:
            raise ValueError(f"unexpected OpenMMLab scope {default_scope!r}")
    except ImportError as error:
        raise SystemExit(
            "the selected downstream protocol requires a PyTorch/CUDA-matched "
            "compiled mmcv build and its OpenMMLab package; mmcv-lite is not "
            "sufficient"
        ) from error
    if registered is None:
        raise SystemExit(
            f"{framework} loaded, but integrations.openmmlab did not register "
            "the required mask-aware Ridgon wrapper"
        )


def main() -> None:
    args = parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from mmengine.config import Config
    from mmengine.runner import Runner

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"config does not name a file: {config_path}")
    cfg = Config.fromfile(str(config_path))
    cfg.launcher = args.launcher

    options = _config_options(args.cfg_options)
    if options:
        cfg.merge_from_dict(options)
    if cfg.get("default_scope") not in ("mmdet", "mmseg"):
        raise SystemExit(
            "config.default_scope must be 'mmdet' or 'mmseg', "
            f"got {cfg.get('default_scope')!r}"
        )
    if args.data_root is not None:
        _set_data_root(cfg, args.data_root)
    if args.seed is not None:
        cfg.randomness = dict(cfg.get("randomness", {}), seed=args.seed)
    if args.auto_scale_lr:
        if "auto_scale_lr" not in cfg or "base_batch_size" not in cfg.auto_scale_lr:
            raise SystemExit("the config does not declare auto_scale_lr.base_batch_size")
        cfg.auto_scale_lr.enable = True

    evaluation_only = _configure_checkpoint(cfg, args)
    if args.work_dir is not None:
        cfg.work_dir = str(args.work_dir.expanduser().resolve())
    elif cfg.get("work_dir") is None:
        cfg.work_dir = str(DEFAULT_WORK_ROOT / config_path.stem)

    _require_openmmlab_runtime(str(cfg.default_scope))
    _load_cuda_backend(cfg)
    runner = Runner.from_cfg(cfg)
    if evaluation_only:
        runner.test()
    else:
        runner.train()


if __name__ == "__main__":
    main()
