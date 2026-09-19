"""Controlled short Food-101 ablations using the shared LSSO vision adapter.

Train on the official training split; report the official test curve and the
fixed final epoch, without test-based checkpoint selection or early stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from experiments.imagenet import (
    IMAGENET_MEAN, IMAGENET_STD, _append_jsonl, _atomic_json,
    _atomic_torch_save, _no_weight_decay,
)
from integrations.timm import LSSODeiT3


VARIANTS = ("dynamic", "static", "zero")


def build_model(variant: str, *, seed: int = 0, implementation: str = "reference") -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    torch.manual_seed(seed)
    # timm stores the learned 14 x 14 patch grid flattened in raster order.
    # Every spatial site has its own vector; CLS has no position embedding.
    return LSSODeiT3(
        image_size=224, patch_size=16, num_classes=101,
        embed_dim=384, depth=12, num_heads=6, rank=32,
        core_mode=variant,
        implementation=implementation, drop_path_rate=0.05,
    )


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loaders(root: Path, batch: int, workers: int, seed: int):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(), normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), normalize,
    ])
    train = datasets.Food101(root, split="train", transform=train_transform)
    test = datasets.Food101(root, split="test", transform=test_transform)
    if len(train) != 75750 or len(test) != 25250 or train.classes != test.classes:
        raise RuntimeError("Food-101 split contract mismatch")
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch, num_workers=workers, pin_memory=True,
                  worker_init_fn=seed_worker, persistent_workers=False)
    if workers:
        common["prefetch_factor"] = 2
    return (
        DataLoader(train, shuffle=True, generator=generator, **common),
        DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(seed), **common),
        generator,
    )


def learning_rate(step: int, steps: int, epochs: int, peak: float) -> float:
    warmup = min(3, epochs) * steps
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, (epochs * steps - warmup - 1))
    return 1e-6 + 0.5 * (peak - 1e-6) * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    totals = torch.zeros(4, device="cuda")
    for images, targets in loader:
        images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
        totals += torch.stack((loss * len(targets),
                              (logits.argmax(1) == targets).sum(),
                              logits.topk(5, 1).indices.eq(targets[:, None]).any(1).sum(),
                              targets.new_tensor(len(targets))))
    loss, top1, top5, count = totals.tolist()
    return dict(loss=loss/count, top1=100*top1/count, top5=100*top5/count, samples=int(count))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--implementation", choices=("reference", "cuda"), default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.implementation == "cuda":
        from lsso.ball import cuda
        cuda.load()
    model = build_model(args.variant, seed=args.seed, implementation=args.implementation).cuda()
    # Reset stochastic-depth RNG identically after model construction.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train, test, generator = make_loaders(args.data, args.batch_size, args.workers, args.seed)
    excluded = _no_weight_decay(model)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name in excluded else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.05},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8, fused=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    contract.update(architecture="LSSO-S/16", dim=384, depth=12, heads=6, rank=32,
                    position_encoding="learned-2d-full-grid-14x14-no-cls",
                    precision="bf16-autocast-fp32-core", implementation=args.implementation,
                    pretrained=False, weight_decay=0.05, label_smoothing=0.1,
                    warmup_epochs=min(3, args.epochs), evaluation="official-test-curve-fixed-final-epoch",
                    train_samples=len(train.dataset), test_samples=len(test.dataset),
                    torch=torch.__version__, parameters=sum(p.numel() for p in model.parameters()))
    contract["metadata_sha256"] = {name: hashlib.sha256((args.data / "food-101/meta" / name).read_bytes()).hexdigest()
                                   for name in ("train.json", "test.json", "classes.txt")}
    start = 0
    if args.resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["contract"] != contract:
            raise RuntimeError("resume contract mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        generator.set_state(checkpoint["loader_rng"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
    elif (output / "last.pt").exists() or (output / "metrics.jsonl").exists():
        raise RuntimeError("refusing to overwrite an existing run; use --resume")
    _atomic_json(output / "run.json", contract)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(start, args.epochs):
        model.train()
        begin = time.monotonic()
        totals = torch.zeros(3, device="cuda")
        for step, (images, targets) in enumerate(train):
            lr = learning_rate(epoch * len(train) + step, len(train), args.epochs, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            optimizer.step()
            totals += torch.stack((loss.detach()*len(targets),
                                   (logits.detach().argmax(1) == targets).sum(),
                                   targets.new_tensor(len(targets))))
            if step % 20 == 0 or step + 1 == len(train):
                _atomic_json(output / "status.json", dict(
                    state="training", pid=os.getpid(), epoch=epoch+1, epochs=args.epochs,
                    step=step+1, steps=len(train), loss=loss.item(), grad_norm=norm.item(),
                    seconds=time.monotonic()-begin, updated=time.time()))
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - begin
        _atomic_json(output / "status.json", dict(state="evaluating", epoch=epoch+1, updated=time.time()))
        metrics = evaluate(model, test)
        train_loss, train_correct, count = totals.tolist()
        row = dict(epoch=epoch+1, train_loss=train_loss/count, train_top1=100*train_correct/count,
                   test=metrics, lr=lr, train_seconds=train_seconds,
                   epoch_seconds=time.monotonic()-begin,
                   peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        checkpoint = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          contract=contract, torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state(), loader_rng=generator.get_state(),
                          python_rng=random.getstate(), numpy_rng=np.random.get_state(), metrics=row)
        _atomic_torch_save(checkpoint, output / "last.pt")
        _append_jsonl(output / "metrics.jsonl", row)
        print(json.dumps(row), flush=True)
    final = torch.load(output / "last.pt", map_location="cpu", weights_only=False)["metrics"]
    _atomic_json(output / "final_metrics.json", final)
    _atomic_json(output / "status.json", dict(state="complete", epoch=args.epochs, updated=time.time()))


if __name__ == "__main__":
    main()
