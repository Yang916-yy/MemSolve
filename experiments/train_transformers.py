"""Shared PyTorch sequence runner for GenomicBenchmarks and Long Range Arena.

The runner intentionally owns no Ridgon mathematics.  It uses the public
``Ridgon`` module for the current operator and a matched PyTorch MHA block for
baselines, while data/tokenization contracts live in ``sequence_data.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 is supported by the package.
    import tomli as tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ridgon import Ridgon, RidgonConfig
from ridgon.ball import cuda as cuda_backend

from experiments.sequence_data import (
    DatasetBundle,
    GENOMIC_BENCHMARKS,
    LRA_TASKS,
    PATHFINDER_TASKS,
    PATHX_RESOLUTION,
    collate_token_pairs,
    collate_tokens,
    collate_values,
    limit_dataset,
    make_loader,
    prepare_genomic_benchmarks,
    prepare_lra,
    is_immutable_revision,
    validate_formal_source_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_CACHE_ROOT = ROOT / "data" / "sequence_cache"


@dataclass(frozen=True)
class TaskDefaults:
    max_length: int | None
    epochs: int
    batch_size: int
    eval_batch_size: int
    grad_accum: int
    lr: float
    warmup_ratio: float
    patience: int
    dim: int
    depth: int
    heads: int
    rank: int
    mlp_ratio: float
    dropout: float
    weight_decay: float
    pooling: Literal["mean", "meanmax"]


PATHFINDER_DEFAULTS = TaskDefaults(
    max_length=None, epochs=200, batch_size=64, eval_batch_size=256, grad_accum=2,
    lr=2e-4, warmup_ratio=0.05, patience=10, dim=256, depth=6, heads=4,
    rank=32, mlp_ratio=2.0, dropout=0.0, weight_decay=0.01,
    pooling="meanmax",
)


# Path-X is deliberately not a formal recipe. Its inherited defaults provide a
# runnable local probe only and are not part of the four-task formal panel.
PATHX_DEFAULTS = TaskDefaults(
    max_length=None, epochs=200, batch_size=12, eval_batch_size=12, grad_accum=10,
    lr=2e-4, warmup_ratio=0.05, patience=20, dim=128, depth=6, heads=8,
    rank=32, mlp_ratio=4.0, dropout=0.1, weight_decay=0.01,
    pooling="mean",
)


LRA_DEFAULTS = {
    "listops": TaskDefaults(
        max_length=2048, epochs=40, batch_size=25, eval_batch_size=25, grad_accum=2,
        lr=5e-4, warmup_ratio=0.05, patience=8, dim=128, depth=6, heads=8,
        rank=32, mlp_ratio=4.0, dropout=0.1, weight_decay=0.01,
        pooling="mean",
    ),
    "text": TaskDefaults(
        max_length=4096, epochs=32, batch_size=16, eval_batch_size=16, grad_accum=2,
        lr=5e-4, warmup_ratio=0.05, patience=6, dim=256, depth=6, heads=8,
        rank=32, mlp_ratio=4.0, dropout=0.1, weight_decay=0.01,
        pooling="mean",
    ),
    "retrieval": TaskDefaults(
        max_length=4000, epochs=20, batch_size=16, eval_batch_size=16, grad_accum=4,
        lr=5e-4, warmup_ratio=0.05, patience=5, dim=128, depth=6, heads=8,
        rank=32, mlp_ratio=4.0, dropout=0.1, weight_decay=0.01,
        pooling="mean",
    ),
    "pathfinder": PATHFINDER_DEFAULTS,
    "pathx": PATHX_DEFAULTS,
}
GENOMIC_DEFAULTS = TaskDefaults(
    max_length=0, epochs=40, batch_size=128, eval_batch_size=256, grad_accum=1,
    lr=3e-4, warmup_ratio=0.05, patience=8, dim=128, depth=4, heads=4,
    rank=32, mlp_ratio=4.0, dropout=0.1, weight_decay=0.01,
    pooling="mean",
)
DEFAULT_PATHFINDER_RESOLUTION = 32
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 2026
FROZEN_DNA_MIXERS = frozenset({"nystromformer", "rebased"})
MixerName = Literal[
    "mha",
    "mha_rope",
    "ridgon",
    "linear_transformer",
    "performer",
    "nystromformer",
    "cosformer",
    "rebased",
]


class MaskedMultiheadAttention(nn.Module):
    """Bidirectional MHA baseline with the same valid-token contract as Ridgon."""

    def __init__(self, dim: int, num_heads: int, *, bias: bool, rope: bool = False) -> None:
        super().__init__()
        self.rope = rope
        if rope and (dim // num_heads) % 2:
            raise ValueError("RoPE requires even head dimension")
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=0.0,
            bias=bias,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, N] for MHA")
        result = torch.zeros_like(x)
        nonempty = valid_mask.any(dim=1)
        if bool(nonempty.any()):
            active_x = x[nonempty]
            active_mask = valid_mask[nonempty]
            if self.rope:
                active = self._rotary_attention(active_x, active_mask)
            else:
                active, _weights = self.attention(
                    active_x,
                    active_x,
                    active_x,
                    key_padding_mask=~active_mask,
                    need_weights=False,
                )
            result[nonempty] = active.to(dtype=result.dtype)
        return torch.where(valid_mask[:, :, None], result, torch.zeros_like(result))

    def _rotary_attention(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """RoFormer adjacent-pair Q/K rotation, base 10000; V is unchanged.

        See https://github.com/ZhuiyiTechnology/roformer . The existing MHA
        projections preserve the baseline's parameter names and initialization.
        Learned absolute embeddings remain in the shared encoder.
        """
        batch, length, dim = x.shape
        heads = self.attention.num_heads
        width = dim // heads
        qkv = F.linear(x, self.attention.in_proj_weight, self.attention.in_proj_bias)
        q, k, v = (t.reshape(batch, length, heads, width).transpose(1, 2)
                   for t in qkv.chunk(3, dim=-1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            frequency = 10000.0 ** (-torch.arange(0, width, 2, device=x.device, dtype=dtype) / width)
            angles = torch.arange(length, device=x.device, dtype=dtype)[:, None] * frequency
            cosine, sine = angles.cos(), angles.sin()
            def rotate(t: torch.Tensor) -> torch.Tensor:
                even, odd = t.to(dtype)[..., 0::2], t.to(dtype)[..., 1::2]
                return torch.stack((even*cosine - odd*sine, even*sine + odd*cosine), dim=-1).flatten(-2).to(t.dtype)
            q, k = rotate(q), rotate(k)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid_mask[:, None, None, :], dropout_p=0.0,
        )
        return self.attention.out_proj(attended.transpose(1, 2).reshape(batch, length, dim))


def _orthogonal_random_matrix(rows: int, columns: int) -> torch.Tensor:
    """Gaussian orthogonal features used by the official Performer implementation."""
    blocks = []
    for _ in range(rows // columns):
        q, _ = torch.linalg.qr(torch.randn(columns, columns), mode="reduced")
        blocks.append(q.T)
    remainder = rows % columns
    if remainder:
        q, _ = torch.linalg.qr(torch.randn(columns, columns), mode="reduced")
        blocks.append(q.T[:remainder])
    matrix = torch.cat(blocks, dim=0)
    return matrix * torch.randn(rows, columns).norm(dim=1, keepdim=True)


class MaskedPerformerAttention(nn.Module):
    """Bidirectional FAVOR+ attention adapted from performer-pytorch 1.1.4."""

    def __init__(self, dim: int, num_heads: int, *, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_features = max(1, int(self.head_dim * math.log(self.head_dim)))
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.register_buffer(
            "projection_matrix",
            _orthogonal_random_matrix(self.num_features, self.head_dim),
        )

    def _feature_map(self, x: torch.Tensor, *, query: bool) -> torch.Tensor:
        projection = self.projection_matrix.to(dtype=x.dtype)
        data_normalizer = self.head_dim ** -0.25
        ratio = self.num_features ** -0.5
        projected = torch.einsum("bhnd,md->bhnm", x * data_normalizer, projection)
        diagonal = (x.square().sum(dim=-1, keepdim=True) * data_normalizer**2) / 2
        if query:
            maximum = projected.amax(dim=-1, keepdim=True).detach()
        else:
            maximum = projected.amax(dim=(-2, -1), keepdim=True).detach()
        return ratio * (torch.exp(projected - diagonal - maximum) + 1e-4)

    def _forward_core(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        reshape = lambda value: value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q = self._feature_map(reshape(self.q_proj(x)), query=True)
        k = self._feature_map(reshape(self.k_proj(x)), query=False)
        v = reshape(self.v_proj(x))
        key_mask = valid_mask[:, None, :, None]
        k = k * key_mask
        v = v * key_mask
        denominator = torch.einsum("bhnm,bhm->bhn", q, k.sum(dim=2)).clamp_min(1e-6)
        context = torch.einsum("bhnm,bhnd->bhmd", k, v)
        output = torch.einsum("bhnm,bhmd,bhn->bhnd", q, context, denominator.reciprocal())
        output = output.transpose(1, 2).reshape(batch, length, -1)
        output = self.out_proj(output)
        return torch.where(valid_mask[:, :, None], output, torch.zeros_like(output))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, N] for Performer")
        context = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with context:
            output = self._forward_core(x.float() if x.device.type == "cuda" else x, valid_mask)
        return output.to(x.dtype)


class MaskedLinearTransformerAttention(nn.Module):
    """Bidirectional ELU+1 linear attention from the Linear Transformer."""

    def __init__(self, dim: int, num_heads: int, *, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _forward_core(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        reshape = lambda value: value.reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = F.elu(reshape(self.q_proj(x))) + 1
        k = F.elu(reshape(self.k_proj(x))) + 1
        v = reshape(self.v_proj(x))
        key_mask = valid_mask[:, None, :, None]
        k = k * key_mask
        v = v * key_mask
        context = torch.einsum("bhnm,bhnd->bhmd", k, v)
        denominator = torch.einsum("bhnm,bhm->bhn", q, k.sum(dim=2)).clamp_min(1e-6)
        output = torch.einsum(
            "bhnm,bhmd,bhn->bhnd", q, context, denominator.reciprocal()
        )
        output = output.transpose(1, 2).reshape(batch, length, -1)
        output = self.out_proj(output)
        return torch.where(valid_mask[:, :, None], output, torch.zeros_like(output))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, N] for Linear Transformer")
        context = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with context:
            output = self._forward_core(x.float() if x.device.type == "cuda" else x, valid_mask)
        return output.to(x.dtype)


class MaskedReBasedAttention(nn.Module):
    """Bidirectional ReBased attention adapted from FLA's reference path."""

    def __init__(
        self, dim: int, num_heads: int, *, bias: bool, feature_dim: int = 16
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = dim // num_heads
        self.eps = 1e-5
        self.q_proj = nn.Linear(dim, num_heads * feature_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * feature_dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))
        self.register_buffer(
            "_off_diagonal_indices",
            torch.triu_indices(feature_dim, feature_dim, offset=1),
            persistent=False,
        )

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x, (self.feature_dim,), self.gamma, self.beta)
        outer = torch.einsum("...i,...j->...ij", x, x)
        diagonal = outer.diagonal(dim1=-2, dim2=-1)
        off_diagonal = outer[
            ..., self._off_diagonal_indices[0], self._off_diagonal_indices[1]
        ]
        return torch.cat(
            (
                diagonal * self.feature_dim**-0.5,
                off_diagonal * (2 / self.feature_dim) ** 0.5,
            ),
            dim=-1,
        )

    def _forward_core(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        reshape_qk = lambda value: value.reshape(
            batch, length, self.num_heads, self.feature_dim
        ).transpose(1, 2)
        reshape_v = lambda value: value.reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = reshape_qk(self.q_proj(x))
        k = reshape_qk(self.k_proj(x))
        v = reshape_v(self.v_proj(x))
        q, k = self._feature_map(q), self._feature_map(k)
        key_mask = valid_mask[:, None, :, None]
        k = k * key_mask
        v = v * key_mask
        context = torch.einsum("bhnm,bhnd->bhmd", k, v)
        denominator = torch.einsum("bhnm,bhm->bhn", q, k.sum(dim=2)) + self.eps
        output = torch.einsum(
            "bhnm,bhmd,bhn->bhnd", q, context, denominator.reciprocal()
        )
        output = output.transpose(1, 2).reshape(batch, length, dim)
        output = self.out_proj(output)
        return torch.where(valid_mask[:, :, None], output, torch.zeros_like(output))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, N] for ReBased")
        context = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with context:
            output = self._forward_core(x.float() if x.device.type == "cuda" else x, valid_mask)
        return output.to(x.dtype)


def _iterative_pinv(matrix: torch.Tensor, iterations: int = 6) -> torch.Tensor:
    absolute = matrix.abs()
    scale = absolute.sum(dim=-1).amax() * absolute.sum(dim=-2).amax()
    estimate = matrix.transpose(-1, -2) / scale.clamp_min(torch.finfo(matrix.dtype).tiny)
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    for _ in range(iterations):
        product = matrix @ estimate
        estimate = 0.25 * estimate @ (
            13 * identity - product @ (15 * identity - product @ (7 * identity - product))
        )
    return estimate


class MaskedNystromAttention(nn.Module):
    """Nyströmformer attention with the official approximation and no local residual conv."""

    def __init__(self, dim: int, num_heads: int, *, bias: bool, landmarks: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.landmarks = landmarks
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _fixed_length(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        heads, head_dim = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        reshape = lambda value: value.reshape(batch, length, heads, head_dim).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        # The iterative inverse is unstable in FP16. Projections still use AMP;
        # only the approximation and its reductions use the public FP32 path.
        q, k, v = (value.float() for value in (q, k, v))
        q = q * self.scale
        landmarks = min(self.landmarks, length)
        padded_length = math.ceil(length / landmarks) * landmarks
        padding = padded_length - length
        if padding:
            q, k, v = (F.pad(value, (0, 0, padding, 0)) for value in (q, k, v))
        chunk = padded_length // landmarks
        landmark_counts = torch.ones(padded_length, device=x.device, dtype=q.dtype)
        if padding:
            landmark_counts[:padding] = 0
        landmark_counts = landmark_counts.reshape(landmarks, chunk).sum(dim=1).clamp_min(1)
        landmark_counts = landmark_counts[None, None, :, None]
        q_landmarks = q.reshape(batch, heads, landmarks, chunk, head_dim).sum(dim=3) / landmark_counts
        k_landmarks = k.reshape(batch, heads, landmarks, chunk, head_dim).sum(dim=3) / landmark_counts
        sim1 = torch.einsum("bhnd,bhmd->bhnm", q, k_landmarks).softmax(dim=-1)
        sim2 = torch.einsum("bhnd,bhmd->bhnm", q_landmarks, k_landmarks).softmax(dim=-1)
        sim3 = torch.einsum("bhnd,bhmd->bhnm", q_landmarks, k).softmax(dim=-1)
        output = (sim1 @ _iterative_pinv(sim2)) @ (sim3 @ v)
        return output[:, :, -length:].transpose(1, 2).reshape(batch, length, dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, N] for Nyströmformer")
        output = torch.zeros_like(x)
        lengths = valid_mask.sum(dim=1)
        for length in lengths.unique(sorted=True):
            size = int(length)
            if size == 0:
                continue
            selected = lengths == length
            value = self.out_proj(self._fixed_length(x[selected, :size]).to(x.dtype))
            output[selected, :size] = value.to(output.dtype)
        return torch.where(valid_mask[:, :, None], output, torch.zeros_like(output))


class MaskedCosformerAttention(nn.Module):
    """Bidirectional cosFormer attention adapted from OpenNLPLab/cosFormer."""

    def __init__(self, dim: int, num_heads: int, *, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def _forward_fp32(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        reshape = lambda value: value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.relu(reshape(self.q_proj(x)))
        k = F.relu(reshape(self.k_proj(x)))
        v = reshape(self.v_proj(x))
        positions = torch.arange(1, length + 1, device=x.device, dtype=q.dtype)[None, None, :, None]
        lengths = valid_mask.sum(dim=1).clamp_min(1).to(q.dtype)[:, None, None, None]
        angles = (math.pi / 2) * positions / lengths
        q = torch.cat((q * angles.sin(), q * angles.cos()), dim=-1)
        k = torch.cat((k * angles.sin(), k * angles.cos()), dim=-1)
        key_mask = valid_mask[:, None, :, None]
        k = k * key_mask
        v = v * key_mask
        context = torch.einsum("bhnm,bhnd->bhmd", k, v)
        denominator = torch.einsum("bhnm,bhm->bhn", q, k.sum(dim=2)).clamp_min(1e-6)
        output = torch.einsum("bhnm,bhmd,bhn->bhnd", q, context, denominator.reciprocal())
        output = output.transpose(1, 2).reshape(batch, length, -1)
        output = self.out_proj(output)
        return torch.where(valid_mask[:, :, None], output, torch.zeros_like(output))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, N] for cosFormer")
        # The official 1e-6 denominator is finite in FP32 but can overflow the
        # FP16 backward when a ReLU feature head is initially all zero.
        context = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with context:
            output = self._forward_fp32(x.float(), valid_mask)
        output = output.to(x.dtype)
        return torch.where(valid_mask[:, :, None], output, torch.zeros_like(output))


class SequenceBlock(nn.Module):
    """A pre-norm residual encoder block shared by both mixer families."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        rank: int,
        mixer: MixerName,
        implementation: Literal["reference", "cuda"],
        mlp_ratio: float,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.mixer_kind = mixer
        self.implementation = implementation
        self.norm1 = nn.LayerNorm(dim)
        if mixer == "ridgon":
            self.mixer: nn.Module = Ridgon(
                RidgonConfig(
                    dim=dim,
                    num_heads=num_heads,
                    rank=rank,
                    bias=bias,
                )
            )
        elif mixer in ("mha", "mha_rope"):
            self.mixer = MaskedMultiheadAttention(dim, num_heads, bias=bias, rope=mixer == "mha_rope")
        elif mixer == "performer":
            self.mixer = MaskedPerformerAttention(dim, num_heads, bias=bias)
        elif mixer == "linear_transformer":
            self.mixer = MaskedLinearTransformerAttention(dim, num_heads, bias=bias)
        elif mixer == "nystromformer":
            self.mixer = MaskedNystromAttention(dim, num_heads, bias=bias)
        elif mixer == "cosformer":
            self.mixer = MaskedCosformerAttention(dim, num_heads, bias=bias)
        elif mixer == "rebased":
            self.mixer = MaskedReBasedAttention(dim, num_heads, bias=bias)
        else:
            raise ValueError(f"unsupported mixer {mixer!r}")
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(round(dim * mlp_ratio))
        if hidden_dim <= 0:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=bias),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        if self.mixer_kind == "ridgon":
            if normalized.device.type == "cuda" and torch.is_autocast_enabled():
                amp_dtype = torch.get_autocast_dtype("cuda")
                if self.implementation == "cuda" and amp_dtype not in (torch.float16, torch.bfloat16):
                    raise TypeError("the CUDA Ridgon sequence path supports FP16 or BF16 AMP")
                normalized = normalized.to(dtype=amp_dtype)
            mixed = self.mixer(  # type: ignore[operator]
                normalized,
                valid_mask=valid_mask,
                implementation=self.implementation,
            )
        else:
            mixed = self.mixer(normalized, valid_mask)
        x = x + self.residual_dropout(mixed)
        x = torch.where(valid_mask[:, :, None], x, torch.zeros_like(x))
        x = x + self.residual_dropout(self.mlp(self.norm2(x)))
        return torch.where(valid_mask[:, :, None], x, torch.zeros_like(x))


class SequenceEncoder(nn.Module):
    """Learned absolute coordinate features plus current mixer blocks.

    Learned position embeddings are shared by MHA and Ridgon. Ridgon itself
    has no position-dependent feature rotation.
    """

    def __init__(
        self,
        *,
        input_kind: Literal["tokens", "values"],
        vocab_size: int | None,
        pad_token_id: int | None,
        max_length: int,
        dim: int,
        depth: int,
        num_heads: int,
        rank: int,
        mixer: MixerName,
        implementation: Literal["reference", "cuda"],
        mlp_ratio: float,
        dropout: float,
        bias: bool,
        grid_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.input_kind = input_kind
        self.max_length = max_length
        self._cuda_ridgon = mixer == "ridgon" and implementation == "cuda"
        if input_kind == "tokens":
            if vocab_size is None or pad_token_id is None:
                raise ValueError("token models require vocab_size and pad_token_id")
            self.input_embedding: nn.Module = nn.Embedding(
                vocab_size, dim, padding_idx=pad_token_id
            )
        else:
            self.input_embedding = nn.Linear(1, dim, bias=bias)
        self.grid_shape = grid_shape
        if grid_shape is None:
            self.position_embedding: nn.Embedding | None = nn.Embedding(max_length, dim)
            self.row_embedding: nn.Embedding | None = None
            self.column_embedding: nn.Embedding | None = None
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            rows, columns = grid_shape
            if input_kind != "values":
                raise ValueError("factorized grid positions require value inputs")
            if rows <= 0 or columns <= 0 or rows * columns != max_length:
                raise ValueError("factorized grid shape must exactly cover max_length")
            self.position_embedding = None
            self.row_embedding = nn.Embedding(rows, dim)
            self.column_embedding = nn.Embedding(columns, dim)
            position_std = 0.02 / math.sqrt(2.0)
            nn.init.normal_(self.row_embedding.weight, mean=0.0, std=position_std)
            nn.init.normal_(self.column_embedding.weight, mean=0.0, std=position_std)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SequenceBlock(
                dim=dim,
                num_heads=num_heads,
                rank=rank,
                mixer=mixer,
                implementation=implementation,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def _position_features(self, *, length: int, device: torch.device) -> torch.Tensor:
        if self.grid_shape is None:
            assert self.position_embedding is not None
            positions = torch.arange(length, device=device)
            return self.position_embedding(positions)
        rows, columns = self.grid_shape
        if length != rows * columns:
            raise ValueError(
                f"factorized grid positions require a full {rows}x{columns} grid, got length={length}"
            )
        assert self.row_embedding is not None
        assert self.column_embedding is not None
        row_positions = torch.arange(rows, device=device)
        column_positions = torch.arange(columns, device=device)
        return (
            self.row_embedding(row_positions)[:, None, :]
            + self.column_embedding(column_positions)[None, :, :]
        ).reshape(length, -1)

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if inputs.ndim not in (2, 3):
            raise ValueError("sequence inputs must be [B, N] tokens or [B, N, 1] values")
        if valid_mask.shape != inputs.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, N]")
        length = inputs.shape[1]
        if length <= 0 or length > self.max_length:
            raise ValueError(
                f"input length {length} must be in [1, {self.max_length}]"
            )
        if self.input_kind == "tokens":
            if inputs.ndim != 2 or inputs.dtype != torch.long:
                raise TypeError("token inputs must be torch.long [B, N]")
        elif inputs.ndim != 3 or inputs.shape[-1] != 1 or not inputs.is_floating_point():
            raise TypeError("value inputs must be floating [B, N, 1]")
        x = self.input_embedding(inputs) + self._position_features(
            length=length, device=inputs.device
        )[None]
        if inputs.device.type == "cuda" and torch.is_autocast_enabled():
            amp_dtype = torch.get_autocast_dtype("cuda")
            if self._cuda_ridgon and amp_dtype not in (torch.float16, torch.bfloat16):
                raise TypeError("the CUDA Ridgon sequence path supports FP16 or BF16 AMP")
            x = x.to(dtype=amp_dtype)
        x = self.embedding_dropout(x)
        x = torch.where(valid_mask[:, :, None], x, torch.zeros_like(x))
        for block in self.blocks:
            x = block(x, valid_mask)
        return torch.where(valid_mask[:, :, None], self.norm(x), torch.zeros_like(x))

    @staticmethod
    def pooled(encoded: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        counts = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        return (encoded * valid_mask[:, :, None]).sum(dim=1) / counts


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        encoder: SequenceEncoder,
        num_classes: int,
        *,
        pooling: Literal["mean", "meanmax"] = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "meanmax"}:
            raise ValueError(f"unsupported pooling {pooling!r}")
        self.encoder = encoder
        self.pooling = pooling
        dim = encoder.norm.normalized_shape[0]
        self.meanmax_projection = (
            nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
            if pooling == "meanmax"
            else None
        )
        self.readout_projection = (
            nn.Linear(dim, dim, bias=False) if pooling == "meanmax" else None
        )
        self.classifier = nn.Linear(dim, num_classes)

    def _pool(self, encoded: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mean = self.encoder.pooled(encoded, valid_mask)
        if self.pooling == "mean":
            return mean
        maximum = encoded.masked_fill(
            ~valid_mask[:, :, None], torch.finfo(encoded.dtype).min
        ).amax(dim=1)
        maximum = torch.where(
            valid_mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        assert self.meanmax_projection is not None
        assert self.readout_projection is not None
        features = self.meanmax_projection(torch.cat((mean, maximum), dim=-1))
        return self.readout_projection(features)

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._pool(self.encoder(inputs, valid_mask), valid_mask))


class SequencePairClassifier(nn.Module):
    def __init__(self, encoder: SequenceEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        dim = encoder.norm.normalized_shape[0]
        self.classifier = nn.Linear(4 * dim, num_classes)

    def forward(
        self,
        first: torch.Tensor,
        first_mask: torch.Tensor,
        second: torch.Tensor,
        second_mask: torch.Tensor,
    ) -> torch.Tensor:
        first_state = self.encoder.pooled(self.encoder(first, first_mask), first_mask)
        second_state = self.encoder.pooled(self.encoder(second, second_mask), second_mask)
        features = torch.cat(
            (first_state, second_state, (first_state - second_state).abs(), first_state * second_state),
            dim=-1,
        )
        return self.classifier(features)


@dataclass(frozen=True)
class TrainingConfig:
    output: Path
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    min_lr_ratio: float
    grad_accum: int
    grad_clip: float
    patience: int
    early_stop_min_epochs: int
    early_stop_accuracy_delta: float
    early_stop_loss_relative_delta: float
    seed: int
    resume: bool
    validation_only: bool
    max_train_batches: int
    max_eval_batches: int
    amp: bool
    # A non-formal diagnostic cap. The scheduler horizon remains ``epochs``.
    pilot_epochs: int = 0
    pacing_file: Path | None = None


_SEQUENCE_CHECKPOINT_FORMAT = 2


def _is_better_validation_checkpoint(
    accuracy: float,
    loss: float,
    best_accuracy: float,
    best_loss: float,
) -> bool:
    """Select exactly by validation rank, independently of early stopping."""

    return accuracy > best_accuracy or (
        accuracy == best_accuracy and loss < best_loss
    )


def _makes_early_stop_progress(
    accuracy: float,
    loss: float,
    best_accuracy: float,
    best_loss: float,
    *,
    accuracy_delta: float,
    loss_relative_delta: float,
) -> tuple[bool, bool]:
    """Return tolerance-qualified progress for early stopping only."""

    return (
        accuracy > best_accuracy + accuracy_delta,
        loss < best_loss * (1.0 - loss_relative_delta),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    initial, _unknown = bootstrap.parse_known_args(argv)
    parser = _make_parser()
    if initial.config is not None:
        settings = _load_toml_defaults(initial.config)
        known = {action.dest for action in parser._actions}
        unknown = sorted(set(settings) - known)
        if unknown:
            raise ValueError(f"unknown keys in {initial.config}: {', '.join(unknown)}")
        parser.set_defaults(**settings)
    return parser.parse_args(argv)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--suite", choices=("genomic", "lra"))
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--data-revision")
    parser.add_argument("--pathfinder-resolution", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument(
        "--mixer",
        choices=(
            "mha",
            "mha_rope",
            "ridgon",
            "linear_transformer",
            "performer",
            "nystromformer",
            "cosformer",
            "rebased",
        ),
        default="ridgon",
    )
    parser.add_argument("--implementation", choices=("reference", "cuda"), default="cuda")


    parser.add_argument("--rank", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--mlp-ratio", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--pooling", choices=("mean", "meanmax"))
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pilot-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--early-stop-min-epochs", type=int, default=-1)
    parser.add_argument("--early-stop-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-loss-relative-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pacing-file", type=Path, help="Runtime JSON with delay_seconds/pause for cooperative scheduling.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def _load_toml_defaults(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    values: dict[str, Any] = {}
    for section, entries in loaded.items():
        if not isinstance(entries, dict):
            values[section] = entries
            continue
        for key, value in entries.items():
            if key in values:
                raise ValueError(f"duplicate config key {key} in {path}")
            values[key] = value
    return values


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.suite is None or args.task is None:
        raise ValueError("--suite and --task are required directly or through --config")
    if args.suite == "genomic":
        if args.task not in GENOMIC_BENCHMARKS:
            raise ValueError(f"unsupported GenomicBenchmarks task {args.task!r}")
        defaults = GENOMIC_DEFAULTS
        if args.task == "dummy_mouse_enhancers_ensembl":
            defaults = TaskDefaults(
                **{
                    **asdict(GENOMIC_DEFAULTS),
                    "batch_size": 64,
                    "grad_accum": 2,
                }
            )
    else:
        if args.task not in LRA_TASKS:
            raise ValueError(f"unsupported LRA task {args.task!r}")
        defaults = LRA_DEFAULTS[args.task]
    pathfinder = args.suite == "lra" and args.task in PATHFINDER_TASKS
    pathx = args.suite == "lra" and args.task == "pathx"
    uses_validation_split = args.suite == "genomic" or (
        args.suite == "lra" and args.task == "text"
    )
    for name in (
        "rank",
        "dim",
        "depth",
        "heads",
        "mlp_ratio",
        "dropout",
        "pooling",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "grad_accum",
        "lr",
        "weight_decay",
        "warmup_ratio",
        "patience",
    ):
        if getattr(args, name) is None:
            setattr(args, name, getattr(defaults, name))
    if pathfinder:
        if args.max_length is not None:
            if pathx:
                raise ValueError(
                    "--max-length is not supported for LRA Path-X; "
                    "its resolution is fixed at 128"
                )
            raise ValueError(
                "--max-length is not supported for LRA Pathfinder; "
                "use --pathfinder-resolution"
            )
        if pathx:
            if args.pathfinder_resolution is not None:
                raise ValueError(
                    "--pathfinder-resolution is not supported for LRA Path-X; "
                    "it is fixed at 128"
                )
            args.pathfinder_resolution = PATHX_RESOLUTION
        elif args.pathfinder_resolution is None:
            args.pathfinder_resolution = DEFAULT_PATHFINDER_RESOLUTION
    else:
        if args.pathfinder_resolution is not None:
            raise ValueError(
                "--pathfinder-resolution is only supported for generic LRA Pathfinder"
            )
        if args.max_length is None:
            if defaults.max_length is None:
                raise RuntimeError("task has no default max_length")
            args.max_length = defaults.max_length
        if args.suite == "lra" and args.max_length == 0:
            if defaults.max_length is None:
                raise RuntimeError("task has no default max_length")
            args.max_length = defaults.max_length
    if uses_validation_split:
        if args.validation_fraction is None:
            args.validation_fraction = DEFAULT_VALIDATION_FRACTION
        if args.split_seed is None:
            args.split_seed = DEFAULT_SPLIT_SEED
    else:
        if args.validation_fraction is not None:
            if pathfinder:
                task_name = "Path-X" if pathx else "Pathfinder"
                raise ValueError(
                    f"--validation-fraction is not supported for LRA {task_name}; "
                    "it uses the official hard 80/10/10 split"
                )
            raise ValueError(
                "--validation-fraction is only supported for GenomicBenchmarks and LRA Text"
            )
        if args.split_seed is not None:
            if pathfinder:
                task_name = "Path-X" if pathx else "Pathfinder"
                raise ValueError(
                    f"--split-seed is not supported for LRA {task_name}; "
                    "it uses the official hard 80/10/10 split"
                )
            raise ValueError(
                "--split-seed is only supported for GenomicBenchmarks and LRA Text"
            )
    if args.early_stop_min_epochs < 0:
        args.early_stop_min_epochs = math.ceil(0.75 * args.epochs)
    if args.output is None:
        args.output = ROOT / "runs" / "sequence" / f"{args.suite}-{args.task}-{args.mixer}-s{args.seed}"
    args.data_root = Path(args.data_root).resolve()
    args.cache_root = Path(args.cache_root).resolve()
    args.output = Path(args.output).resolve()
    _validate_resolved_args(args)
    return args


def _validate_resolved_args(args: argparse.Namespace) -> None:
    if args.mixer in {
        "linear_transformer", "performer", "nystromformer", "cosformer", "rebased"
    } and args.suite != "genomic":
        raise ValueError(f"{args.mixer} is a GenomicBenchmarks-only baseline")
    if args.formal and args.mixer in FROZEN_DNA_MIXERS:
        raise ValueError(
            f"--formal rejects frozen baseline {args.mixer}; its current "
            "implementation is retained only as a numerical reference"
        )
    for name in ("rank", "dim", "depth", "heads", "epochs", "batch_size", "eval_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.dim % args.heads:
        raise ValueError("dim must be divisible by heads")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if args.mlp_ratio <= 0.0:
        raise ValueError("mlp_ratio must be positive")
    if args.validation_fraction is not None and not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if args.pathfinder_resolution is not None and args.pathfinder_resolution <= 0:
        raise ValueError("pathfinder_resolution must be positive")
    if args.grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if args.pilot_epochs < 0:
        raise ValueError("pilot_epochs must be non-negative")
    if args.pilot_epochs > args.epochs:
        raise ValueError("pilot_epochs must not exceed epochs")
    if args.pilot_epochs:
        if args.formal:
            raise ValueError("--formal rejects --pilot-epochs")
        if not args.validation_only:
            raise ValueError("--pilot-epochs requires --validation-only")
    if args.grad_clip < 0.0:
        raise ValueError("grad_clip must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if args.patience < 0 or args.early_stop_min_epochs < 0:
        raise ValueError("early-stop values must be non-negative")
    if args.early_stop_accuracy_delta < 0.0:
        raise ValueError("early_stop_accuracy_delta must be non-negative")
    if not 0.0 <= args.early_stop_loss_relative_delta < 1.0:
        raise ValueError("early_stop_loss_relative_delta must be in [0, 1)")
    if args.implementation == "cuda" and args.mixer == "ridgon":
        if args.rank not in (16, 32, 48, 64):
            raise ValueError("CUDA supports Ridgon rank in {16, 32, 48, 64}")


def _validate_formal_data_source(args: argparse.Namespace) -> None:
    if args.allow_download and not is_immutable_revision(args.data_revision):
        raise ValueError(
            "--formal with --allow-download requires an immutable full-SHA "
            "--data-revision"
        )


def build_bundle(args: argparse.Namespace) -> DatasetBundle:
    if args.suite == "genomic":
        return prepare_genomic_benchmarks(
            args.task,
            data_root=args.data_root,
            max_length=args.max_length,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            allow_download=args.allow_download,
            revision=args.data_revision,
            formal=args.formal,
        )
    return prepare_lra(
        args.task,
        data_root=args.data_root,
        cache_root=args.cache_root,
        max_length=args.max_length,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        # ``128`` is the resolved public identity for Path-X.  The data
        # loader owns that fixed resolution and uses ``None`` to distinguish
        # it from a caller attempting to override generic Pathfinder.
        pathfinder_resolution=None if args.task == "pathx" else args.pathfinder_resolution,
        allow_download=args.allow_download,
        revision=args.data_revision,
        formal=args.formal,
    )


def build_model(args: argparse.Namespace, bundle: DatasetBundle) -> nn.Module:
    implementation: Literal["reference", "cuda"] = args.implementation
    grid_shape = None
    if args.suite == "lra" and args.task == "pathfinder":
        resolution = bundle.metadata.get("resolution")
        if not isinstance(resolution, int):
            raise ValueError("factorized grid positions require integer Pathfinder resolution")
        grid_shape = (resolution, resolution)
    encoder = SequenceEncoder(
        input_kind=bundle.input_kind,
        vocab_size=bundle.vocab_size,
        pad_token_id=bundle.pad_token_id,
        max_length=bundle.max_length,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        rank=args.rank,
        mixer=args.mixer,
        implementation=implementation,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        bias=args.bias,
        grid_shape=grid_shape,
    )
    if bundle.paired:
        if args.pooling != "mean":
            raise ValueError("--pooling=meanmax is not supported for paired tasks")
        return SequencePairClassifier(encoder, bundle.num_classes)
    return SequenceClassifier(encoder, bundle.num_classes, pooling=args.pooling)


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _build_collate(bundle: DatasetBundle):
    if bundle.input_kind == "values":
        return functools.partial(collate_values, value_masking=bundle.value_masking)
    if bundle.paired:
        assert bundle.pad_token_id is not None
        return functools.partial(collate_token_pairs, pad_token_id=bundle.pad_token_id)
    assert bundle.pad_token_id is not None
    return functools.partial(collate_tokens, pad_token_id=bundle.pad_token_id)


def _build_loaders(
    args: argparse.Namespace, bundle: DatasetBundle, device: torch.device
):
    train = limit_dataset(bundle.train, args.max_train_samples, args.seed + 11)
    validation = limit_dataset(bundle.validation, args.max_eval_samples, args.seed + 12)
    test = limit_dataset(bundle.test, args.max_eval_samples, args.seed + 13)
    collate = _build_collate(bundle)
    return (
        make_loader(
            train,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            collate_fn=collate,
            train=True,
            seed=args.seed,
        ),
        make_loader(
            validation,
            batch_size=args.eval_batch_size,
            workers=args.eval_workers,
            device=device,
            collate_fn=collate,
            train=False,
            seed=args.seed,
        ),
        make_loader(
            test,
            batch_size=args.eval_batch_size,
            workers=args.eval_workers,
            device=device,
            collate_fn=collate,
            train=False,
            seed=args.seed,
        ),
        {"train": len(train), "validation": len(validation), "test": len(test)},
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    if "first" in batch:
        return model(batch["first"], batch["first_mask"], batch["second"], batch["second_mask"])
    return model(batch["inputs"], batch["mask"])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_revision() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _runtime_metadata(device: torch.device, *, cuda_enabled: bool) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    if cuda_enabled:
        metadata["ridgon_cuda_contract"] = cuda_backend._CUDA_CONTRACT_VERSION
    return metadata


def _config_digest(payload: dict[str, Any]) -> str:
    # ``--resume`` changes how a run is entered, not the run it represents.
    # Keep it in the on-disk manifest for auditability, but exclude it from the
    # checkpoint identity so an explicit continuation can match its own run.
    identity = dict(payload)
    resolved_arguments = identity.get("resolved_arguments")
    if isinstance(resolved_arguments, dict):
        arguments = dict(resolved_arguments)
        arguments.pop("resume", None)
        identity["resolved_arguments"] = arguments
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_save(state: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _loader_state(loader: Any) -> dict[str, Any]:
    sampler = loader.batch_sampler
    if hasattr(sampler, "state_dict"):
        return {"batch_sampler": sampler.state_dict()}
    return {}


def _restore_loader_state(loader: Any, state: dict[str, Any] | None) -> None:
    if not state:
        return
    sampler = loader.batch_sampler
    if "batch_sampler" in state and hasattr(sampler, "load_state_dict"):
        sampler.load_state_dict(state["batch_sampler"])


def _scheduler_lambda(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _classification_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> dict[str, float]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1
    total = int(confusion.sum())
    accuracy = float(confusion.diag().sum()) / max(total, 1)
    f1_values = []
    for label in range(num_classes):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum()) - true_positive
        false_negative = int(confusion[label, :].sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {"accuracy": accuracy, "macro_f1": sum(f1_values) / num_classes}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    num_classes: int,
    max_batches: int,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    losses, predictions, targets = [], [], []
    for batch_index, raw_batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        with _autocast_context(device, amp):
            logits = forward_batch(model, batch)
            loss = F.cross_entropy(logits, batch["labels"])
        losses.append(loss.detach().float() * batch["labels"].numel())
        predictions.append(logits.argmax(dim=-1).detach().cpu())
        targets.append(batch["labels"].detach().cpu())
    if not predictions:
        raise RuntimeError("evaluation loader produced no batches")
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    metrics = _classification_metrics(all_predictions, all_targets, num_classes)
    metrics["loss"] = float(torch.stack(losses).sum()) / all_targets.numel()
    metrics["examples"] = float(all_targets.numel())
    return metrics


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _build_run_payload(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    sizes: dict[str, int],
    model: nn.Module,
    device: torch.device,
    *,
    cuda_enabled: bool,
) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    inactive_data_arguments = _inactive_data_argument_names(args)
    if args.suite == "lra" and args.task == "pathfinder":
        position_encoding = "factorized-grid-absolute"
    else:
        position_encoding = "learned-absolute"
    return {
        "schema": 1,
        "resolved_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in inactive_data_arguments
        },
        "dataset": bundle.metadata,
        "data_contract": {
            "input_kind": bundle.input_kind,
            "max_length": bundle.max_length,
            "vocab_size": bundle.vocab_size,
            "paired": bundle.paired,
            "value_masking": bundle.value_masking
            if bundle.input_kind == "values"
            else None,
        },
        "split_sizes": sizes,
        "model": {
            "mixer": args.mixer,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "rank": args.rank,
            "mlp_ratio": args.mlp_ratio,
            "dropout": args.dropout,
            "bias": args.bias,
            "pooling": args.pooling,
            "implementation": (
                args.implementation
                if args.mixer == "ridgon"
                else {
                    "mha": "torch-sdpa",
                    "mha_rope": "torch-sdpa-roformer-qk-adjacent-pairs-base10000",
                    "linear_transformer": "linear-transformer-elu-plus-one-adapted-fp32-core",
                    "performer": "performer-pytorch-1.1.4-adapted-fp32-core",
                    "nystromformer": "nystrom-attention-0.0.14-adapted-no-conv",
                    "cosformer": "opennlplab-cosformer-official-adapted",
                    "rebased": "fla-rebased-noncausal-reference-adapted",
                }[args.mixer]
            ),
            "position_encoding": position_encoding,
            **({"attention_rotation": "rope-qk", "rope_base": 10000.0,
                "rope_dimensions": args.dim // args.heads} if args.mixer == "mha_rope" else {}),
            "position_initialization": "normal-0.02"
            if not (args.suite == "lra" and args.task == "pathfinder")
            else "factorized-normal-0.02",
            "parameters": parameters,
        },
        "runtime": _runtime_metadata(device, cuda_enabled=cuda_enabled),
        "source_revision": _source_revision(),
        "selection": {
            "checkpoint": "highest validation accuracy, then lower validation loss",
            "test_evaluation": "once after checkpoint selection",
            "validation_only": args.validation_only,
        },
    }


def _inactive_data_argument_names(args: argparse.Namespace) -> frozenset[str]:
    pathfinder = args.suite == "lra" and args.task in PATHFINDER_TASKS
    uses_validation_split = args.suite == "genomic" or (
        args.suite == "lra" and args.task == "text"
    )
    inactive = set()
    if not pathfinder:
        inactive.add("pathfinder_resolution")
    else:
        inactive.add("max_length")
    if not uses_validation_split:
        inactive.update(("validation_fraction", "split_seed"))
    return frozenset(inactive)


def train(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    test_loader: Any,
    *,
    num_classes: int,
    config: TrainingConfig,
    run_payload: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    digest = _config_digest(run_payload)
    run_record = {**run_payload, "config_digest": digest, "training": asdict(config)}
    last_path = output / "last.pt"
    resume_state: dict[str, Any] | None = None
    if last_path.exists():
        if not config.resume:
            raise FileExistsError(
                f"{last_path} exists; choose a new --output or pass --resume explicitly"
            )
        resume_state = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume_state.get("config_digest") != digest:
            raise RuntimeError("checkpoint configuration does not match this run")
        if resume_state.get("format_version") != _SEQUENCE_CHECKPOINT_FORMAT:
            raise RuntimeError(
                "checkpoint does not match the current selection-state contract"
            )
        if resume_state.get("completed"):
            target = "validation" if config.validation_only else "test"
            result = resume_state.get("final_result")
            if resume_state.get("final_target") != target or not isinstance(result, dict):
                raise RuntimeError("completed checkpoint has an invalid final-result contract")
            print(json.dumps({"resumed": result}, sort_keys=True), flush=True)
            return result
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    batches_per_epoch = len(train_loader)
    if config.max_train_batches:
        batches_per_epoch = min(batches_per_epoch, config.max_train_batches)
    total_steps = max(1, math.ceil(batches_per_epoch / config.grad_accum) * config.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduler_lambda(
            step,
            round(total_steps * config.warmup_ratio),
            total_steps,
            config.min_lr_ratio,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    start_epoch = 0
    best_accuracy, best_loss = float("-inf"), float("inf")
    early_stop_best_accuracy, early_stop_best_loss = float("-inf"), float("inf")
    stale_epochs = 0
    state = resume_state
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state.get("scaler", {}))
        start_epoch = int(resume_state["epoch"]) + 1
        best_accuracy = float(resume_state["best_accuracy"])
        best_loss = float(resume_state["best_loss"])
        early_stop_best_accuracy = float(resume_state["early_stop_best_accuracy"])
        early_stop_best_loss = float(resume_state["early_stop_best_loss"])
        stale_epochs = int(resume_state["stale_epochs"])
        _restore_rng_state(resume_state.get("rng"))
        _restore_loader_state(train_loader, resume_state.get("train_loader"))
    (output / "config.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    metrics_path = output / "metrics.jsonl"
    run_epochs = min(config.epochs, config.pilot_epochs or config.epochs)
    for epoch in range(start_epoch, run_epochs):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        train_examples, accumulated_examples = 0, 0
        for batch_index, raw_batch in enumerate(train_loader):
            if config.max_train_batches and batch_index >= config.max_train_batches:
                break
            batch = _move_batch(raw_batch, device)
            batch_size = batch["labels"].numel()
            with _autocast_context(device, config.amp):
                logits = forward_batch(model, batch)
                loss = F.cross_entropy(logits, batch["labels"])
            scaler.scale(loss * batch_size).backward()
            train_loss_sum.add_(loss.detach().to(dtype=torch.float64), alpha=batch_size)
            train_examples += batch_size
            accumulated_examples += batch_size
            last_batch = batch_index + 1 == batches_per_epoch
            if (batch_index + 1) % config.grad_accum == 0 or last_batch:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_examples)
                if config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler backs off on overflow, including fused AdamW's
                # kernel-level skips. Unchanged/growing scale means an update.
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_examples = 0
                if config.pacing_file is not None:
                    torch.cuda.synchronize(device) if device.type == "cuda" else None
                    while True:
                        pacing = json.loads(config.pacing_file.read_text())
                        delay = float(pacing.get("delay_seconds", 0.0))
                        if not math.isfinite(delay) or delay < 0:
                            raise ValueError("pacing delay_seconds must be finite and nonnegative")
                        if not pacing.get("pause", False):
                            if delay:
                                time.sleep(delay)
                            break
                        time.sleep(1.0)
            if batch_index % 25 == 0:
                progress = {"state": "training", "epoch": epoch + 1,
                            "step": batch_index + 1, "steps_per_epoch": batches_per_epoch,
                            "elapsed_seconds": time.perf_counter() - started,
                            "updated": time.time()}
                temp = output / "status.tmp"
                temp.write_text(json.dumps(progress))
                temp.replace(output / "status.json")
        (output / "status.json").write_text(json.dumps(
            {"state": "validation", "epoch": epoch + 1, "updated": time.time()}))
        validation = evaluate(
            model,
            validation_loader,
            device=device,
            num_classes=num_classes,
            max_batches=config.max_eval_batches,
            amp=config.amp,
        )
        checkpoint_improved = _is_better_validation_checkpoint(
            validation["accuracy"],
            validation["loss"],
            best_accuracy,
            best_loss,
        )
        if checkpoint_improved:
            best_accuracy, best_loss = validation["accuracy"], validation["loss"]
        accuracy_progress, loss_progress = _makes_early_stop_progress(
            validation["accuracy"],
            validation["loss"],
            early_stop_best_accuracy,
            early_stop_best_loss,
            accuracy_delta=config.early_stop_accuracy_delta,
            loss_relative_delta=config.early_stop_loss_relative_delta,
        )
        if accuracy_progress:
            early_stop_best_accuracy = validation["accuracy"]
        if loss_progress:
            early_stop_best_loss = validation["loss"]
        stale_epochs = 0 if accuracy_progress or loss_progress else stale_epochs + 1
        metrics = {
            "epoch": epoch,
            "train_loss": (train_loss_sum / max(train_examples, 1)).item(),
            "train_examples": train_examples,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "seconds": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "early_stop_stale_epochs": stale_epochs,
            "early_stop_eligible": epoch + 1 >= config.early_stop_min_epochs,
            "checkpoint_selected": checkpoint_improved,
            "early_stop_accuracy_progress": accuracy_progress,
            "early_stop_loss_progress": loss_progress,
            "peak_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
        }
        print(json.dumps(metrics, sort_keys=True), flush=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        state = {
            "format_version": _SEQUENCE_CHECKPOINT_FORMAT,
            "config_digest": digest,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "best_loss": best_loss,
            "early_stop_best_accuracy": early_stop_best_accuracy,
            "early_stop_best_loss": early_stop_best_loss,
            "stale_epochs": stale_epochs,
            "rng": _rng_state(),
            "train_loader": _loader_state(train_loader),
        }
        if checkpoint_improved or not (output / "best.pt").exists():
            _atomic_save(state, output / "best.pt")
        _atomic_save(state, last_path)
        if (
            config.patience
            and epoch + 1 >= config.early_stop_min_epochs
            and stale_epochs >= config.patience
        ):
            break
    best_state = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"])
    target_loader = validation_loader if config.validation_only else test_loader
    target_name = "validation" if config.validation_only else "test"
    final_metrics = evaluate(
        model,
        target_loader,
        device=device,
        num_classes=num_classes,
        max_batches=config.max_eval_batches,
        amp=config.amp,
    )
    result = {
        **final_metrics,
        "selected_epoch": int(best_state["epoch"]),
        "best_validation_accuracy": float(best_state["best_accuracy"]),
        "target": target_name,
    }
    (output / f"{target_name}_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    if state is None:
        raise RuntimeError("training ended without a checkpoint state")
    state["completed"] = True
    state["final_target"] = target_name
    state["final_result"] = result
    _atomic_save(state, last_path)
    (output / "status.json").write_text(json.dumps(
        {"state": "completed", "updated": time.time(), "result": result}))
    print(json.dumps({target_name: result}, sort_keys=True), flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    args = resolve_args(parse_args(argv))
    _seed_all(args.seed)
    device = _choose_device(args.device)
    if args.mixer == "ridgon" and args.implementation == "cuda":
        if device.type != "cuda":
            raise RuntimeError("Ridgon CUDA implementation requires a CUDA device")
        cuda_backend.load(device=device)
    if args.formal:
        revision = _source_revision()
        if revision["git_commit"] is None or revision["git_dirty"]:
            raise RuntimeError("--formal requires a clean committed source revision")
        if args.max_train_samples or args.max_eval_samples or args.max_train_batches or args.max_eval_batches:
            raise ValueError("--formal rejects pilot sample and batch limits")
        _validate_formal_data_source(args)
    bundle = build_bundle(args)
    if args.formal:
        validate_formal_source_provenance(bundle)
    model = build_model(args, bundle)
    train_loader, validation_loader, test_loader, sizes = _build_loaders(args, bundle, device)
    run_payload = _build_run_payload(
        args,
        bundle,
        sizes,
        model,
        device,
        cuda_enabled=args.mixer == "ridgon" and args.implementation == "cuda",
    )
    if args.prepare_only:
        print(json.dumps(run_payload, indent=2, sort_keys=True, default=str), flush=True)
        return
    config = TrainingConfig(
        output=args.output,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        grad_accum=args.grad_accum,
        grad_clip=args.grad_clip,
        patience=args.patience,
        early_stop_min_epochs=args.early_stop_min_epochs,
        early_stop_accuracy_delta=args.early_stop_accuracy_delta,
        early_stop_loss_relative_delta=args.early_stop_loss_relative_delta,
        seed=args.seed,
        resume=args.resume,
        validation_only=args.validation_only,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        amp=not args.no_amp and device.type == "cuda",
        pilot_epochs=args.pilot_epochs,
        pacing_file=args.pacing_file,
    )
    train(
        model,
        train_loader,
        validation_loader,
        test_loader,
        num_classes=bundle.num_classes,
        config=config,
        run_payload=run_payload,
        device=device,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "MaskedMultiheadAttention",
    "SequenceClassifier",
    "SequenceEncoder",
    "SequencePairClassifier",
    "build_bundle",
    "build_model",
    "forward_batch",
    "main",
    "parse_args",
    "resolve_args",
]
