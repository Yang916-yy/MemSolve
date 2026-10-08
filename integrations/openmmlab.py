"""Thin OpenMMLab adapters for the ViT³-style MemSolve vision backbone.

This module owns framework registration and padded-image plumbing only. The
operator remains entirely in :mod:`memsolve.ball`, while the ViT³-style encoder is
owned by :mod:`integrations.timm`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional

from experiments.imagenet import (
    interpolate_position_embedding,
    validate_checkpoint_contract,
)
from integrations.timm import (
    MemSolveViT,
    VisionTokenLayout,
    vision_default_rank,
    vision_spec,
)


def _validate_out_indices(
    out_indices: Sequence[int],
    *,
    depth: int,
) -> tuple[int, ...]:
    indices = tuple(out_indices)
    if len(indices) != 4:
        raise ValueError("out_indices must contain exactly four block indices")
    if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
        raise ValueError("out_indices must be strictly increasing")
    if any(not isinstance(index, int) or index < 0 or index >= depth for index in indices):
        raise ValueError(f"out_indices must lie in [0, {depth}), got {indices}")
    return indices


def _variant_out_indices(variant: str) -> tuple[int, ...]:
    spec = vision_spec(variant)
    if spec.depth == 12:
        return (3, 5, 7, 11)
    return (7, 11, 15, 23)


def _checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("ImageNet checkpoint must be a mapping")
    return payload


def _checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor | object]:
    state = payload.get("model")
    if not isinstance(state, dict):
        raise TypeError("ImageNet checkpoint must contain a model state dictionary")
    return {
        (key.removeprefix("module.")): value
        for key, value in state.items()
    }


class MemSolveViTBackbone(MemSolveViT):
    """ViT³-style MemSolve features for Mask R-CNN/FPN and UperNet.

    Four intermediate plain-ViT maps are converted to strides 4, 8, 16, and
    32 with the same simple feature pyramid used by the XCiT downstream code.
    A pixel-validity mask becomes the MemSolve token mask before every global mix.
    """

    def __init__(
        self,
        *,
        variant: str,
        image_size: int = 224,
        rank: int | None = None,
        qk_conv_kernel_size: int = 3,
        output_gate_rank: int = 32,
        out_indices: Sequence[int] | None = None,
        bias: bool = True,
        implementation: str = "cuda",
        checkpoint: str | Path | None = None,
    ) -> None:
        spec = vision_spec(variant)
        if implementation not in ("reference", "cuda"):
            raise ValueError(
                "implementation must be 'reference' or 'cuda', "
                f"got {implementation!r}"
            )
        resolved_rank = vision_default_rank(variant) if rank is None else rank

        self.variant = variant
        self.out_indices = _validate_out_indices(
            _variant_out_indices(variant) if out_indices is None else out_indices,
            depth=spec.depth,
        )
        self.out_channels = (spec.embed_dim,) * 4
        self._pretrained_model_contract = {
            "patch_size": 16,
            "num_classes": 1000,
            "mlp_ratio": 4.0,
            "architecture": "vit3_qkconv_lpe2d_mean_swiglu_v5", "ffn": "swiglu",
            "layer_scale": False,
            "class_token": False,
            "pooling": "token_ln_mean",
            "position_encoding": "learned_2d",
            "drop_path_schedule": "linear",
            "norm_eps": 1e-6,
            "embed_dim": spec.embed_dim,
            "depth": spec.depth,
            "num_heads": spec.num_heads,
            "rank": resolved_rank,
            "drop_path_rate": spec.drop_path_rate,
        }
        self._pretrained_operator_contract = {
            "bias": bias,
            "implementation": implementation,
            "qk_conv_kernel_size": qk_conv_kernel_size,
            "output_gate_rank": output_gate_rank,
        }
        super().__init__(
            image_size=image_size,
            patch_size=16,
            num_classes=0,
            embed_dim=spec.embed_dim,
            depth=spec.depth,
            num_heads=spec.num_heads,
            rank=resolved_rank,
            qk_conv_kernel_size=qk_conv_kernel_size,
            output_gate_rank=output_gate_rank,
            mlp_ratio=4.0,
            bias=bias,
            implementation=implementation,
            drop_path_rate=spec.drop_path_rate,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(spec.embed_dim, spec.embed_dim, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(spec.embed_dim, spec.embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.ConvTranspose2d(
            spec.embed_dim,
            spec.embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        if checkpoint is not None:
            self.load_pretrained(checkpoint)

    def load_pretrained(self, checkpoint: str | Path) -> None:
        """Load an ImageNet runner checkpoint, dropping only its classifier."""

        payload = _checkpoint_payload(checkpoint)
        self._validate_pretrained_contract(validate_checkpoint_contract(payload))
        state = _checkpoint_state(payload)
        source_position = state.get("encoder.pos_embed")
        target_position = self.state_dict().get("encoder.pos_embed")
        if isinstance(source_position, torch.Tensor) and isinstance(
            target_position, torch.Tensor
        ):
            if source_position.shape != target_position.shape:
                state["encoder.pos_embed"] = interpolate_position_embedding(
                    source_position,
                    target_position,
                )
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if key not in {"encoder.head.weight", "encoder.head.bias"}
        ]
        if unexpected:
            joined = ", ".join(unexpected)
            raise RuntimeError(f"unexpected pretrained checkpoint keys: {joined}")
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(("fpn1.", "fpn2."))
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"pretrained checkpoint is missing backbone keys: {joined}")

    def _validate_pretrained_contract(self, contract: dict[str, Any]) -> None:
        if contract["tier"] != self.variant:
            raise ValueError(
                "ImageNet checkpoint tier does not match the requested downstream "
                f"backbone: {contract['tier']!r} != {self.variant!r}"
            )
        if contract["phase"] not in ("pretrain", "finetune_224"):
            raise ValueError("downstream initialization requires an ImageNet training checkpoint")
        model = contract["model"]
        operator = contract["operator"]
        if not isinstance(model, dict) or not isinstance(operator, dict):
            raise ValueError("ImageNet checkpoint has an invalid model or operator contract")
        if not isinstance(model.get("image_size"), int) or model["image_size"] <= 0:
            raise ValueError("ImageNet checkpoint model contract has an invalid image_size")
        for key, expected in self._pretrained_model_contract.items():
            if model.get(key) != expected:
                raise ValueError(
                    "ImageNet checkpoint model contract does not match the downstream "
                    f"backbone at {key}: {model.get(key)!r} != {expected!r}"
                )
        if operator != self._pretrained_operator_contract:
            raise ValueError(
                "ImageNet checkpoint operator contract does not match the downstream backbone"
            )

    def _patch_valid_mask(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}")
        batch, _channels, height, width = x.shape
        grid_size = self.encoder.patch_embed.dynamic_feat_size((height, width))
        grid_height, grid_width = int(grid_size[0]), int(grid_size[1])
        if valid_mask is None:
            return torch.ones(
                batch,
                grid_height,
                grid_width,
                dtype=torch.bool,
                device=x.device,
            )
        if valid_mask.shape != (batch, height, width):
            raise ValueError(
                "valid_mask must have shape "
                f"{(batch, height, width)}, got {tuple(valid_mask.shape)}"
            )
        if valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must use torch.bool")
        patch_size = self.encoder.patch_embed.patch_size
        pooled = functional.max_pool2d(
            valid_mask.to(device=x.device, dtype=torch.float32).unsqueeze(1),
            kernel_size=patch_size,
            stride=patch_size,
            ceil_mode=True,
        ).squeeze(1).to(dtype=torch.bool)
        if pooled.shape[-2:] != (grid_height, grid_width):
            raise RuntimeError(
                "patch validity grid does not match the ViT³-style patch embedding"
            )
        return pooled

    @staticmethod
    def _token_layout(
        patch_mask: torch.Tensor,
        *,
        needs_mask: bool,
    ) -> VisionTokenLayout:
        return VisionTokenLayout(patch_mask.flatten(1) if needs_mask else None)

    @staticmethod
    def _scale_mask(mask: torch.Tensor, index: int) -> torch.Tensor:
        if index == 0:
            return functional.interpolate(
                mask.unsqueeze(1).to(dtype=torch.float32),
                scale_factor=4.0,
                mode="nearest",
            ).to(dtype=torch.bool)
        if index == 1:
            return functional.interpolate(
                mask.unsqueeze(1).to(dtype=torch.float32),
                scale_factor=2.0,
                mode="nearest",
            ).to(dtype=torch.bool)
        if index == 2:
            return mask.unsqueeze(1)
        if index == 3:
            return functional.max_pool2d(
                mask.unsqueeze(1).to(dtype=torch.float32),
                kernel_size=2,
                stride=2,
            ).to(dtype=torch.bool)
        raise ValueError(f"invalid feature index {index}")

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_mask = self._patch_valid_mask(x, valid_mask)
        safe_x = x
        if valid_mask is not None:
            pixel_mask = valid_mask.to(device=x.device)
            safe_x = torch.where(pixel_mask[:, None], x, torch.zeros_like(x))
        layout = self._token_layout(
            patch_mask,
            needs_mask=valid_mask is not None,
        )
        maps = self.forward_intermediates(
            safe_x,
            indices=self.out_indices,
            valid_mask=layout.valid_mask,
            norm=False,
        )
        operations = (self.fpn1, self.fpn2, self.fpn3, self.fpn4)
        outputs: list[torch.Tensor] = []
        for index, (feature, operation) in enumerate(zip(maps, operations, strict=True)):
            input_mask = patch_mask.unsqueeze(1)
            safe_feature = torch.where(
                input_mask,
                feature,
                torch.zeros_like(feature),
            )
            output_mask = self._scale_mask(patch_mask, index)
            if index == 3:
                safe_feature = torch.where(
                    input_mask,
                    feature,
                    torch.full_like(feature, -torch.inf),
                )
            output = operation(safe_feature)
            if output.shape[-2:] != output_mask.shape[-2:]:
                raise RuntimeError("pyramid feature and validity mask shapes differ")
            outputs.append(
                torch.where(
                    output_mask,
                    output,
                    torch.zeros_like(output),
                )
            )
        return tuple(outputs)  # type: ignore[return-value]


def _pixel_valid_mask(
    inputs: torch.Tensor,
    data_samples: Sequence[Any] | None,
) -> torch.Tensor | None:
    if data_samples is None:
        return None
    batch, _channels, height, width = inputs.shape
    if len(data_samples) != batch:
        raise ValueError("data_samples length must match the image batch size")
    mask = torch.zeros(batch, height, width, dtype=torch.bool, device=inputs.device)
    for index, sample in enumerate(data_samples):
        metainfo = sample.metainfo
        shape = metainfo.get("img_shape", (height, width))
        image_height, image_width = int(shape[0]), int(shape[1])
        if not 0 < image_height <= height or not 0 < image_width <= width:
            raise ValueError(
                "img_shape must be positive and fit in the padded batch input, "
                f"got {(image_height, image_width)} for {(height, width)}"
            )
        mask[index, :image_height, :image_width] = True
    return mask


class _MaskAwareMixin:
    """Thread OpenMMLab's per-image shape metadata into the token mixer."""

    _memsolve_valid_mask: torch.Tensor | None = None

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Sequence[Any] | None = None,
        mode: str = "tensor",
    ) -> Any:
        self._memsolve_valid_mask = _pixel_valid_mask(inputs, data_samples)
        try:
            return super().forward(inputs, data_samples, mode)
        finally:
            self._memsolve_valid_mask = None

    def extract_feat(self, batch_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(batch_inputs, valid_mask=self._memsolve_valid_mask)
        if self.with_neck:
            features = self.neck(features)
        return features


def _register_openmmlab() -> None:
    """Register only when the framework and its compiled operators are present."""

    try:
        from mmdet.models.detectors import MaskRCNN
        from mmdet.registry import MODELS as MMDET_MODELS
    except ImportError:
        MaskRCNN = None  # type: ignore[assignment,misc]
        MMDET_MODELS = None  # type: ignore[assignment]
    try:
        from mmseg.models.segmentors import EncoderDecoder
        from mmseg.registry import MODELS as MMSEG_MODELS
    except ImportError:
        EncoderDecoder = None  # type: ignore[assignment,misc]
        MMSEG_MODELS = None  # type: ignore[assignment]

    if MMDET_MODELS is not None:
        MMDET_MODELS.register_module(module=MemSolveViTBackbone, force=True)
        if MaskRCNN is not None:
            class MemSolveMaskRCNN(_MaskAwareMixin, MaskRCNN):
                pass

            MMDET_MODELS.register_module(module=MemSolveMaskRCNN, force=True)
            globals()["MemSolveMaskRCNN"] = MemSolveMaskRCNN

    if MMSEG_MODELS is not None:
        MMSEG_MODELS.register_module(module=MemSolveViTBackbone, force=True)
        if EncoderDecoder is not None:
            class MemSolveEncoderDecoder(_MaskAwareMixin, EncoderDecoder):
                pass

            MMSEG_MODELS.register_module(module=MemSolveEncoderDecoder, force=True)
            globals()["MemSolveEncoderDecoder"] = MemSolveEncoderDecoder


_register_openmmlab()


__all__ = ["MemSolveViTBackbone"]
