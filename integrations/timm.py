"""MemSolve-ViT models and timm registration, backed by the shared operator.

Import this module before calling timm.create_model("memsolve_vit_small").
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Sequence, cast

import torch
import torch.nn as nn

from memsolve import MemSolve, MemSolveConfig


_Implementation = Literal["reference", "cuda"]


@dataclass(frozen=True)
class VisionSpec:
    """Geometry and stochastic depth for a MemSolve vision scale."""

    embed_dim: int
    depth: int
    num_heads: int
    drop_path_rate: float


_VISION_SPECS: dict[str, VisionSpec] = {
    "tiny": VisionSpec(192, 12, 6, 0.0),
    "small": VisionSpec(384, 12, 6, 0.1),
    "base": VisionSpec(768, 12, 12, 0.4),
    "large": VisionSpec(1024, 24, 16, 0.45),
}
_VISION_DEFAULT_RANKS: dict[str, int] = {
    "tiny": 16,
    "small": 32,
    "base": 48,
    "large": 64,
}


@dataclass(frozen=True)
class VisionTokenLayout:
    """Token validity for the vision mixer."""

    valid_mask: torch.Tensor | None
    grid_size: tuple[int, int] | None = None


def vision_spec(variant: str) -> VisionSpec:
    """Return the MemSolve vision geometry; large is for optional feature adapters."""

    try:
        return _VISION_SPECS[variant]
    except KeyError as error:
        choices = ", ".join(_VISION_SPECS)
        raise ValueError(
            f"variant must be one of {{{choices}}}, got {variant!r}"
        ) from error


def vision_default_rank(variant: str) -> int:
    """Return the default MemSolve rank for a vision scale."""

    vision_spec(variant)
    return _VISION_DEFAULT_RANKS[variant]


def _validate_implementation(implementation: str) -> _Implementation:
    if implementation not in ("reference", "cuda"):
        raise ValueError(
            "implementation must be 'reference' or 'cuda', "
            f"got {implementation!r}"
        )
    return cast(_Implementation, implementation)


def _require_timm_attention_mask_api(vision_transformer: type[nn.Module]) -> None:
    """Reject timm releases predating the token-layout forwarding contract."""

    required = ("forward", "forward_intermediates")
    missing = [
        name
        for name in required
        if "attn_mask" not in inspect.signature(
            getattr(vision_transformer, name)
        ).parameters
    ]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "MemSolve vision adapters require timm>=1.0.16 with attn_mask support "
            f"in VisionTransformer.{joined}"
        )


class _TimmMemSolveMixer(nn.Module):
    """Adapt MemSolve to timm's attention call signature without owning math."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rank: int,
        qk_conv_kernel_size: int = 3,
        output_gate_rank: int = 32,
        implementation: _Implementation,
        qkv_bias: bool,
        qk_norm: bool,
        scale_norm: bool,
        proj_bias: bool,
        attn_drop: float,
        proj_drop: float,
        norm_layer: Any,
        depth: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        del norm_layer, depth
        self.implementation = _validate_implementation(implementation)
        if qk_norm or scale_norm:
            raise ValueError("MemSolve does not implement qk_norm or attention scale_norm")
        if attn_drop != 0.0 or proj_drop != 0.0:
            raise ValueError("mixer dropout belongs outside the MemSolve operator")
        if qkv_bias != proj_bias:
            raise ValueError("MemSolve requires matching input and output bias settings")

        self.mixer = MemSolve(
            MemSolveConfig(
                dim=dim,
                num_heads=num_heads,
                rank=rank,
                bias=qkv_bias,
                qk_conv_dim=2,
                qk_conv_kernel_size=qk_conv_kernel_size,
                output_gate_rank=output_gate_rank,
            )
        )
        if device is not None or dtype is not None:
            self.mixer.to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | VisionTokenLayout | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise ValueError(
                "the vision adapter accepts only unmasked bidirectional input"
            )
        if attn_mask is None:
            layout = None
        elif isinstance(attn_mask, VisionTokenLayout):
            layout = attn_mask
        else:
            raise ValueError(
                "the vision adapter accepts only its token validity layout, "
                "not a generic attention mask"
            )
        valid_mask = None if layout is None else layout.valid_mask
        # Both implementations use the same AMP input/output boundary. timm's
        # FP32 residual stream must not make the oracle skip the output cast.
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        return self.mixer(
            x, valid_mask=valid_mask, implementation=self.implementation,
            spatial_shape=None if layout is None else layout.grid_size,
        )


class MemSolveViT(nn.Module):
    """MemSolve-ViT: Q/K conv, learned 2D patch positions and SwiGLU.

    mlp_ratio specifies the equivalent two-projection MLP weight budget.
    SwiGLU uses two-thirds of that hidden width, rounded up to 16 channels.
    """

    def __init__(
        self, *, image_size: int, embed_dim: int, depth: int, num_heads: int,
        rank: int, patch_size: int = 16, num_classes: int = 1000,
        mlp_ratio: float = 4.0, bias: bool = True,
        implementation: _Implementation = "reference", drop_path_rate: float = 0.0,
        drop_path_schedule: Literal["linear", "constant"] = "linear",
        norm_eps: float = 1e-6, dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        qk_conv_kernel_size: int = 3, output_gate_rank: int = 32,
    ) -> None:
        super().__init__()
        resolved = _validate_implementation(implementation)
        if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
            raise ValueError("depth must be a positive integer")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if drop_path_schedule not in ("linear", "constant"):
            raise ValueError("drop_path_schedule must be 'linear' or 'constant'")
        if norm_eps != 1e-6:
            raise ValueError("the vision scaffold requires norm_eps = 1e-6")
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if not mlp_ratio > 0:
            raise ValueError("mlp_ratio must be positive")
        from timm.layers import GluMlp
        from timm.models.vision_transformer import Block, VisionTransformer
        _require_timm_attention_mask_api(VisionTransformer)
        default_grid = (image_size // patch_size,) * 2

        class ConfiguredMixer(_TimmMemSolveMixer):
            def __init__(self, dim: int, num_heads: int, **kwargs: Any) -> None:
                super().__init__(dim, num_heads, rank=rank,
                                 qk_conv_kernel_size=qk_conv_kernel_size,
                                 output_gate_rank=output_gate_rank,
                                 implementation=resolved, **kwargs)

        class ConfiguredSwiGLU(GluMlp):
            def __init__(self, in_features, hidden_features=None, act_layer=None, **kwargs):
                budget = hidden_features or in_features
                gate_width = ((2 * budget // 3 + 15) // 16) * 16
                super().__init__(
                    in_features, hidden_features=2 * gate_width,
                    act_layer=nn.SiLU, gate_last=False, **kwargs,
                )

            def init_weights(self):
                # Keep timm's Linear trunc_normal_(std=.02), zero-bias init
                # for both branches. GluMlp's initializer instead overwrites
                # the second half, which is the value branch for gate_last=False.
                pass

        class ConfiguredBlock(Block):
            def __init__(self, dim: int, *args: Any, **kwargs: Any) -> None:
                kwargs["mlp_layer"] = ConfiguredSwiGLU
                if drop_path_schedule == "constant":
                    kwargs["drop_path"] = drop_path_rate
                super().__init__(dim, *args, **kwargs)

            def forward(self, x, attn_mask=None, is_causal=False):
                if attn_mask is not None and not isinstance(attn_mask, VisionTokenLayout):
                    raise ValueError("MemSolve accepts only the vision token validity layout")
                grid = default_grid if attn_mask is None or attn_mask.grid_size is None else attn_mask.grid_size
                if x.shape[1] != grid[0] * grid[1]:
                    raise ValueError("MemSolve requires the patch grid without prefix tokens")
                valid = None if attn_mask is None else attn_mask.valid_mask
                patches = x if valid is None else torch.where(valid[..., None], x, 0)
                return super().forward(
                    patches, attn_mask=VisionTokenLayout(valid, grid), is_causal=is_causal,
                )

        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, num_classes=num_classes,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=bias, proj_bias=bias, class_token=False, pos_embed="learn",
            global_pool="avg", fc_norm=False, init_values=None,
            dynamic_img_size=dynamic_img_size, dynamic_img_pad=dynamic_img_pad,
            drop_path_rate=drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=norm_eps),
            block_fn=ConfiguredBlock, attn_layer=ConfiguredMixer,
        )

    def get_extra_state(self) -> dict[str, object]:
        return {"version": 5, "architecture": "vit3_qkconv_lpe2d", "ffn": "swiglu",
                "pooling": "token_ln_mean", "class_token": False, "layer_scale": False}

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("incompatible MemSolve vision scaffold checkpoint")

    @property
    def blocks(self) -> nn.Module:
        return self.encoder.blocks

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed

    @property
    def num_classes(self) -> int:
        return self.encoder.num_classes

    @property
    def num_features(self) -> int:
        return self.encoder.num_features

    def get_classifier(self) -> nn.Module:
        return self.encoder.get_classifier()

    def reset_classifier(self, num_classes: int, global_pool: str | None = None) -> None:
        if global_pool not in (None, "avg"):
            raise ValueError("MemSolve-ViT uses token-LN followed by mean pooling")
        self.encoder.reset_classifier(num_classes)

    def _layout(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> VisionTokenLayout:
        grid = self.encoder.patch_embed.dynamic_feat_size(x.shape[-2:])
        if valid_mask is not None:
            if valid_mask.dtype != torch.bool or valid_mask.shape != (x.shape[0], grid[0] * grid[1]):
                raise ValueError("valid_mask must be bool [B, number of patches], without CLS")
            valid_mask = valid_mask.to(device=x.device)
        return VisionTokenLayout(valid_mask, grid)

    def forward(self, x: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        layout = self._layout(x, valid_mask)
        features = self.encoder.forward_features(x, attn_mask=layout)
        if layout.valid_mask is None:
            return self.encoder.forward_head(features)
        # Features already passed through per-token final LN. Excluded patches
        # must not enter mean pooling, including their residual or LN bias.
        mask = layout.valid_mask
        pooled = torch.where(mask[..., None], features, 0).sum(1)
        pooled = pooled / mask.sum(1).clamp_min(1).unsqueeze(-1)
        return self.encoder.head(self.encoder.head_drop(pooled))

    def forward_intermediates(
        self, x: torch.Tensor, *, indices: int | Sequence[int] | None = None,
        valid_mask: torch.Tensor | None = None, norm: bool = False,
    ) -> list[torch.Tensor]:
        result = self.encoder.forward_intermediates(
            x, indices=indices, intermediates_only=True, norm=norm, output_fmt="NCHW",
            attn_mask=self._layout(x, valid_mask),
        )
        if not isinstance(result, list):
            raise RuntimeError("timm did not return intermediate feature maps")
        return result


def create_memsolve_vit(
    *, image_size: int, num_classes: int, embed_dim: int, depth: int, num_heads: int,
    rank: int, mlp_ratio: float, bias: bool, implementation: _Implementation = "reference",
    drop_path_rate: float = 0.0, norm_eps: float = 1e-6, patch_size: int = 16,
    drop_path_schedule: Literal["linear", "constant"] = "linear",
    dynamic_img_size: bool = False, dynamic_img_pad: bool = False,
    qk_conv_kernel_size: int = 3, output_gate_rank: int = 32,
) -> MemSolveViT:
    """Build the canonical MemSolve-ViT classifier or feature encoder."""
    return MemSolveViT(
        image_size=image_size, patch_size=patch_size, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, rank=rank,
        mlp_ratio=mlp_ratio, bias=bias, implementation=implementation,
        drop_path_rate=drop_path_rate, norm_eps=norm_eps,
        drop_path_schedule=drop_path_schedule,
        dynamic_img_size=dynamic_img_size, dynamic_img_pad=dynamic_img_pad,
        qk_conv_kernel_size=qk_conv_kernel_size, output_gate_rank=output_gate_rank,
    )


def create_memsolve_vit_variant(
    variant: str, *, image_size: int, num_classes: int = 1000, rank: int | None = None,
    bias: bool = True, implementation: _Implementation = "reference",
    dynamic_img_size: bool = False, dynamic_img_pad: bool = False,
    qk_conv_kernel_size: int = 3, output_gate_rank: int = 32,
) -> MemSolveViT:
    spec = vision_spec(variant)
    return create_memsolve_vit(
        image_size=image_size, num_classes=num_classes, embed_dim=spec.embed_dim,
        depth=spec.depth, num_heads=spec.num_heads,
        rank=vision_default_rank(variant) if rank is None else rank,
        mlp_ratio=4.0,
        bias=bias, implementation=implementation, drop_path_rate=spec.drop_path_rate,
        dynamic_img_size=dynamic_img_size, dynamic_img_pad=dynamic_img_pad,
        qk_conv_kernel_size=qk_conv_kernel_size, output_gate_rank=output_gate_rank,
    )


__all__ = ["MemSolveViT", "VisionSpec", "VisionTokenLayout", "create_memsolve_vit",
           "create_memsolve_vit_variant", "vision_default_rank", "vision_spec"]


# These describe input preprocessing, not published pretrained weights.
default_cfgs = {
    f"memsolve_vit_{variant}": {
        "input_size": (3, 224, 224), "fixed_input_size": True,
        "num_classes": 1000, "interpolation": "bicubic", "crop_pct": 0.875,
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "first_conv": "encoder.patch_embed.proj", "classifier": "encoder.head",
    }
    for variant in ("tiny", "small", "base")
}


def _registered_vit(variant: str, pretrained: bool, **kwargs: Any) -> MemSolveViT:
    from timm.models import build_model_with_cfg

    if pretrained:
        raise ValueError("No pretrained MemSolve-ViT weights are published for this contract")
    if kwargs.pop("features_only", False):
        raise ValueError("Use MemSolveViT.forward_intermediates for feature maps")
    if kwargs.pop("in_chans", 3) != 3:
        raise ValueError("MemSolve-ViT currently expects three input channels")
    if kwargs.pop("global_pool", "avg") != "avg":
        raise ValueError("MemSolve-ViT uses token-LN followed by mean pooling")
    if "image_size" in kwargs and "img_size" in kwargs:
        raise ValueError("Pass img_size (timm) or image_size, not both")
    image_size = kwargs.pop("img_size", kwargs.pop("image_size", 224))
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2 or image_size[0] != image_size[1]:
            raise ValueError("The configured image size must be square; use dynamic_img_size for other grids")
        image_size = image_size[0]
    spec = vision_spec(variant)
    defaults = dict(
        image_size=image_size, patch_size=16, embed_dim=spec.embed_dim,
        depth=spec.depth, num_heads=spec.num_heads, rank=vision_default_rank(variant),
        mlp_ratio=4.0, drop_path_rate=spec.drop_path_rate,
    )
    return build_model_with_cfg(
        MemSolveViT, f"memsolve_vit_{variant}", pretrained=False,
        kwargs_filter=("img_size", "in_chans"), **(defaults | kwargs),
    )


def memsolve_vit_tiny(pretrained: bool = False, **kwargs: Any) -> MemSolveViT:
    """MemSolve-ViT-T: width 192, 12 blocks, 6 heads, rank 16."""
    return _registered_vit("tiny", pretrained, **kwargs)


def memsolve_vit_small(pretrained: bool = False, **kwargs: Any) -> MemSolveViT:
    """MemSolve-ViT-S: width 384, 12 blocks, 6 heads, rank 32."""
    return _registered_vit("small", pretrained, **kwargs)


def memsolve_vit_base(pretrained: bool = False, **kwargs: Any) -> MemSolveViT:
    """MemSolve-ViT-B: width 768, 12 blocks, 12 heads, rank 48."""
    return _registered_vit("base", pretrained, **kwargs)


# Geometry helpers remain importable without the optional vision dependency.
try:
    from timm.models import register_model
except ModuleNotFoundError as error:
    if error.name != "timm":
        raise
else:
    for entrypoint in (memsolve_vit_tiny, memsolve_vit_small, memsolve_vit_base):
        register_model(entrypoint)
