"""Thin timm VisionTransformer adapter for MemSolve."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Sequence, cast

import torch
import torch.nn as nn
from torch.fx.experimental.symbolic_shapes import has_static_value

from memsolve import CoreMode, MemSolve, MemSolveConfig
from memsolve.ball.reference import axial_rotary_position_factors


_Implementation = Literal["reference", "cuda"]


@dataclass(frozen=True)
class VisionSpec:
    """Geometry and stochastic depth for one visual model scale."""

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
    rotary_factors: tuple[torch.Tensor, torch.Tensor] | None = None


def vision_spec(variant: str) -> VisionSpec:
    """Return the geometry of a visual model scale."""

    try:
        return _VISION_SPECS[variant]
    except KeyError as error:
        choices = ", ".join(_VISION_SPECS)
        raise ValueError(
            f"variant must be one of {{{choices}}}, got {variant!r}"
        ) from error


def vision_default_rank(variant: str) -> int:
    """Return the default operator rank for a visual model scale."""

    vision_spec(variant)
    return _VISION_DEFAULT_RANKS[variant]


def vision_default_mlp_ratio(variant: str) -> float:
    """Return the default FFN expansion ratio for a visual model scale."""

    vision_spec(variant)
    return 4.125 if variant == "small" else 4.0


def vision_variant_kwargs(
    variant: str,
    *,
    spec: VisionSpec | None = None,
    rank: int | None = None,
    mlp_ratio: float | None = None,
    drop_path_rate: float | None = None,
) -> dict[str, Any]:
    """Resolve canonical geometry, rank, MLP ratio, and DropPath for a scale."""

    resolved_spec = vision_spec(variant) if spec is None else spec
    return {
        "embed_dim": resolved_spec.embed_dim,
        "depth": resolved_spec.depth,
        "num_heads": resolved_spec.num_heads,
        "rank": vision_default_rank(variant) if rank is None else rank,
        "mlp_ratio": (
            vision_default_mlp_ratio(variant) if mlp_ratio is None else mlp_ratio
        ),
        "drop_path_rate": (
            resolved_spec.drop_path_rate
            if drop_path_rate is None
            else drop_path_rate
        ),
    }


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
        core_mode: CoreMode,
        implementation: _Implementation,
        qkv_bias: bool,
        qk_norm: bool,
        scale_norm: bool,
        proj_bias: bool,
        attn_drop: float,
        proj_drop: float,
        norm_layer: Any,
        rotary_2d: bool = False,
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
                core_mode=core_mode,
                bias=qkv_bias,
                rotary=rotary_2d,
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
        if attn_mask is not None and not isinstance(attn_mask, VisionTokenLayout):
            raise ValueError(
                "the vision adapter accepts only its token validity layout, "
                "not a generic attention mask"
            )
        valid_mask = None if attn_mask is None else attn_mask.valid_mask
        # Match the activation rounding of both backends under AMP.
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        kwargs = {}
        if self.mixer.config.rotary:
            if attn_mask is None or attn_mask.rotary_factors is None:
                raise ValueError("2D RoVE requires the spatial token layout")
            kwargs["rotary_factors"] = attn_mask.rotary_factors
        return self.mixer(
            x, valid_mask=valid_mask, implementation=self.implementation, **kwargs,
        )


def _create_vision_encoder(
    *,
    image_size: int,
    patch_size: int,
    num_classes: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    rank: int,
    mlp_ratio: float,
    core_mode: CoreMode | str = CoreMode.EQUILIBRIUM,
    bias: bool,
    implementation: _Implementation,
    drop_path_rate: float,
    norm_eps: float,
    dynamic_img_size: bool,
    dynamic_img_pad: bool,
    position_encoding: str = "cpe",
    drop_path_schedule: str = "linear",
    norm_type: str = "layernorm",
    rotary_2d: bool = False,
) -> nn.Module:
    """Build the ViT3-style CPE/GELU encoder with normalized-token mean pooling.

    Like the pinned upstream ordinary ViT3, blocks have no LayerScale and
    the token sequence has no CLS prefix. Recipes select stochastic depth.
    """

    if norm_type not in {"layernorm", "rmsnorm"}:
        raise ValueError("norm_type must be layernorm or rmsnorm")
    if rotary_2d and (embed_dim // num_heads) % 4:
        raise ValueError("2D RoVE requires head_dim divisible by four")
    if position_encoding not in {"learned", "cpe"}:
        raise ValueError("position_encoding must be learned or cpe")
    if drop_path_schedule not in {"constant", "linear"}:
        raise ValueError("drop_path_schedule must be constant or linear")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    if not 0.0 <= drop_path_rate < 1.0:
        raise ValueError("drop_path_rate must be in [0, 1)")
    if norm_eps != 1e-6:
        raise ValueError("the vision scaffold requires norm_eps = 1e-6")

    from timm.models.vision_transformer import Block, VisionTransformer

    _require_timm_attention_mask_api(VisionTransformer)

    mode = CoreMode(core_mode)
    resolved_implementation = _validate_implementation(implementation)

    class ConfiguredMixer(_TimmMemSolveMixer):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                dim,
                num_heads,
                rank=rank,
                core_mode=mode,
                implementation=resolved_implementation,
                rotary_2d=rotary_2d,
                **kwargs,
            )

    class ConstantDropPathBlock(Block):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["drop_path"] = drop_path_rate
            super().__init__(*args, **kwargs)

    block_type = ConstantDropPathBlock if drop_path_schedule == "constant" else Block

    class CPEBlock(block_type):
        # ViT³ / CPVT residual CPE via cuDNN depthwise convolution.
        def __init__(self, dim: int, *args: Any, **kwargs: Any) -> None:
            super().__init__(dim, *args, **kwargs)
            self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        def forward(self, x, attn_mask=None, is_causal=False):
            if attn_mask is not None and not isinstance(attn_mask, VisionTokenLayout):
                raise ValueError("CPE accepts only the vision token validity layout")
            grid = None if attn_mask is None else attn_mask.grid_size
            height, width = grid or (image_size // patch_size, image_size // patch_size)
            if x.shape[1] != height * width:
                raise ValueError("CPE token count must match the spatial grid")
            valid = None if attn_mask is None else attn_mask.valid_mask
            patches = x
            if valid is not None:
                patches = torch.where(valid[..., None], patches, 0)
            spatial = patches.reshape(x.shape[0], height, width, x.shape[2]).permute(0, 3, 1, 2)
            if torch.compiler.is_compiling() and not all(has_static_value(s) for s in spatial.shape):
                # Avoid symbolic NHWC view stride-order ambiguity in Inductor AOT saved views.
                spatial = spatial.contiguous()
            # NHWC tokens give a channels-last convolution view; cuDNN owns dispatch.
            local = self.cpe(spatial).flatten(2).transpose(1, 2)
            patches = patches + local
            if valid is not None:
                patches = torch.where(valid[..., None], patches, 0)
            x = patches
            return super().forward(x, attn_mask=attn_mask, is_causal=is_causal)

    encoder = VisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        global_pool="avg",
        class_token=False, fc_norm=False,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=bias,
        proj_bias=bias,
        no_embed_class=True,
        pos_embed="none" if position_encoding == "cpe" else "learn",
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
        init_values=None,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm, eps=norm_eps),
        block_fn=CPEBlock if position_encoding == "cpe" else block_type,
        attn_layer=ConfiguredMixer,
    )
    return encoder


class MemSolveViT(nn.Module):
    """ViT3-derived encoder with CPE, GELU and normalized-token mean pooling."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int,
        depth: int,
        num_heads: int,
        rank: int,
        mlp_ratio: float = 4.0,
        core_mode: CoreMode | str = CoreMode.EQUILIBRIUM,
        bias: bool = True,
        implementation: _Implementation = "reference",
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        position_encoding: str = "cpe",
        drop_path_schedule: str = "linear",
        norm_type: str = "layernorm",
        rotary_2d: bool = False,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.norm_type = norm_type
        self.rotary_2d = rotary_2d
        self.head_dim = embed_dim // num_heads
        self.encoder = _create_vision_encoder(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            rank=rank,
            mlp_ratio=mlp_ratio,
            core_mode=core_mode,
            bias=bias,
            implementation=implementation,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            position_encoding=position_encoding,
            drop_path_schedule=drop_path_schedule,
            norm_type=norm_type,
            rotary_2d=rotary_2d,
        )

    def get_extra_state(self):
        state = {"version": 7, "architecture": "accretive_equilibrium_vit",
                "ffn": "gelu", "pooling": "token_ln_mean", "class_token": False,
                "layer_scale": False, "position_encoding": self.position_encoding}
        if self.norm_type != "layernorm":
            state.update(norm_type=self.norm_type, pooling="token_rms_mean")
        if self.rotary_2d:
            state["rotary_2d"] = True
        return state

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("incompatible MemSolve vision scaffold checkpoint")

    @property
    def num_classes(self):
        return self.encoder.num_classes

    @property
    def num_features(self):
        return self.encoder.num_features

    def get_classifier(self):
        return self.encoder.get_classifier()

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool not in (None, "avg"):
            raise ValueError("MemSolve-ViT uses token-LN mean pooling")
        self.encoder.reset_classifier(num_classes, global_pool="avg")

    @property
    def blocks(self) -> nn.Module:
        return self.encoder.blocks  # type: ignore[no-any-return]

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed  # type: ignore[no-any-return]

    def _token_layout(self, x, valid_mask):
        grid = self.encoder.patch_embed.dynamic_feat_size(x.shape[-2:])
        factors = None
        if self.rotary_2d:
            rows, cols = torch.meshgrid(
                torch.arange(grid[0], device=x.device),
                torch.arange(grid[1], device=x.device), indexing="ij",
            )
            coordinates = torch.stack((rows, cols), dim=-1).reshape(1, -1, 2)
            # All heads and blocks share the same unit-amplitude axial rotations.
            factors = axial_rotary_position_factors(coordinates, head_dim=self.head_dim)
        return VisionTokenLayout(valid_mask, grid, factors)

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = self._token_layout(x, valid_mask)
        features = self.encoder.forward_features(x, attn_mask=layout)
        if valid_mask is None:
            return self.encoder.forward_head(features)
        mask = valid_mask.to(device=features.device)
        pooled = torch.where(mask[..., None], features, 0).sum(1)
        pooled = pooled / mask.sum(1).clamp_min(1).unsqueeze(-1)
        return self.encoder.head(self.encoder.head_drop(pooled))

    def forward_intermediates(
        self,
        x: torch.Tensor,
        *,
        indices: int | Sequence[int] | None = None,
        valid_mask: torch.Tensor | None = None,
        norm: bool = False,
    ) -> list[torch.Tensor]:
        layout = self._token_layout(x, valid_mask)
        return self.encoder.forward_intermediates(
            x,
            indices=indices,
            intermediates_only=True,
            norm=norm,
            output_fmt="NCHW",
            attn_mask=layout,
        )


def create_memsolve_vit(
    *,
    image_size: int,
    num_classes: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    rank: int,
    mlp_ratio: float,
    core_mode: CoreMode | str = CoreMode.EQUILIBRIUM,
    bias: bool,
    implementation: _Implementation = "reference",
    drop_path_rate: float = 0.0,
    norm_eps: float = 1e-6,
    patch_size: int = 16,
    dynamic_img_size: bool = False,
    dynamic_img_pad: bool = False,
    position_encoding: str = "cpe",
    drop_path_schedule: str = "linear",
    norm_type: str = "layernorm",
    rotary_2d: bool = False,
) -> MemSolveViT:
    """Build the ViT3-style MemSolve classifier or feature encoder."""

    return MemSolveViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        rank=rank,
        mlp_ratio=mlp_ratio,
        core_mode=core_mode,
        bias=bias,
        implementation=implementation,
        drop_path_rate=drop_path_rate,
        norm_eps=norm_eps,
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
        position_encoding=position_encoding,
        drop_path_schedule=drop_path_schedule,
        norm_type=norm_type,
        rotary_2d=rotary_2d,
    )


def create_memsolve_vit_variant(
    variant: str,
    *,
    image_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 1000,
    rank: int | None = None,
    mlp_ratio: float | None = None,
    core_mode: CoreMode | str = CoreMode.EQUILIBRIUM,
    bias: bool = True,
    implementation: _Implementation = "reference",
    drop_path_rate: float | None = None,
    norm_eps: float = 1e-6,
    dynamic_img_size: bool = False,
    dynamic_img_pad: bool = False,
    position_encoding: str = "cpe",
    drop_path_schedule: str = "linear",
    norm_type: str = "layernorm",
    rotary_2d: bool = False,
) -> MemSolveViT:
    """Build a MemSolve-ViT model at the requested scale."""

    geometry = vision_variant_kwargs(
        variant,
        rank=rank,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
    )
    return create_memsolve_vit(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        **geometry,
        core_mode=core_mode,
        bias=bias,
        implementation=implementation,
        norm_eps=norm_eps,
        dynamic_img_size=dynamic_img_size,
        dynamic_img_pad=dynamic_img_pad,
        position_encoding=position_encoding,
        drop_path_schedule=drop_path_schedule,
        norm_type=norm_type,
        rotary_2d=rotary_2d,
    )


__all__ = [
    "VisionSpec",
    "MemSolveViT",
    "VisionTokenLayout",
    "create_memsolve_vit",
    "create_memsolve_vit_variant",
    "memsolve_vit_base",
    "memsolve_vit_large",
    "memsolve_vit_small",
    "memsolve_vit_tiny",
    "vision_default_mlp_ratio",
    "vision_default_rank",
    "vision_spec",
    "vision_variant_kwargs",
]


# These describe input preprocessing, not published pretrained weights.
default_cfgs = {
    f"memsolve_vit_{variant}": {
        "input_size": (3, 224, 224), "fixed_input_size": True,
        "num_classes": 1000, "interpolation": "bicubic", "crop_pct": 0.875,
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "first_conv": "encoder.patch_embed.proj", "classifier": "encoder.head",
    }
    for variant in _VISION_SPECS
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
        raise ValueError("MemSolve-ViT uses token-LN mean pooling")
    if "image_size" in kwargs and "img_size" in kwargs:
        raise ValueError("Pass img_size (timm) or image_size, not both")
    image_size = kwargs.pop("img_size", kwargs.pop("image_size", 224))
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2 or image_size[0] != image_size[1]:
            raise ValueError("The configured image size must be square; use dynamic_img_size for other grids")
        image_size = image_size[0]
    defaults = {
        "image_size": image_size,
        "patch_size": 16,
        **vision_variant_kwargs(variant),
    }
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


def memsolve_vit_large(pretrained: bool = False, **kwargs: Any) -> MemSolveViT:
    """MemSolve-ViT-L: width 1024, 24 blocks, 16 heads, rank 64."""
    return _registered_vit("large", pretrained, **kwargs)


# Geometry helpers remain importable without the optional vision dependency.
try:
    from timm.models import register_model
except ModuleNotFoundError as error:
    if error.name != "timm":
        raise
else:
    for entrypoint in (
        memsolve_vit_tiny,
        memsolve_vit_small,
        memsolve_vit_base,
        memsolve_vit_large,
    ):
        register_model(entrypoint)
