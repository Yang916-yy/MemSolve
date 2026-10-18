"""ViG-style ViT-Adapter integration for MMDetection and MMSegmentation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from experiments.imagenet import validate_checkpoint_contract
from integrations.timm import MemSolveViT, VisionTokenLayout, vision_spec, vision_default_rank
from integrations.vit_adapter import SpatialPriorModule, InteractionBlock, deform_inputs
from integrations.deform_attention import MSDeformAttn
from memsolve import CoreMode


class MemSolveViTBackbone(MemSolveViT):
    """Full ViT-Adapter: spatial prior, four interactions and stride 4/8/16/32 maps.

    The classification encoder retains its per-block CPE. Adapter initialization
    follows Vision-RWKV; no classifier or final classification norm is retained.
    """

    def __init__(self, *, variant: str, image_size: int = 224,
                 rank: int | None = None, core_mode: CoreMode | str = CoreMode.EQUILIBRIUM,
                 bias: bool = True, implementation: str = "cuda",
                 mlp_ratio: float | None = None, drop_path_rate: float = 0.2,
                 conv_inplane: int = 64, deform_num_heads: int = 6,
                 deform_ratio: float = 1.0, cffn_ratio: float = 0.25,
                 n_points: int = 4, with_cp: bool = False,
                 checkpoint: str | Path | None = None, checkpoint_weights: str = "auto"):
        spec = vision_spec(variant)
        if spec.depth % 4:
            raise ValueError("ViT-Adapter requires four equally sized block groups")
        if checkpoint_weights not in {"auto", "model", "model_ema"}:
            raise ValueError("checkpoint_weights must be auto, model or model_ema")
        rank = vision_default_rank(variant) if rank is None else rank
        mlp_ratio = (4.125 if variant == "small" else 4.0) if mlp_ratio is None else mlp_ratio
        super().__init__(image_size=image_size, patch_size=16, num_classes=0,
                         embed_dim=spec.embed_dim, depth=spec.depth, num_heads=spec.num_heads,
                         rank=rank, mlp_ratio=mlp_ratio, core_mode=core_mode, bias=bias,
                         implementation=implementation, drop_path_rate=drop_path_rate,
                         position_encoding="cpe", dynamic_img_size=True, dynamic_img_pad=True)
        self.encoder.norm = nn.Identity()
        self.variant = variant
        self.checkpoint_weights = checkpoint_weights
        self.out_channels = (spec.embed_dim,) * 4
        self._pretrained_geometry = dict(patch_size=16, embed_dim=spec.embed_dim,
            depth=spec.depth, num_heads=spec.num_heads, rank=rank, mlp_ratio=mlp_ratio,
            norm_eps=1e-6, position_encoding="cpe")
        self._pretrained_operator = dict(core_mode=CoreMode(core_mode).value, bias=bias)
        step = spec.depth // 4
        self.interaction_indexes = [(i * step, (i + 1) * step) for i in range(4)]
        self.level_embed = nn.Parameter(torch.empty(3, spec.embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=spec.embed_dim)
        self.interactions = nn.ModuleList([
            InteractionBlock(dim=spec.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                deform_ratio=deform_ratio, cffn_ratio=cffn_ratio, drop_path=drop_path_rate,
                extra_extractor=(i == 3), with_cp=with_cp) for i in range(4)])
        self.up = nn.ConvTranspose2d(spec.embed_dim, spec.embed_dim, 2, 2)
        self.output_norms = nn.ModuleList([nn.SyncBatchNorm(spec.embed_dim) for _ in range(4)])
        for module in (self.spm, self.interactions, self.up):
            module.apply(self._init_adapter)
        for module in self.interactions.modules():
            if isinstance(module, MSDeformAttn):
                module._reset_parameters()
        nn.init.normal_(self.level_embed)
        if checkpoint is not None:
            self.load_pretrained(checkpoint)

    @staticmethod
    def _init_adapter(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = math.prod(module.kernel_size) * module.out_channels / module.groups
            nn.init.normal_(module.weight, std=math.sqrt(2 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def load_pretrained(self, checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        contract = validate_checkpoint_contract(payload)
        if contract["tier"] != self.variant:
            raise ValueError("ImageNet checkpoint tier does not match the backbone")
        for key, expected in self._pretrained_geometry.items():
            if contract["model"].get(key) != expected:
                raise ValueError(f"ImageNet checkpoint geometry differs at {key}: expected {expected}")
        for key, expected in self._pretrained_operator.items():
            if contract["operator"].get(key) != expected:
                raise ValueError(f"ImageNet checkpoint operator differs at {key}")
        weights = self.checkpoint_weights
        if weights == "auto":
            weights = payload.get("selected_weights", "model")
        if weights not in {"model", "model_ema"} or not isinstance(payload.get(weights), dict):
            raise ValueError(f"Checkpoint does not contain selected weights: {weights}")
        state = {k.removeprefix("module."): v for k, v in payload[weights].items()}
        discarded = {"encoder.head.weight", "encoder.head.bias", "encoder.norm.weight", "encoder.norm.bias"}
        state = {k: v for k, v in state.items() if k not in discarded}
        result = self.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if k.startswith("encoder.")]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"Invalid pretrained backbone: missing={missing}, unexpected={result.unexpected_keys}")
        self.pretrained_weights = weights

    def forward(self, x, valid_mask=None):
        b, _, height, width = x.shape
        if valid_mask is not None:
            if valid_mask.shape != (b, height, width) or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool [batch, height, width]")
            valid_mask = valid_mask.to(x.device)
            x = x.masked_fill(~valid_mask[:, None], 0)
        else:
            valid_mask = torch.ones((b, height, width), dtype=torch.bool, device=x.device)
        # SPM's coarsest map has stride 32; preserve arbitrary rectangular crops.
        padding = (0, -width % 32, 0, -height % 32)
        x = F.pad(x, padding)
        valid_mask = F.pad(valid_mask, padding, value=False)
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        patch_mask = F.max_pool2d(valid_mask[:, None].float(), 16, 16).bool().flatten(1)
        layout = VisionTokenLayout(patch_mask, (h, w))
        deform1, deform2 = deform_inputs(x)
        c1, c2, c3, c4 = self.spm(x)
        sizes = [c2.shape[1], c3.shape[1], c4.shape[1]]
        c = torch.cat([v + self.level_embed[i] for i, v in enumerate((c2, c3, c4))], 1)
        tokens = self.encoder.norm_pre(self.encoder.patch_drop(
            self.encoder._pos_embed(self.encoder.patch_embed(x))))
        maps = []
        for interaction, (start, end) in zip(self.interactions, self.interaction_indexes):
            tokens, c = interaction(tokens, c, self.encoder.blocks[start:end],
                                    deform1, deform2, h, w, layout)
            maps.append(tokens.transpose(1, 2).reshape(b, -1, h, w))
        grids = [(h * 2, w * 2), (h, w), (h // 2, w // 2)]
        c2, c3, c4 = [v.transpose(1, 2).reshape(b, -1, *grid)
                      for v, grid in zip(c.split(sizes, dim=1), grids)]
        c1 = c1 + self.up(c2)
        outputs = []
        for stride, feature, vit, norm in zip((4, 8, 16, 32), (c1, c2, c3, c4), maps, self.output_norms):
            feature = norm(feature + F.interpolate(vit, size=feature.shape[-2:], mode="bilinear", align_corners=False))
            mask = F.max_pool2d(valid_mask[:, None].float(), stride, stride).bool()
            outputs.append(feature.masked_fill(~mask, 0))
        return tuple(outputs)


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


class _SegmentationMaskMixin:
    """MMSeg sliding-window inference with matching crop-local validity masks."""

    def slide_inference(self, inputs, batch_img_metas):
        # Adapted from MMSegmentation 1.2.2 EncoderDecoder (Apache-2.0).
        hs, ws = self.test_cfg.stride
        hc, wc = self.test_cfg.crop_size
        batch, _, height, width = inputs.shape
        preds = inputs.new_zeros((batch, self.out_channels, height, width))
        counts = inputs.new_zeros((batch, 1, height, width))
        full_mask = self._memsolve_valid_mask
        try:
            for row in range(max(height - hc + hs - 1, 0) // hs + 1):
                for col in range(max(width - wc + ws - 1, 0) // ws + 1):
                    y2, x2 = min(row * hs + hc, height), min(col * ws + wc, width)
                    y1, x1 = max(y2 - hc, 0), max(x2 - wc, 0)
                    crop = inputs[:, :, y1:y2, x1:x2]
                    self._memsolve_valid_mask = (None if full_mask is None else
                                                full_mask[:, y1:y2, x1:x2])
                    metas = [dict(m, img_shape=crop.shape[-2:]) for m in batch_img_metas]
                    logits = self.encode_decode(crop, metas)
                    preds[:, :, y1:y2, x1:x2] += logits
                    counts[:, :, y1:y2, x1:x2] += 1
        finally:
            self._memsolve_valid_mask = full_mask
        return preds / counts


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
        from mmdet.models.dense_heads import RPNHead
        from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
        from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

        @MMDET_MODELS.register_module(force=True)
        class MemSolveRPNHead(RPNHead):
            def forward_single(self, x):
                # Native NMS requires FP32 boxes and scores with matching dtypes.
                return tuple(v.float() for v in super().forward_single(x))

        @MMDET_MODELS.register_module(force=True)
        class MemSolveBBoxHead(Shared2FCBBoxHead):
            def forward(self, x):
                return tuple(v.float() if v is not None else None for v in super().forward(x))

        @MMDET_MODELS.register_module(force=True)
        class MemSolveRoIExtractor(SingleRoIExtractor):
            def forward(self, feats, rois, roi_scale_factor=None):
                # MMCV 2.1 RoIAlign has no BF16 kernel; retain FP32 coordinates.
                with torch.autocast(device_type=feats[0].device.type, enabled=False):
                    output = super().forward(tuple(x.float() for x in feats),
                                             rois.float(), roi_scale_factor)
                return output.to(feats[0].dtype)

        class MemSolveMaskRCNN(_MaskAwareMixin, MaskRCNN):
            pass

        MMDET_MODELS.register_module(module=MemSolveMaskRCNN, force=True)
        globals()["MemSolveMaskRCNN"] = MemSolveMaskRCNN

    if MMSEG_MODELS is not None:
        MMSEG_MODELS.register_module(module=MemSolveViTBackbone, force=True)
        class MemSolveEncoderDecoder(_SegmentationMaskMixin, _MaskAwareMixin, EncoderDecoder):
            pass

        MMSEG_MODELS.register_module(module=MemSolveEncoderDecoder, force=True)
        globals()["MemSolveEncoderDecoder"] = MemSolveEncoderDecoder


_register_openmmlab()


def layer_decay_groups(module, *, depth, decay_rate, lr, weight_decay):
    """BEiT layer decay: patch embedding 0, blocks 1..depth, adapter/heads depth+1."""
    groups = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        layer = depth + 1
        if name.startswith("backbone.encoder.patch_embed."):
            layer = 0
        elif name.startswith("backbone.encoder.blocks."):
            layer = int(name.split(".")[3]) + 1
        no_decay = param.ndim == 1 or name.endswith(".bias")
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = dict(params=[], lr=lr * decay_rate ** (depth + 1 - layer),
                               weight_decay=0.0 if no_decay else weight_decay)
        groups[key]["params"].append(param)
    return list(groups.values())


def _register_optimizer():
    try:
        from mmengine.optim import DefaultOptimWrapperConstructor
        from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
    except ImportError:
        return

    @OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
    class MemSolveLayerDecayConstructor(DefaultOptimWrapperConstructor):
        def add_params(self, params, module, **kwargs):
            params.extend(layer_decay_groups(module,
                depth=self.paramwise_cfg["num_layers"],
                decay_rate=self.paramwise_cfg["layer_decay_rate"],
                lr=self.base_lr, weight_decay=self.base_wd))


_register_optimizer()


__all__ = ["MemSolveViTBackbone"]
