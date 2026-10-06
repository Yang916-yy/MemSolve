from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("timm")

import integrations.openmmlab as openmmlab
from experiments.imagenet import (
    IMAGENET_CHECKPOINT_FORMAT,
    IMAGENET_TRAIN_SAMPLES,
    IMAGENET_TRAIN_SHARDS,
    IMAGENET_VALIDATION_SAMPLES,
    IMAGENET_VALIDATION_SHARDS,
    IMAGENET_WDS_MANIFEST_SHA256,
    IMAGENET_WDS_SOURCE,
    WDS_SAMPLE_SHUFFLE_INITIAL,
    WDS_SAMPLE_SHUFFLE_SIZE,
    checkpoint_contract_digest,
    interpolate_position_embedding,
)
from experiments.train_openmmlab import _configure_checkpoint
from integrations.timm import VisionSpec, MemSolveViT


pytestmark = pytest.mark.integration


CONFIG_ROOT = (
    Path(__file__).resolve().parents[2] / "experiments" / "openmmlab" / "configs"
)


def _tiny_imagenet_checkpoint(
    source: MemSolveViT,
    *,
    image_size: int = 32,
) -> dict[str, object]:
    contract = {
        "tier": "small",
        "phase": "pretrain",
        "model": {
            "image_size": image_size,
            "patch_size": 16,
            "num_classes": 1000,
            "mlp_ratio": 4.0,
            "architecture": "vit3_qkconv_rope2d_mean_swiglu_v4", "ffn": "swiglu",
            "class_token": False,
            "layer_scale": False,
            "pooling": "token_ln_mean",
            "position_encoding": "rope_2d_axial",
            "drop_path_schedule": "linear",
            "norm_eps": 1e-6,
            "embed_dim": 32,
            "depth": 4,
            "num_heads": 4,
            "rank": 4,
            "drop_path_rate": 0.0,
        },
        "operator": {
            "qk_conv_kernel_size": 3,
            "output_gate_rank": 32,
            "bias": True,
            "implementation": "reference",
        },
        "train": {"recipe": "test"},
        "data": {
            "format": "webdataset-v1",
            "source": IMAGENET_WDS_SOURCE,
            "manifest_sha256": IMAGENET_WDS_MANIFEST_SHA256,
            "train": {
                "samples": IMAGENET_TRAIN_SAMPLES,
                "shards": IMAGENET_TRAIN_SHARDS,
            },
            "validation": {
                "samples": IMAGENET_VALIDATION_SAMPLES,
                "shards": IMAGENET_VALIDATION_SHARDS,
            },
            "streaming": {
                "shard_order": "global-epoch-permutation-then-rank-stride-worker-quota",
                "sample_shuffle": {
                    "buffer_size": WDS_SAMPLE_SHUFFLE_SIZE,
                    "initial_size": WDS_SAMPLE_SHUFFLE_INITIAL,
                },
                "source_views": 3,
                "repeated_augmentation_placement": "rank-local-physical-batch-interleave",
                "validation_partition": "worker-stride-full-per-rank",
                "worker_rng": "epoch-rank-worker-v1",
            },
        },
        "batching": {
            "world_size": 1,
            "physical_batch_size": 1,
            "effective_batch_size": 1,
            "grad_accum": 1,
            "samples_per_epoch": 1,
            "updates_per_epoch": 1,
        },
    }
    return {
        "format_version": IMAGENET_CHECKPOINT_FORMAT,
        "contract": contract,
        "contract_digest": checkpoint_contract_digest(contract),
        "model": source.state_dict(),
    }


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> openmmlab.MemSolveViTBackbone:
    """Build the real downstream adapter with a CPU-sized DeiT-shaped encoder."""

    spec = VisionSpec(
        embed_dim=32,
        depth=4,
        num_heads=4,
        drop_path_rate=0.0,
    )
    monkeypatch.setattr(openmmlab, "vision_spec", lambda variant: spec)
    model = openmmlab.MemSolveViTBackbone(
        variant="small",
        image_size=32,
        rank=4,
        out_indices=(0, 1, 2, 3),
        implementation="reference",
    ).eval()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    return model


def test_backbone_emits_xcit_style_four_level_pyramid(
    tiny_backbone: openmmlab.MemSolveViTBackbone,
) -> None:
    """A plain ViT grid must become strides 4, 8, 16, and 32 for OpenMMLab."""

    torch.manual_seed(101)
    image = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        features = tiny_backbone(image)

    assert len(features) == 4
    assert tuple(feature.shape for feature in features) == (
        (1, 32, 8, 8),
        (1, 32, 4, 4),
        (1, 32, 2, 2),
        (1, 32, 1, 1),
    )
    assert all(torch.isfinite(feature).all() for feature in features)


def test_backbone_masks_padded_pixels_before_global_mixing(
    tiny_backbone: openmmlab.MemSolveViTBackbone,
) -> None:
    """Padding must neither enter MemSolve nor leak back through the pyramid."""

    torch.manual_seed(102)
    valid_mask = torch.zeros(1, 48, 48, dtype=torch.bool)
    valid_mask[:, :32, :32] = True
    image = torch.randn(1, 3, 48, 48)
    changed_padding = torch.where(
        valid_mask.unsqueeze(1),
        image,
        torch.randn_like(image) * 100.0,
    )

    with torch.inference_mode():
        expected = tiny_backbone(image, valid_mask=valid_mask)
        actual = tiny_backbone(changed_padding, valid_mask=valid_mask)
        patch_mask = tiny_backbone._patch_valid_mask(image, valid_mask)

    for index, (expected_feature, actual_feature) in enumerate(
        zip(expected, actual, strict=True)
    ):
        output_mask = tiny_backbone._scale_mask(patch_mask, index)
        expanded_mask = output_mask.expand_as(actual_feature)
        torch.testing.assert_close(
            actual_feature[expanded_mask],
            expected_feature[expanded_mask],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual_feature[~expanded_mask],
            torch.zeros_like(actual_feature[~expanded_mask]),
            rtol=0.0,
            atol=0.0,
        )


def test_backbone_zeros_partial_patch_padding_before_embedding(
    tiny_backbone: openmmlab.MemSolveViTBackbone,
) -> None:
    """A valid edge patch must not retain arbitrary pixel-padding values."""

    torch.manual_seed(103)
    valid_mask = torch.zeros(1, 48, 48, dtype=torch.bool)
    valid_mask[:, :35, :34] = True
    image = torch.randn(1, 3, 48, 48)
    changed_padding = torch.where(
        valid_mask.unsqueeze(1),
        image,
        torch.randn_like(image) * 100.0,
    )

    with torch.inference_mode():
        expected = tiny_backbone(image, valid_mask=valid_mask)
        actual = tiny_backbone(changed_padding, valid_mask=valid_mask)

    for expected_feature, actual_feature in zip(expected, actual, strict=True):
        torch.testing.assert_close(
            actual_feature,
            expected_feature,
            rtol=0.0,
            atol=0.0,
        )


def test_backbone_accepts_the_imagenet_checkpoint_contract(
    tiny_backbone: openmmlab.MemSolveViTBackbone,
    tmp_path: Path,
) -> None:
    """The downstream backbone consumes the classifier checkpoint verbatim."""

    source = MemSolveViT(
        image_size=32,
        patch_size=16,
        num_classes=1000,
        embed_dim=32,
        depth=4,
        num_heads=4,
        rank=4,
        bias=True,
        implementation="reference",
        drop_path_rate=0.0, mlp_ratio=4.0,
    )
    checkpoint = tmp_path / "imagenet.pt"
    torch.save(_tiny_imagenet_checkpoint(source), checkpoint)

    tiny_backbone.load_pretrained(checkpoint)

    torch.testing.assert_close(
        tiny_backbone.encoder.patch_embed.proj.weight,
        source.encoder.patch_embed.proj.weight,
    )
    torch.testing.assert_close(
        tiny_backbone.encoder.pos_embed,
        source.encoder.pos_embed,
    )


def test_backbone_reuses_qk_filters_and_rebuilds_rope_at_a_larger_resolution(
    tiny_backbone: openmmlab.MemSolveViTBackbone,
    tmp_path: Path,
) -> None:
    target = type(tiny_backbone)(
        variant="small",
        image_size=48,
        rank=4,
        out_indices=(0, 1, 2, 3),
        implementation="reference",
    ).eval()
    source = MemSolveViT(
        image_size=32,
        patch_size=16,
        num_classes=1000,
        embed_dim=32,
        depth=4,
        num_heads=4,
        rank=4,
        bias=True,
        implementation="reference",
        drop_path_rate=0.0, mlp_ratio=4.0,
    )
    checkpoint = tmp_path / "imagenet-32px.pt"
    torch.save(_tiny_imagenet_checkpoint(source, image_size=32), checkpoint)

    target.load_pretrained(checkpoint)
    assert target.encoder.pos_embed is None
    torch.testing.assert_close(target.blocks[0].attn.mixer.qk_conv.weight,
                               source.blocks[0].attn.mixer.qk_conv.weight)
    with torch.no_grad():
        assert all(torch.isfinite(f).all() for f in target(torch.randn(1, 3, 48, 48)))


def test_backbone_rejects_a_digest_valid_but_incompatible_imagenet_checkpoint(
    tiny_backbone: openmmlab.MemSolveViTBackbone,
    tmp_path: Path,
) -> None:
    source = MemSolveViT(
        image_size=32,
        patch_size=16,
        num_classes=1000,
        embed_dim=32,
        depth=4,
        num_heads=4,
        rank=4,
        bias=True,
        implementation="reference",
        drop_path_rate=0.0,
    )
    payload = _tiny_imagenet_checkpoint(source)
    contract = payload["contract"]
    assert isinstance(contract, dict)
    model_contract = contract["model"]
    assert isinstance(model_contract, dict)
    model_contract["rank"] = 8
    payload["contract_digest"] = checkpoint_contract_digest(contract)
    checkpoint = tmp_path / "wrong-rank.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="at rank"):
        tiny_backbone.load_pretrained(checkpoint)


def test_launcher_places_new_run_checkpoint_on_backbone(tmp_path: Path) -> None:
    """The OpenMMLab bridge must use the backbone's explicit checkpoint field."""

    checkpoint = tmp_path / "imagenet.pt"
    checkpoint.touch()
    config = SimpleNamespace(model=SimpleNamespace(backbone=SimpleNamespace()))
    arguments = SimpleNamespace(
        test=None,
        resume=None,
        backbone_checkpoint=checkpoint,
    )

    assert not _configure_checkpoint(config, arguments)
    assert config.model.backbone.checkpoint == str(checkpoint.resolve())
    assert not hasattr(config.model.backbone, "init_cfg")


def test_coco_leaf_configs_keep_the_mask_rcnn_3x_contract() -> None:
    """COCO leaves should vary only the MemSolve DeiT III scale and rank."""

    expected = {
        "small": (32, 384, (3, 5, 7, 11)),
        "base": (48, 768, (3, 5, 7, 11)),
        "large": (64, 1024, (7, 11, 15, 23)),
    }
    for variant, (rank, channels, out_indices) in expected.items():
        path = CONFIG_ROOT / f"coco_mask_rcnn_memsolve_deit3_{variant}_3x.py"
        config = runpy.run_path(str(path))
        backbone = config["model"]["backbone"]
        assert config["_base_"] == "./_base_/coco_mask_rcnn_fpn_3x.py"
        assert config["custom_imports"] == {
            "imports": ["integrations.openmmlab"],
            "allow_failed_imports": False,
        }
        assert backbone == {
            "type": "MemSolveViTBackbone",
            "variant": variant,
            "rank": rank,
            "out_indices": out_indices,
            "implementation": "cuda",
        }
        assert config["model"]["neck"] == {
            "in_channels": [channels] * 4,
        }

    base = runpy.run_path(str(CONFIG_ROOT / "_base_" / "coco_mask_rcnn_fpn_3x.py"))
    assert base["model"]["type"] == "MemSolveMaskRCNN"
    assert "init_cfg" not in base["model"]["backbone"]
    assert base["model"]["neck"]["type"] == "FPN"
    assert base["model"]["neck"]["num_outs"] == 5
    assert base["max_epochs"] == 36
    assert base["param_scheduler"][1]["milestones"] == [27, 33]
    policy = base["train_pipeline"][3]["transforms"]
    assert policy[0][0]["scales"] == [
        (1333, short_side) for short_side in range(480, 801, 32)
    ]
    assert policy[1][0]["scales"] == [(1333, short_side) for short_side in (400, 500, 600)]


def test_ade20k_leaf_configs_keep_the_upernet_160k_contract() -> None:
    """ADE20K leaves should preserve the common UperNet 160k protocol."""

    expected = {
        "small": (32, 384, (3, 5, 7, 11), 384),
        "base": (48, 768, (3, 5, 7, 11), 512),
        "large": (64, 1024, (7, 11, 15, 23), 512),
    }
    for variant, (rank, channels, out_indices, decoder_channels) in expected.items():
        path = CONFIG_ROOT / f"ade20k_upernet_memsolve_deit3_{variant}_160k.py"
        config = runpy.run_path(str(path))
        backbone = config["model"]["backbone"]
        assert config["_base_"] == "./_base_/ade20k_upernet_160k.py"
        assert backbone["variant"] == variant
        assert backbone["rank"] == rank
        assert backbone["out_indices"] == out_indices
        assert backbone["implementation"] == "cuda"
        assert "core_mode" not in backbone
        assert config["model"]["decode_head"] == {
            "in_channels": [channels] * 4,
            "channels": decoder_channels,
        }
        assert config["model"]["auxiliary_head"] == {"in_channels": channels}

    base = runpy.run_path(str(CONFIG_ROOT / "_base_" / "ade20k_upernet_160k.py"))
    assert base["model"]["type"] == "MemSolveEncoderDecoder"
    assert "init_cfg" not in base["model"]["backbone"]
    assert base["model"]["decode_head"]["type"] == "UPerHead"
    assert base["crop_size"] == (512, 512)
    assert base["train_cfg"]["max_iters"] == 160000
