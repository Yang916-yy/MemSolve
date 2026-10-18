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
)
from experiments.train_openmmlab import _configure_checkpoint
from integrations.timm import VisionSpec, MemSolveViT
from memsolve import CoreMode


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
            "norm_eps": 1e-6,
            "embed_dim": 32,
            "depth": 4,
            "num_heads": 4,
            "rank": 4,
            "drop_path_rate": 0.0,
            "position_encoding": "cpe",
        },
        "operator": {
            "core_mode": "equilibrium",
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
                "worker_rng": "epoch-rank-worker-v1",
                "shard_order": "global-epoch-permutation-then-rank-stride-worker-quota",
                "sample_shuffle": {
                    "buffer_size": WDS_SAMPLE_SHUFFLE_SIZE,
                    "initial_size": WDS_SAMPLE_SHUFFLE_INITIAL,
                },
                "source_views": 3,
                "repeated_augmentation_placement": "rank-local-physical-batch-interleave",
                "validation_partition": "worker-stride-full-per-rank",
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
    """Build the real downstream adapter with a CPU-sized CPE encoder."""

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
        mlp_ratio=4.0, conv_inplane=8, deform_num_heads=4, drop_path_rate=0.0,
        core_mode=CoreMode.EQUILIBRIUM,

        implementation="reference",
    ).eval()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    return model


def test_backbone_emits_adapter_four_level_pyramid(
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


    for index, (expected_feature, actual_feature) in enumerate(
        zip(expected, actual, strict=True)
    ):
        padded = torch.nn.functional.pad(valid_mask[:, None].float(), (0, 16, 0, 16))
        output_mask = torch.nn.functional.max_pool2d(padded, 4 * 2**index).bool()
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
        core_mode=CoreMode.EQUILIBRIUM,
        position_encoding="cpe",

        bias=True,
        implementation="reference",
        drop_path_rate=0.0,
    )
    checkpoint = tmp_path / "imagenet.pt"
    torch.save(_tiny_imagenet_checkpoint(source), checkpoint)

    tiny_backbone.load_pretrained(checkpoint)

    torch.testing.assert_close(
        tiny_backbone.encoder.patch_embed.proj.weight,
        source.encoder.patch_embed.proj.weight,
    )
    torch.testing.assert_close(
        tiny_backbone.encoder.blocks[0].cpe.weight,
        source.encoder.blocks[0].cpe.weight,
    )


def test_pretrained_ema_selection_and_rectangular_grid(tiny_backbone, tmp_path):
    source = MemSolveViT(image_size=32, patch_size=16, num_classes=1000,
        embed_dim=32, depth=4, num_heads=4, rank=4, implementation="reference")
    payload = _tiny_imagenet_checkpoint(source)
    payload["model_ema"] = {k: v.clone() if isinstance(v, torch.Tensor) else v
                            for k, v in payload["model"].items()}
    payload["model_ema"]["encoder.patch_embed.proj.weight"].add_(1)
    payload["selected_weights"] = "model_ema"
    checkpoint = tmp_path / "ema.pt"
    torch.save(payload, checkpoint)
    tiny_backbone.load_pretrained(checkpoint)
    assert tiny_backbone.pretrained_weights == "model_ema"
    torch.testing.assert_close(tiny_backbone.patch_embed.proj.weight,
                               source.patch_embed.proj.weight + 1)
    with torch.no_grad():
        maps = tiny_backbone(torch.randn(1, 3, 32, 64))
    assert maps[-1].shape == (1, 32, 1, 2)


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
        core_mode=CoreMode.EQUILIBRIUM,
        position_encoding="cpe",

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


@pytest.mark.parametrize("variant", ["tiny", "small", "base"])
def test_adapter_recipe_budget(variant):
    for task, suffix in [("coco_mask_rcnn", "1x"), ("ade20k_upernet", "160k")]:
        leaf = runpy.run_path(str(CONFIG_ROOT / f"{task}_memsolve_adapter_{variant}_{suffix}.py"))
        base = runpy.run_path(str(CONFIG_ROOT / leaf["_base_"]))
        assert base["train_dataloader"]["batch_size"] * 2 == 16
        assert base["optim_wrapper"]["dtype"] == "bfloat16"
        if suffix == "1x":
            assert base["train_cfg"]["max_epochs"] == 12
            assert base["param_scheduler"][1]["milestones"] == [8, 11]
        else:
            assert base["train_cfg"]["max_iters"] == 160000
            assert base["model"]["test_cfg"]["mode"] == "slide"


def test_full_adapter_gradients_include_injector_and_spatial_prior(tiny_backbone):
    outputs = tiny_backbone(torch.randn(2, 3, 32, 64))
    sum(x.square().mean() for x in outputs).backward()
    for name, parameter in tiny_backbone.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert tiny_backbone.interactions[0].injector.gamma.grad.abs().sum() > 0


def test_layer_decay_assigns_backbone_and_new_parameters(tiny_backbone):
    model = torch.nn.Module()
    model.backbone = tiny_backbone
    groups = openmmlab.layer_decay_groups(model, depth=4, decay_rate=.8, lr=.001, weight_decay=.05)
    by_id = {id(p): g for g in groups for p in g["params"]}
    assert len(by_id) == len(list(model.parameters()))
    assert by_id[id(tiny_backbone.patch_embed.proj.weight)]["lr"] == pytest.approx(.001 * .8**5)
    assert by_id[id(tiny_backbone.interactions[0].injector.gamma)]["weight_decay"] == 0
    assert by_id[id(tiny_backbone.up.weight)]["lr"] == .001


def test_dynamic_cpe_matches_fixed_classifier():
    kwargs = dict(image_size=32, num_classes=7, embed_dim=32, depth=2,
                  num_heads=4, rank=4, implementation="reference")
    fixed = MemSolveViT(**kwargs).eval()
    dynamic = MemSolveViT(**kwargs, dynamic_img_size=True, dynamic_img_pad=True).eval()
    dynamic.load_state_dict(fixed.state_dict())
    with torch.no_grad():
        image = torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(dynamic(image), fixed(image), rtol=0, atol=0)
        assert dynamic(torch.randn(2, 3, 35, 60)).shape == (2, 7)


def test_sliding_inference_slices_masks_and_preserves_metadata():
    class Segmenter(openmmlab._SegmentationMaskMixin):
        test_cfg = SimpleNamespace(stride=(16, 16), crop_size=(32, 32))
        out_channels = 1
        def encode_decode(self, crop, metas):
            assert self._memsolve_valid_mask.shape == crop[:, 0].shape
            return crop[:, :1].masked_fill(~self._memsolve_valid_mask[:, None], 0)
    model = Segmenter()
    mask = torch.ones(1, 40, 50, dtype=torch.bool)
    mask[:, 35:, :] = False
    model._memsolve_valid_mask = mask
    image = torch.randn(1, 3, 40, 50)
    metas = [dict(img_shape=(35, 50))]
    result = model.slide_inference(image, metas)
    torch.testing.assert_close(result, image[:, :1].masked_fill(~mask[:, None], 0))
    assert model._memsolve_valid_mask is mask
    assert metas[0]["img_shape"] == (35, 50)


@pytest.mark.parametrize("task", ["mmdet", "mmseg"])
@pytest.mark.parametrize("amp", [False, True])
def test_openmmlab_heads_train_and_predict(task, amp, monkeypatch):
    """Exercise the actual heads, loss functions and framework dispatch on CPU."""
    pytest.importorskip("mmcv.ops")
    torch.manual_seed(19)
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.structures import InstanceData, PixelData
    import numpy as np

    monkeypatch.setattr(openmmlab, "vision_spec", lambda _: VisionSpec(32, 4, 4, 0))
    name = ("coco_mask_rcnn_memsolve_adapter_small_1x.py" if task == "mmdet"
            else "ade20k_upernet_memsolve_adapter_small_160k.py")
    cfg = Config.fromfile(CONFIG_ROOT / name)
    cfg.model.backbone.update(rank=4, mlp_ratio=4, implementation="reference",
                             deform_num_heads=4, conv_inplane=8, drop_path_rate=0)
    init_default_scope(task)
    if task == "mmdet":
        from mmdet.registry import MODELS
        from mmdet.structures import DetDataSample
        from mmdet.structures.mask import BitmapMasks
        cfg.model.neck.in_channels = [32] * 4
        model = MODELS.build(cfg.model)
        samples = []
        for _ in range(2):
            sample = DetDataSample(metainfo=dict(img_shape=(64, 64), ori_shape=(64, 64),
                                                 batch_input_shape=(64, 64), scale_factor=(1., 1.)))
            mask = np.zeros((1, 64, 64), dtype=np.uint8)
            mask[:, 8:40, 8:40] = 1
            sample.gt_instances = InstanceData(bboxes=torch.tensor([[8., 8., 40., 40.]]),
                labels=torch.tensor([0]), masks=BitmapMasks(mask, 64, 64))
            samples.append(sample)
    else:
        from mmseg.registry import MODELS
        from mmseg.structures import SegDataSample
        cfg.model.data_preprocessor.size = (64, 64)
        cfg.model.decode_head.update(in_channels=[32] * 4, channels=16)
        cfg.model.auxiliary_head.update(in_channels=32, channels=16)
        cfg.model.test_cfg.update(crop_size=(32, 32), stride=(24, 24))
        model = MODELS.build(cfg.model)
        samples = [SegDataSample(metainfo=dict(img_shape=(64, 64), ori_shape=(64, 64),
                   pad_shape=(64, 64), padding_size=[0, 0, 0, 0]),
                   gt_sem_seg=PixelData(data=torch.randint(0, 150, (1, 64, 64))))
                   for _ in range(2)]
    model.init_weights()
    batch = model.data_preprocessor(dict(inputs=[torch.randn(3, 64, 64) for _ in range(2)],
                                         data_samples=samples), training=True)
    image, samples = batch["inputs"], batch["data_samples"]
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        losses = model(image, samples, mode="loss")
    loss = sum(sum(v) if isinstance(v, list) else v for k, v in losses.items() if "loss" in k)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.encoder.blocks[0].cpe.weight.grad is not None
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        predictions = model(image, samples, mode="predict")
    assert len(predictions) == 2


def test_adapter_checkpointing_preserves_gradients_and_batchnorm(tiny_backbone):
    import copy
    direct = tiny_backbone.train()
    checkpointed = copy.deepcopy(direct)
    for module in checkpointed.modules():
        if isinstance(module, (openmmlab.InteractionBlock,)):
            module.injector.with_cp = True
            module.extractor.with_cp = True
            if module.extra_extractors is not None:
                for extractor in module.extra_extractors:
                    extractor.with_cp = True
    image = torch.randn(2, 3, 32, 64)
    for model in (direct, checkpointed):
        sum(x.square().mean() for x in model(image)).backward()
    for (name, p), (_, q) in zip(direct.named_parameters(), checkpointed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, msg=name)
    for (name, p), (_, q) in zip(direct.named_buffers(), checkpointed.named_buffers()):
        torch.testing.assert_close(p, q, msg=name)


@pytest.mark.cuda
def test_mmcv_sampling_matches_upstream_cpu_oracle():
    pytest.importorskip("mmcv.ops")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
    from integrations.deform_attention import ms_deform_attn_core_pytorch
    torch.manual_seed(42)
    shape = torch.tensor([[2, 4]])
    starts = torch.tensor([0])
    value = torch.randn(2, 8, 4, 8, requires_grad=True)
    locations = torch.rand(2, 6, 4, 1, 4, 2, requires_grad=True)
    weights = torch.randn(2, 6, 4, 1, 4).softmax(-1).requires_grad_()
    expected = ms_deform_attn_core_pytorch(value, shape, locations, weights)
    gpu = [x.detach().cuda().requires_grad_() for x in (value, locations, weights)]
    actual = MultiScaleDeformableAttnFunction.apply(gpu[0], shape.cuda(), starts.cuda(), gpu[1], gpu[2], 64)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for x, y in zip((value, locations, weights), gpu):
        torch.testing.assert_close(y.grad.cpu(), x.grad, rtol=1e-4, atol=1e-5)
