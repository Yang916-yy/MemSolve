from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import functools
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from torch.utils.data import Dataset, TensorDataset

import experiments.train_transformers as train_transformers
from experiments.sequence_data import (
    DatasetBundle,
    NucleotideTokenizer,
    PATHFINDER_SPLIT_PROTOCOL,
    TokenVocabulary,
    build_packed_tokens,
    collate_tokens,
    collate_values,
    pathfinder_split_indices,
    prepare_genomic_benchmarks,
    prepare_lra,
    stratified_split_indices,
    validate_formal_source_provenance,
)
from experiments.train_transformers import (
    SequenceClassifier,
    SequenceEncoder,
    SequencePairClassifier,
    TrainingConfig,
    _build_collate,
    _build_run_payload,
    _runtime_metadata,
    _is_better_validation_checkpoint,
    _makes_early_stop_progress,
    _validate_formal_data_source,
    build_model,
    parse_args,
    resolve_args,
    train,
)
from experiments.sequence_data import make_loader
from memsolve.ball import cuda


pytestmark = pytest.mark.experiment


def _encoder(mixer: str) -> SequenceEncoder:
    return SequenceEncoder(
        input_kind="tokens",
        vocab_size=12,
        pad_token_id=0,
        max_length=8,
        dim=16,
        depth=2,
        num_heads=2,
        rank=4,
        mixer=mixer,  # type: ignore[arg-type]


        implementation="reference",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    )


def _grid_encoder(mixer: str) -> SequenceEncoder:
    return SequenceEncoder(
        input_kind="values",
        vocab_size=None,
        pad_token_id=None,
        max_length=6,
        dim=16,
        depth=2,
        num_heads=2,
        rank=4,
        mixer=mixer,  # type: ignore[arg-type]


        implementation="reference",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
        grid_shape=(2, 3),
    )


@pytest.mark.parametrize("position_encoding", ["rove_1d", "rove_2d"])
@pytest.mark.parametrize("mixer", ["memsolve", "mha"])
def test_sequence_rove_replaces_absolute_positions_and_masks_padding(position_encoding, mixer) -> None:
    grid = (2, 3) if position_encoding == "rove_2d" else None
    encoder = SequenceEncoder(input_kind="values", vocab_size=None, pad_token_id=None,
        max_length=6, dim=16, depth=2, num_heads=2, rank=4, mixer=mixer,
        implementation="reference", mlp_ratio=2., dropout=0., bias=True,
        grid_shape=grid, position_encoding=position_encoding)
    assert encoder.position_embedding is encoder.row_embedding is encoder.column_embedding is None
    assert not any("embedding" in name for name in encoder.state_dict() if "input_embedding" not in name)
    assert encoder.rotary_cos.shape == (1, 6, 4)
    assert encoder.rotary_cos.dtype == torch.float32
    if mixer == "memsolve":
        assert all(block.mixer.config.rotary for block in encoder.blocks)
    inputs = torch.randn(2, 6, 1)
    valid = torch.tensor([[True]*6, [True]*4+[False]*2])
    output = encoder(inputs, valid)
    changed = inputs.clone()
    changed[~valid] = 100.
    torch.testing.assert_close(output, encoder(changed, valid))
    assert torch.count_nonzero(output[~valid]) == 0
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_learned_position_embedding_uses_small_normal_initialization() -> None:
    torch.manual_seed(31)
    weights = _encoder("mha").position_embedding.weight.detach()
    assert abs(float(weights.mean())) < 0.006
    assert float(weights.std(unbiased=False)) == pytest.approx(0.02, abs=0.004)


def test_factorized_grid_positions_use_flat_position_scale() -> None:
    torch.manual_seed(31)
    encoder = _grid_encoder("mha")
    assert encoder.position_embedding is None
    assert encoder.row_embedding is not None
    assert encoder.column_embedding is not None
    assert not any(isinstance(module, nn.Conv2d) for module in encoder.modules())
    coordinates = encoder._position_features(length=6, device=torch.device("cpu"))
    torch.testing.assert_close(
        coordinates[5], encoder.row_embedding.weight[1] + encoder.column_embedding.weight[2]
    )
    assert float(coordinates.detach().std(unbiased=False)) == pytest.approx(0.02, abs=0.006)


def test_factorized_grid_positions_reject_invalid_grids() -> None:
    with pytest.raises(ValueError, match="value inputs"):
        SequenceEncoder(
            input_kind="tokens",
            vocab_size=8,
            pad_token_id=0,
            max_length=6,
            dim=8,
            depth=1,
            num_heads=2,
            rank=4,
            mixer="mha",
            implementation="reference",
            mlp_ratio=2.0,
            dropout=0.0,
            bias=True,
            grid_shape=(2, 3),
        )
    with pytest.raises(ValueError, match="exactly cover"):
        SequenceEncoder(
            input_kind="values",
            vocab_size=None,
            pad_token_id=None,
            max_length=8,
            dim=8,
            depth=1,
            num_heads=2,
            rank=4,
            mixer="mha",
            implementation="reference",
            mlp_ratio=2.0,
            dropout=0.0,
            bias=True,
            grid_shape=(2, 3),
        )
    with pytest.raises(ValueError, match="full 2x3 grid"):
        _grid_encoder("mha")(torch.ones(1, 5, 1), torch.ones(1, 5, dtype=torch.bool))


def test_pathx_length_is_covered_by_the_learned_position_table() -> None:
    encoder = SequenceEncoder(
        input_kind="values",
        vocab_size=None,
        pad_token_id=None,
        max_length=128**2,
        dim=8,
        depth=1,
        num_heads=2,
        rank=4,
        mixer="mha",
        implementation="reference",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    )
    last_position = torch.tensor([128**2 - 1])
    assert encoder.position_embedding(last_position).shape == (1, 8)


def test_pathx_full_length_reference_forward_and_backward_are_finite() -> None:
    encoder = SequenceEncoder(
        input_kind="values",
        vocab_size=None,
        pad_token_id=None,
        max_length=128**2,
        dim=8,
        depth=1,
        num_heads=2,
        rank=4,
        mixer="memsolve",
        implementation="reference",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    )
    inputs = torch.rand(1, 128**2, 1, requires_grad=True)
    valid_mask = torch.ones(1, 128**2, dtype=torch.bool)
    loss = encoder(inputs, valid_mask).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_byte_vocabulary_preserves_utf8_bytes_and_eos() -> None:
    vocabulary = TokenVocabulary.byte_level()
    encoded = vocabulary.encode_bytes("e\u0301", max_length=4)
    assert encoded == [
        vocabulary.lookup["e"],
        vocabulary.lookup[chr(0xCC)],
        vocabulary.lookup[chr(0x81)],
        vocabulary.eos_token_id,
    ]


def test_cuda_runtime_metadata_records_the_current_native_contract() -> None:
    metadata = _runtime_metadata(torch.device("cpu"), cuda_enabled=True)
    assert metadata["memsolve_cuda_contract"] == cuda._NATIVE_CONTRACT_VERSION


def test_nucleotide_tokenizer_does_not_confuse_unknown_with_padding() -> None:
    tokenizer = NucleotideTokenizer()
    assert tokenizer.encode("ACGTNZ", 8).tolist() == [2, 3, 4, 5, 6, 1]
    with pytest.raises(ValueError, match="empty genomic"):
        tokenizer.encode("", 8)


def test_stratified_split_is_deterministic_and_keeps_every_class_in_train() -> None:
    labels = [0] * 8 + [1] * 4 + [2]
    train, validation = stratified_split_indices(labels, 0.25, 19)
    assert (train, validation) == stratified_split_indices(labels, 0.25, 19)
    assert set(labels[index] for index in train) == {0, 1, 2}
    assert not set(train).intersection(validation)
    assert sorted(train + validation) == list(range(len(labels)))


def test_pathfinder_split_is_deterministic_80_10_10() -> None:
    keys = [f"folder/file_{index}.png/{index}" for index in range(100)]
    train, validation, test = pathfinder_split_indices(keys)
    assert (len(train), len(validation), len(test)) == (80, 10, 10)
    assert (train, validation, test) == pathfinder_split_indices(keys)
    assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))


def test_pathfinder_split_matches_tfds_hard_md5_order() -> None:
    keys = [f"sample-{index}" for index in range(10)]
    train, validation, test = pathfinder_split_indices(keys)
    assert train + validation + test == [7, 4, 2, 5, 0, 1, 3, 9, 6, 8]


def test_collate_uses_lengths_not_the_token_value() -> None:
    batch = collate_tokens(
        [(torch.tensor([0, 4]), 1), (torch.tensor([3]), 0)], pad_token_id=0
    )
    assert batch["mask"].tolist() == [[True, True], [True, False]]


@pytest.mark.parametrize(
    "mixer", ("mha", "memsolve", "linear_transformer", "performer", "nystromformer", "cosformer", "rebased")
)
@pytest.mark.parametrize("pooling", ("mean", "meanmax"))
def test_sequence_classifier_masks_padding_for_both_mixers(
    mixer: str, pooling: str
) -> None:
    torch.manual_seed(4)
    model = SequenceClassifier(_encoder(mixer), 3, pooling=pooling).eval()  # type: ignore[arg-type]
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    first = torch.tensor([[2, 3, 0, 0], [4, 0, 0, 0]])
    second = torch.tensor([[2, 3, 10, 9], [4, 8, 7, 6]])
    torch.testing.assert_close(model(first, mask), model(second, mask), rtol=0, atol=0)


@pytest.mark.parametrize("mixer", ("mha", "memsolve"))
def test_factorized_grid_positions_mask_invalid_pixels_for_both_mixers(mixer: str) -> None:
    torch.manual_seed(17)
    model = SequenceClassifier(_grid_encoder(mixer), 3, pooling="meanmax").eval()  # type: ignore[arg-type]
    valid = torch.tensor([[True, False, True, False, True, False]])
    first = torch.tensor([[[1.0], [0.0], [2.0], [0.0], [3.0], [0.0]]])
    second = torch.tensor([[[1.0], [8.0], [2.0], [7.0], [3.0], [6.0]]])
    torch.testing.assert_close(model(first, valid), model(second, valid), rtol=0, atol=0)


def test_meanmax_readout_handles_an_empty_valid_set() -> None:
    model = SequenceClassifier(_encoder("mha"), 3, pooling="meanmax")
    encoded = torch.randn(2, 3, 16)
    valid = torch.tensor([[False, False, False], [True, False, False]])
    pooled = model._pool(encoded, valid)
    assert pooled.shape == (2, 16)
    assert torch.isfinite(pooled).all()
    assert model.meanmax_projection is not None
    assert model.readout_projection is not None


def test_mean_readout_keeps_the_existing_parameterization() -> None:
    model = SequenceClassifier(_encoder("mha"), 3)
    assert model.meanmax_projection is None
    assert model.readout_projection is None


def test_pair_classifier_accepts_independent_lengths() -> None:
    model = SequencePairClassifier(_encoder("memsolve"), 2).eval()
    first = torch.tensor([[2, 3, 0], [4, 0, 0]])
    second = torch.tensor([[7, 6, 5], [3, 2, 0]])
    first_mask = first.ne(0)
    second_mask = second.ne(0)
    logits = model(first, first_mask, second, second_mask)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_meanmax_pooling_is_rejected_for_paired_tasks() -> None:
    args = resolve_args(
        parse_args(["--suite", "lra", "--task", "retrieval", "--pooling", "meanmax"])
    )
    dataset = TensorDataset(
        torch.ones(1, 4, dtype=torch.long), torch.zeros(1, dtype=torch.long)
    )
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="tokens",
        num_classes=2,
        max_length=4,
        metadata={},
        vocab_size=8,
        pad_token_id=0,
        paired=True,
    )
    with pytest.raises(ValueError, match="not supported for paired"):
        build_model(args, bundle)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mha_baseline_accepts_fp16_autocast_with_padding() -> None:
    model = SequenceClassifier(_encoder("mha"), 2).cuda().train()
    inputs = torch.tensor([[2, 3, 0], [4, 0, 0]], device="cuda")
    mask = inputs.ne(0)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(inputs, mask).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_memsolve_meanmax_readout_is_finite_with_masked_values() -> None:
    cuda.load()
    encoder = SequenceEncoder(
        input_kind="values",
        vocab_size=None,
        pad_token_id=None,
        max_length=8,
        dim=16,
        depth=1,
        num_heads=2,
        rank=16,
        mixer="memsolve",
        implementation="cuda",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
        grid_shape=(2, 4),
    )
    model = SequenceClassifier(encoder, 2, pooling="meanmax").cuda().train()
    inputs = torch.rand(2, 8, 1, device="cuda")
    valid = torch.tensor(
        [[True, True, False, False, False, False, False, False],
         [True, True, True, False, False, False, False, False]],
        device="cuda",
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(inputs, valid).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_memsolve_sequence_path_receives_fp16_amp_activations() -> None:
    cuda.load()
    encoder = SequenceEncoder(
        input_kind="tokens",
        vocab_size=12,
        pad_token_id=0,
        max_length=8,
        dim=16,
        depth=1,
        num_heads=2,
        rank=16,
        mixer="memsolve",
        implementation="cuda",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    ).cuda()
    observed: list[torch.dtype] = []
    hook = encoder.blocks[0].mixer.register_forward_pre_hook(
        lambda _module, arguments: observed.append(arguments[0].dtype)
    )
    inputs = torch.tensor([[2, 3, 0]], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = encoder(inputs, inputs.ne(0))
    hook.remove()
    assert torch.isfinite(outputs).all()
    assert observed == [torch.float16]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_memsolve_sequence_path_accepts_bf16_amp() -> None:
    cuda.load()
    encoder = SequenceEncoder(
        input_kind="tokens",
        vocab_size=12,
        pad_token_id=0,
        max_length=8,
        dim=16,
        depth=1,
        num_heads=2,
        rank=16,
        mixer="memsolve",
        implementation="cuda",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    ).cuda()
    inputs = torch.tensor([[2, 3, 0]], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = encoder(inputs, inputs.ne(0))
    assert torch.isfinite(outputs).all()


def test_listops_cache_is_current_data_only(tmp_path) -> None:
    source = tmp_path / "listops"
    source.mkdir()
    for split in ("train", "val", "test"):
        (source / f"{split}.tsv").write_text(
            "Source\tTarget\n[ MAX 1 2 ]\t2\n[ MIN 9 3 ]\t3\n",
            encoding="utf-8",
        )
    bundle = prepare_lra(
        "listops",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=16,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=None,
        allow_download=False,
        revision=None,
        formal=True,
    )
    assert len(bundle.train) == len(bundle.validation) == len(bundle.test) == 2
    tokens, label = bundle.train[0]
    assert tokens.dtype == torch.long
    assert label in (2, 3)
    assert bundle.metadata["tokenization"] == "official-listops-token-stream-with-eos"
    validate_formal_source_provenance(bundle)
    assert len(bundle.metadata["source"]["content_sha256"]) == 64


def test_packed_token_cache_serializes_a_concurrent_first_build(tmp_path) -> None:
    prefix = tmp_path / "cache" / "tokens"
    rows = [("first", 0), ("second", 1)]
    manifest = {"schema": 1, "case": "concurrent"}

    def build():
        return build_packed_tokens(
            prefix,
            rows,
            lambda text: [len(text)],
            manifest,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        datasets = [
            future.result()
            for future in (executor.submit(build), executor.submit(build))
        ]
    assert [len(dataset) for dataset in datasets] == [2, 2]
    assert not list(prefix.parent.glob("*.tmp"))


def test_retrieval_pair_cache_preserves_independent_documents(tmp_path) -> None:
    source = tmp_path / "aan"
    source.mkdir()
    for split, name in {
        "train": "new_aan_pairs.train.tsv",
        "val": "new_aan_pairs.eval.tsv",
        "test": "new_aan_pairs.test.tsv",
    }.items():
        del split
        (source / name).write_text(
            "1\tfirst\tsecond\tleft abstract\tright abstract\n",
            encoding="utf-8",
        )
    bundle = prepare_lra(
        "retrieval",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=16,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=None,
        allow_download=False,
        revision=None,
    )
    first, second, label = bundle.train[0]
    assert bundle.paired
    assert first.dtype == second.dtype == torch.long
    assert first.numel() and second.numel()
    assert label == 1


def test_pathfinder_bundle_uses_value_tokens_and_official_split(tmp_path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    root = tmp_path / "pathfinder" / "pathfinder2" / "curv_contour_length_14"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    lines = []
    for index in range(10):
        folder = root / f"group_{index}"
        folder.mkdir()
        filename = f"sample_{index}.png"
        pil_image.new("L", (2, 2), color=index).save(folder / filename)
        lines.append(f"group_{index} {filename} unused {index % 2}")
    (metadata / "0.npy").write_text("\n".join(lines), encoding="utf-8")
    bundle = prepare_lra(
        "pathfinder",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=None,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=2,
        allow_download=False,
        revision=None,
    )
    values, label = bundle.train[0]
    assert bundle.input_kind == "values"
    assert bundle.value_masking == "length"
    assert bundle.metadata["value_masking"] == "all-pixels"
    assert bundle.max_length == 4
    assert values.shape == (4, 1)
    assert label in (0, 1)


def test_pathfinder_arrays_preserve_author_splits_and_full_pixel_mask(tmp_path) -> None:
    import numpy as np
    root = tmp_path / "pathfinder32"
    root.mkdir()
    for split,count in (("train",8),("validation",3),("test",5)):
        np.save(root/f"{split}_images.npy", np.zeros((count,32,32),dtype=np.uint8))
        np.save(root/f"{split}_labels.npy", np.arange(count)%2)
    manifest={"difficulty":"hard","resolution":32,"split_protocol":"mega-author-splits-v1"}
    (root/"source-manifest.json").write_text(json.dumps(manifest))
    bundle = prepare_lra("pathfinder",data_root=tmp_path,cache_root=tmp_path/"cache",
        max_length=None,validation_fraction=None,split_seed=None,pathfinder_resolution=32,
        allow_download=False,revision=None,formal=True)
    assert (len(bundle.train),len(bundle.validation),len(bundle.test)) == (8,3,5)
    assert bundle.validation.labels == [0,1,0]
    assert bundle.metadata["split_protocol"] == "mega-author-splits-v1"
    validate_formal_source_provenance(bundle)
    batch = _build_collate(bundle)([bundle.train[0],bundle.train[1]])
    assert batch["mask"].all()
    assert batch["inputs"].shape == (2,1024,1)


def test_collate_values_can_exclude_zero_valued_pixels() -> None:
    batch = collate_values(
        [
            (torch.tensor([[0.0], [1.0], [0.0]]), 0),
            (torch.tensor([[2.0], [0.0]]), 1),
        ],
        value_masking="nonzero",
    )
    assert torch.equal(
        batch["mask"],
        torch.tensor([[False, True, False], [True, False, False]]),
    )
    assert batch["padding_ratio"] == pytest.approx(2.0 / 3.0)


def test_value_masking_contract_reaches_the_training_loader() -> None:
    dataset = TensorDataset(torch.zeros(1, 2, 1), torch.zeros(1, dtype=torch.long))
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="values",
        num_classes=2,
        max_length=2,
        metadata={},
        value_masking="nonzero",
    )
    batch = _build_collate(bundle)([(torch.tensor([[0.0], [3.0]]), 1)])
    assert torch.equal(batch["mask"], torch.tensor([[False, True]]))


def test_pathfinder_manifest_is_scoped_to_its_resolution(tmp_path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    root = tmp_path / "pathfinder" / "pathfinder2" / "curv_contour_length_14"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    lines = []
    for index in range(10):
        folder = root / f"group_{index}"
        folder.mkdir()
        filename = f"sample_{index}.png"
        pil_image.new("L", (2, 2), color=index).save(folder / filename)
        lines.append(f"group_{index} {filename} unused {index % 2}")
    (metadata / "0.npy").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "source-manifest.json").write_text(
        json.dumps(
            {
                "pathfinder32_hard": {
                    "metadata_files": 200,
                    "metadata_sha256": "32-only-signature",
                    "usable_rows": 200000,
                }
            }
        ),
        encoding="utf-8",
    )
    bundle = prepare_lra(
        "pathfinder",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=None,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=2,
        allow_download=False,
        revision=None,
    )
    assert bundle.metadata["resolution"] == 2


def test_pathx_is_fixed_to_pathfinder_128_with_its_own_identity(tmp_path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    root = tmp_path / "pathfinder" / "pathfinder128" / "curv_contour_length_14"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    lines = []
    for index in range(10):
        folder = root / f"group_{index}"
        folder.mkdir()
        filename = f"sample_{index}.png"
        pil_image.new("L", (128, 128), color=index).save(folder / filename)
        lines.append(f"group_{index} {filename} unused {index % 2}")
    (metadata / "0.npy").write_text("\n".join(lines), encoding="utf-8")
    bundle = prepare_lra(
        "pathx",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=None,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=None,
        allow_download=False,
        revision=None,
    )
    values, label = bundle.train[0]
    assert bundle.metadata["task"] == "pathx"
    assert bundle.metadata["resolution"] == 128
    assert bundle.metadata["split_protocol"] == PATHFINDER_SPLIT_PROTOCOL
    assert Path(bundle.metadata["source"]["path"]).name == "pathfinder128"
    assert (len(bundle.train), len(bundle.validation), len(bundle.test)) == (8, 1, 1)
    assert bundle.max_length == 128**2
    assert values.shape == (128**2, 1)
    assert label in (0, 1)


def test_resolved_pathx_delegates_its_fixed_resolution_to_the_data_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = resolve_args(parse_args(["--suite", "lra", "--task", "pathx"]))
    marker = object()
    observed: dict[str, object] = {}

    def fake_prepare_lra(task: str, **kwargs: object) -> object:
        observed["task"] = task
        observed.update(kwargs)
        return marker

    monkeypatch.setattr(train_transformers, "prepare_lra", fake_prepare_lra)
    assert train_transformers.build_bundle(args) is marker
    assert observed["task"] == "pathx"
    assert observed["pathfinder_resolution"] is None


@pytest.mark.parametrize("task,length,epochs,batch", [
    ("listops", 2048, 40, 64), ("text", 4096, 40, 64),
    ("retrieval", 4096, 20, 64), ("image", 1024, 200, 256),
    ("pathfinder", None, 200, 256),
])
def test_lra_defaults_match_the_shared_encoder_budget(task, length, epochs, batch) -> None:
    args = resolve_args(parse_args(["--suite", "lra", "--task", task]))
    assert (args.max_length, args.epochs, args.batch_size, args.eval_batch_size) == (
        length, epochs, batch, batch)
    assert (args.dim, args.depth, args.heads, args.rank, args.mlp_ratio) == (128, 4, 4, 32, 4.)
    assert (args.lr, args.warmup_ratio, args.weight_decay, args.dropout) == (3e-4, .05, .01, .1)
    assert (args.pooling, args.grad_accum, args.patience) == ("mean", 1, 0)


def test_pilot_epochs_preserves_the_scheduler_horizon_and_is_not_formal() -> None:
    args = resolve_args(
        parse_args(
            [
                "--suite",
                "lra",
                "--task",
                "pathfinder",
                "--epochs",
                "200",
                "--pilot-epochs",
                "50",
                "--validation-only",
            ]
        )
    )
    assert (args.epochs, args.pilot_epochs) == (200, 50)
    with pytest.raises(ValueError, match="requires --validation-only"):
        resolve_args(
            parse_args(
                [
                    "--suite",
                    "lra",
                    "--task",
                    "pathfinder",
                    "--pilot-epochs",
                    "1",
                ]
            )
        )
    with pytest.raises(ValueError, match="rejects --pilot-epochs"):
        resolve_args(
            parse_args(
                [
                    "--suite",
                    "lra",
                    "--task",
                    "pathfinder",
                    "--pilot-epochs",
                    "1",
                    "--validation-only",
                    "--formal",
                ]
            )
        )
    with pytest.raises(ValueError, match="must not exceed epochs"):
        resolve_args(
            parse_args(
                [
                    "--suite",
                    "lra",
                    "--task",
                    "pathfinder",
                    "--epochs",
                    "2",
                    "--pilot-epochs",
                    "3",
                    "--validation-only",
                ]
            )
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--max-length", "1024"), "--max-length is not supported for LRA Pathfinder"),
        (
            ("--validation-fraction", "0.2"),
            "--validation-fraction is not supported for LRA Pathfinder",
        ),
        (("--split-seed", "7"), "--split-seed is not supported for LRA Pathfinder"),
    ),
)
def test_pathfinder_rejects_inactive_data_options(
    arguments: tuple[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_args(
            parse_args(["--suite", "lra", "--task", "pathfinder", *arguments])
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--max-length", "1024"), "--max-length is not supported for LRA Path-X"),
        (
            ("--pathfinder-resolution", "32"),
            "--pathfinder-resolution is not supported for LRA Path-X",
        ),
        (
            ("--validation-fraction", "0.2"),
            "--validation-fraction is not supported for LRA Path-X",
        ),
        (("--split-seed", "7"), "--split-seed is not supported for LRA Path-X"),
    ),
)
def test_pathx_rejects_data_overrides(
    arguments: tuple[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_args(parse_args(["--suite", "lra", "--task", "pathx", *arguments]))


def test_pathfinder_metadata_excludes_inactive_data_options() -> None:
    args = resolve_args(
        parse_args(["--suite", "lra", "--task", "pathfinder", "--pooling", "meanmax"])
    )
    dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="values",
        num_classes=2,
        max_length=32**2,
        metadata={},
    )
    payload = _build_run_payload(
        args,
        bundle,
        {"train": 1, "validation": 1, "test": 1},
        torch.nn.Linear(1, 1),
        torch.device("cpu"),
        cuda_enabled=False,
    )
    resolved = payload["resolved_arguments"]
    assert resolved["pathfinder_resolution"] == 32
    assert resolved["pooling"] == "meanmax"
    assert payload["model"]["pooling"] == "meanmax"
    assert {"max_length", "validation_fraction", "split_seed"}.isdisjoint(resolved)


def test_pathfinder_model_uses_factorized_grid_positions_and_records_its_contract() -> None:
    args = resolve_args(
        parse_args(
            [
                "--suite", "lra", "--task", "pathfinder", "--pathfinder-resolution", "2",
                "--mixer", "mha",
            ]
        )
    )
    dataset = TensorDataset(torch.zeros(1, 4, 1), torch.zeros(1, dtype=torch.long))
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="values",
        num_classes=2,
        max_length=4,
        metadata={"resolution": 2},
    )
    model = build_model(args, bundle)
    assert isinstance(model, SequenceClassifier)
    assert model.encoder.grid_shape == (2, 2)
    assert model.encoder.position_embedding is None
    assert model.encoder.row_embedding is not None
    assert model.encoder.column_embedding is not None
    payload = _build_run_payload(
        args,
        bundle,
        {"train": 1, "validation": 1, "test": 1},
        model,
        torch.device("cpu"),
        cuda_enabled=False,
    )
    assert payload["model"]["mlp_ratio"] == 4.0
    assert payload["model"]["dropout"] == 0.1
    assert payload["model"]["bias"] is True
    assert payload["model"]["position_encoding"] == "factorized-grid-absolute"
    assert payload["model"]["position_initialization"] == "factorized-normal-0.02"


def test_pathx_run_metadata_keeps_the_fixed_task_identity() -> None:
    args = resolve_args(parse_args(["--suite", "lra", "--task", "pathx"]))
    dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="values",
        num_classes=2,
        max_length=128**2,
        metadata={"task": "pathx", "resolution": 128},
    )
    payload = _build_run_payload(
        args,
        bundle,
        {"train": 1, "validation": 1, "test": 1},
        torch.nn.Linear(1, 1),
        torch.device("cpu"),
        cuda_enabled=False,
    )
    resolved = payload["resolved_arguments"]
    assert resolved["task"] == "pathx"
    assert resolved["pathfinder_resolution"] == 128
    assert {"max_length", "validation_fraction", "split_seed"}.isdisjoint(resolved)
    assert payload["data_contract"]["value_masking"] == "length"
    assert payload["model"]["position_initialization"] == "normal-0.02"


def test_checkpoint_selection_is_independent_of_early_stop_tolerances() -> None:
    assert _is_better_validation_checkpoint(0.7001, 0.9, 0.7, 0.9)
    assert _is_better_validation_checkpoint(0.7, 0.89, 0.7, 0.9)
    assert _makes_early_stop_progress(
        0.7001,
        0.9,
        0.7,
        0.9,
        accuracy_delta=0.01,
        loss_relative_delta=0.01,
    ) == (False, False)


@pytest.mark.parametrize("task,batch,epochs", [("pathfinder", 256, 200), ("retrieval",64,20)])
def test_shared_lra_config_resolves_the_selected_task_defaults(task, batch, epochs) -> None:
    root = Path(__file__).resolve().parents[2]
    args = resolve_args(parse_args(["--config", str(root / "experiments/configs/lra.toml"), "--task", task]))
    assert (args.dim, args.depth, args.batch_size, args.grad_accum, args.epochs) == (128,4,batch,1,epochs)
    assert (args.execution, args.amp_dtype, args.min_lr_ratio) == ("compile","bfloat16",.01)


def test_sequence_toml_template_supplies_required_task_selection() -> None:
    root = Path(__file__).resolve().parents[2]
    args = resolve_args(parse_args(["--config", str(root / "experiments/configs/lra.toml")]))
    assert (args.suite, args.task, args.rank, args.lr) == ("lra", "listops", 32, 3e-4)


def test_lra_image_keeps_black_pixels_and_partitions_train_only(tmp_path) -> None:
    import numpy as np
    root = tmp_path / "cifar10"
    root.mkdir()
    for split, count in (("train", 40), ("test", 10)):
        np.save(root / f"{split}_images.npy", np.zeros((count,32,32), dtype=np.uint8))
        np.save(root / f"{split}_labels.npy", np.arange(count) % 10)
    bundle = prepare_lra("image", data_root=tmp_path, cache_root=tmp_path/"cache",
        max_length=1024, validation_fraction=.25, split_seed=0,
        pathfinder_resolution=None, allow_download=False, revision=None)
    assert (len(bundle.train), len(bundle.validation), len(bundle.test)) == (30,10,10)
    batch = _build_collate(bundle)([bundle.train[0], bundle.train[1]])
    assert batch["mask"].all()
    assert bundle.metadata["resolution"] == 32
    args = resolve_args(parse_args(["--suite","lra","--task","image","--mixer","mha"]))
    assert build_model(args,bundle).encoder.grid_shape == (32,32)


def test_sdpa_mha_matches_torch_attention_with_padding() -> None:
    from experiments.train_transformers import MaskedMultiheadAttention
    torch.manual_seed(17)
    mixer = MaskedMultiheadAttention(32,4,bias=True)
    x = torch.randn(3,9,32, requires_grad=True)
    mask = torch.arange(9)[None,:] < torch.tensor([9,5,1])[:,None]
    actual = mixer(x,mask)
    expected = mixer.attention(x,x,x,key_padding_mask=~mask,need_weights=False)[0]
    expected = torch.where(mask[:,:,None],expected,torch.zeros_like(expected))
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    upstream=torch.randn_like(actual)
    ga=torch.autograd.grad(actual,x,upstream,retain_graph=True)[0]
    ge=torch.autograd.grad(expected,x,upstream)[0]
    torch.testing.assert_close(ga,ge,atol=2e-6,rtol=2e-5)
    assert mixer(x,torch.zeros_like(mask)).count_nonzero() == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mixer,position_encoding", [("memsolve", "absolute"),
    ("mha", "absolute"), ("memsolve", "rove_1d"), ("memsolve", "rove_2d"),
    ("mha", "rove_1d"), ("mha", "rove_2d")])
def test_bf16_compiled_encoder_matches_eager_outputs_and_gradients(mixer, position_encoding) -> None:
    import copy
    import torch._functorch.config as functorch_config
    from memsolve.ball import MemSolve
    functorch_config.backward_pass_autocast = "off"
    torch.manual_seed(12)
    if mixer == "memsolve":
        cuda.load()
    grid = (8,16) if position_encoding == "rove_2d" else None
    encoder = SequenceEncoder(input_kind="values" if grid else "tokens", vocab_size=20, pad_token_id=0,
        max_length=128, dim=128, depth=1, num_heads=4, rank=32, mixer=mixer,
        implementation="cuda", mlp_ratio=4., dropout=0., bias=True,
        grid_shape=grid, position_encoding=position_encoding)
    eager = SequenceClassifier(encoder, 3).cuda()
    compiled = copy.deepcopy(eager)
    for module in compiled.modules():
        if isinstance(module, MemSolve):
            module.forward = torch.compiler.disable(module.forward)
    compiled.compile(dynamic=True, fullgraph=False,
        options={"triton.cudagraphs":False, "emulate_precision_casts":True})
    inputs = torch.randn(3,128,1,device="cuda") if grid else torch.randint(1,20,(3,128),device="cuda")
    mask = torch.arange(128,device="cuda")[None,:] < torch.tensor([128,73,0],device="cuda")[:,None]
    gradients = []
    outputs = []
    for model in (eager, compiled):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output = model(inputs,mask)
            loss = output.float().square().sum()
        loss.backward()
        outputs.append(output)
        gradients.append(torch.cat([p.grad.float().flatten() for p in model.parameters()
                                     if p.grad is not None]))
    assert all(torch.isfinite(g).all() for g in gradients)
    torch.testing.assert_close(outputs[0],outputs[1],atol=.004,rtol=.02)
    relative_error = (gradients[0]-gradients[1]).norm()/gradients[0].norm().clamp_min(1e-12)
    assert relative_error.item() < .01


@pytest.mark.parametrize("mixer", ("memsolve", "mha"))
def test_fixed_padding_preserves_valid_token_outputs_and_gradients(mixer) -> None:
    import copy
    from experiments.sequence_data import collate_token_pairs
    rows = [(torch.tensor([2,3,4]),0),(torch.tensor([5,6]),1)]
    short = collate_tokens(rows,pad_token_id=0)
    padded = collate_tokens(rows,pad_token_id=0,pad_length=8)
    assert padded["inputs"].shape == (2,8)
    assert padded["mask"].sum() == 5
    pair = collate_token_pairs([(a,a,b) for a,b in rows],pad_token_id=0,pad_length=8)
    assert pair["first"].shape == pair["second"].shape == (2,8)
    assert pair["first_mask"].sum() == pair["second_mask"].sum() == 5
    first = SequenceClassifier(_encoder(mixer),2).eval()
    second = copy.deepcopy(first)
    y1 = first(short["inputs"],short["mask"])
    y2 = second(padded["inputs"],padded["mask"])
    torch.testing.assert_close(y1,y2,atol=1e-6,rtol=1e-5)
    y1.square().sum().backward(); y2.square().sum().backward()
    for p1,p2 in zip(first.parameters(),second.parameters()):
        if p1.grad is not None:torch.testing.assert_close(p1.grad,p2.grad,atol=1e-6,rtol=1e-5)


class _TinyTokenDataset(Dataset):
    labels = [0, 1, 0, 1]
    lengths = [3, 2, 3, 2]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        values = (torch.tensor([2, 3, 4]), torch.tensor([3, 4]))[index % 2]
        return values, self.labels[index]


@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize("growth_interval", (1, 1000))
@pytest.mark.parametrize("grad_accum", (1, 3))
def test_sequence_scheduler_tracks_successful_amp_updates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fused: bool,
    growth_interval: int,
    grad_accum: int,
) -> None:
    """Real CPU GradScaler exercises skipped updates, including fused AdamW."""
    torch.manual_seed(31)
    device = torch.device("cpu")
    loader = make_loader(
        _TinyTokenDataset(), batch_size=1, workers=0, device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=True, seed=31,
    )
    model = SequenceClassifier(_encoder("mha"), 2)
    parameter = next(model.parameters())
    backward_calls = 0

    def overflow_first_and_third_update(gradient):
        nonlocal backward_calls
        backward_calls += 1
        if backward_calls in (1, 2 * grad_accum + 1):
            return torch.full_like(gradient, torch.inf)
        return gradient

    parameter.register_hook(overflow_first_and_third_update)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0, growth_interval=growth_interval)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: scaler)
    successful_update_lrs = []
    optimizer_calls = 0
    adamw = torch.optim.AdamW

    class RecordingAdamW(adamw):
        def __init__(self, *args, **kwargs):
            kwargs["fused"] = fused
            super().__init__(*args, **kwargs)

        def step(self, closure=None):
            nonlocal optimizer_calls
            optimizer_calls += 1
            before = parameter.detach().clone()
            lr = self.param_groups[0]["lr"]
            result = super().step(closure)
            if not torch.equal(parameter, before):
                successful_update_lrs.append(lr)
            return result

    monkeypatch.setattr(torch.optim, "AdamW", RecordingAdamW)
    config = TrainingConfig(
        output=tmp_path / "run", epochs=1, lr=1e-3, weight_decay=0.0,
        warmup_ratio=1.0, min_lr_ratio=0.0, grad_accum=grad_accum,
        grad_clip=1.0, patience=0, early_stop_min_epochs=0,
        early_stop_accuracy_delta=0.0, early_stop_loss_relative_delta=0.0,
        seed=31, resume=False, validation_only=True,
        max_train_batches=0, max_eval_batches=0, amp=True,
    )
    train(
        model, loader, loader, loader, num_classes=2, config=config,
        run_payload={"case": "amp-scheduler"}, device=device,
    )
    checkpoint = torch.load(config.output / "last.pt", map_location="cpu", weights_only=False)
    attempts = math.ceil(len(loader) / grad_accum)
    successful_updates = attempts - (2 if attempts >= 3 else 1)
    assert backward_calls == len(loader)
    assert checkpoint["scheduler"]["last_epoch"] == successful_updates
    assert successful_update_lrs == pytest.approx(
        [config.lr * (step + 1) / attempts for step in range(successful_updates)]
    )
    # The fused optimizer is called even when its kernel skips the update.
    assert optimizer_calls == (attempts if fused else successful_updates)
    for state in checkpoint["optimizer"]["state"].values():
        assert state["step"].item() == successful_updates


def test_pilot_epochs_caps_the_outer_loop_without_shortening_cosine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _TinyTokenDataset()
    device = torch.device("cpu")
    loader = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=True,
        seed=17,
    )
    validation = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=False,
        seed=17,
    )
    recorded_total_steps: list[int] = []
    scheduler_lambda = train_transformers._scheduler_lambda

    def record_scheduler_horizon(
        step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float
    ) -> float:
        recorded_total_steps.append(total_steps)
        return scheduler_lambda(step, warmup_steps, total_steps, min_lr_ratio)

    monkeypatch.setattr(train_transformers, "_scheduler_lambda", record_scheduler_horizon)
    config = TrainingConfig(
        output=tmp_path / "run",
        epochs=4,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        grad_accum=1,
        grad_clip=0.0,
        patience=0,
        early_stop_min_epochs=4,
        early_stop_accuracy_delta=0.0,
        early_stop_loss_relative_delta=0.0,
        seed=17,
        resume=False,
        validation_only=True,
        max_train_batches=0,
        max_eval_batches=0,
        amp=False,
        pilot_epochs=2,
    )
    result = train(
        SequenceClassifier(_encoder("mha"), 2),
        loader,
        validation,
        validation,
        num_classes=2,
        config=config,
        run_payload={"case": "pilot"},
        device=device,
    )
    metrics = (config.output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 2
    assert set(recorded_total_steps) == {len(loader) * config.epochs}
    assert result["target"] == "validation"


def test_checkpoint_refuses_configuration_mismatch_without_rewriting_manifest(tmp_path) -> None:
    dataset = _TinyTokenDataset()
    device = torch.device("cpu")
    loader = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=True,
        seed=3,
    )
    validation = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=False,
        seed=3,
    )
    output = tmp_path / "run"
    config = TrainingConfig(
        output=output,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        grad_accum=1,
        grad_clip=0.0,
        patience=0,
        early_stop_min_epochs=0,
        early_stop_accuracy_delta=0.0,
        early_stop_loss_relative_delta=0.0,
        seed=3,
        resume=False,
        validation_only=True,
        max_train_batches=0,
        max_eval_batches=0,
        amp=False,
    )
    train(
        SequenceClassifier(_encoder("mha"), 2),
        loader,
        validation,
        validation,
        num_classes=2,
        config=config,
        run_payload={"case": "first"},
        device=device,
    )
    manifest = (output / "config.json").read_text(encoding="utf-8")
    mismatch = TrainingConfig(**{**config.__dict__, "resume": True})
    with pytest.raises(RuntimeError, match="configuration does not match"):
        train(
            SequenceClassifier(_encoder("mha"), 2),
            loader,
            validation,
            validation,
            num_classes=2,
            config=mismatch,
            run_payload={"case": "different"},
            device=device,
        )
    assert (output / "config.json").read_text(encoding="utf-8") == manifest


def test_checkpoint_allows_explicit_resume_for_the_same_run(tmp_path) -> None:
    dataset = _TinyTokenDataset()
    device = torch.device("cpu")
    loader = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=True,
        seed=5,
    )
    validation = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=False,
        seed=5,
    )
    config = TrainingConfig(
        output=tmp_path / "run",
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        grad_accum=1,
        grad_clip=0.0,
        patience=0,
        early_stop_min_epochs=0,
        early_stop_accuracy_delta=0.0,
        early_stop_loss_relative_delta=0.0,
        seed=5,
        resume=False,
        validation_only=True,
        max_train_batches=0,
        max_eval_batches=0,
        amp=False,
    )
    first = train(
        SequenceClassifier(_encoder("mha"), 2),
        loader,
        validation,
        validation,
        num_classes=2,
        config=config,
        run_payload={
            "case": "resume",
            "resolved_arguments": {"resume": False, "task": "test"},
        },
        device=device,
    )
    metrics_before = (config.output / "metrics.jsonl").read_text(encoding="utf-8")
    manifest_before = (config.output / "config.json").read_text(encoding="utf-8")
    result = train(
        SequenceClassifier(_encoder("mha"), 2),
        loader,
        validation,
        validation,
        num_classes=2,
        config=TrainingConfig(**{**config.__dict__, "resume": True}),
        run_payload={
            "case": "resume",
            "resolved_arguments": {"resume": True, "task": "test"},
        },
        device=device,
    )
    assert result == first
    assert (config.output / "metrics.jsonl").read_text(encoding="utf-8") == metrics_before
    assert (config.output / "config.json").read_text(encoding="utf-8") == manifest_before
    state = torch.load(config.output / "last.pt", map_location="cpu", weights_only=False)
    assert state["completed"]
    assert state["format_version"] == 2
    assert "early_stop_best_accuracy" in state


def test_mha_rope_matches_explicit_rotation_and_preserves_initialization():
    cls = train_transformers.MaskedMultiheadAttention
    torch.manual_seed(19)
    baseline = cls(8, 2, bias=True).double()
    torch.manual_seed(19)
    rotated = cls(8, 2, bias=True, rope=True).double()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, rotated.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [False]*5])
    actual = rotated(x, mask)
    attn = rotated.attention
    q, k, v = F.linear(x[:1], attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
    q, k, v = [t.reshape(1, 5, 2, 4).transpose(1, 2) for t in (q, k, v)]
    matrices = []
    for position in range(5):
        blocks = []
        for frequency in (1.0, 0.01):
            angle = position * frequency
            blocks.append(torch.tensor([[math.cos(angle), -math.sin(angle)],
                                        [math.sin(angle), math.cos(angle)]], dtype=torch.float64))
        matrices.append(torch.block_diag(*blocks))
    rotation = torch.stack(matrices)
    q, k = [torch.einsum('nij,bhnj->bhni', rotation, t) for t in (q, k)]
    scores = (q @ k.transpose(-2, -1)) / 2
    scores = scores.masked_fill(~mask[:1, None, None, :], -torch.inf)
    expected = attn.out_proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(1, 5, 8))
    expected = expected * mask[:1, :, None]
    torch.testing.assert_close(actual[:1], expected, rtol=1e-10, atol=1e-10)
    assert torch.count_nonzero(actual[1]) == 0
    grad_a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    grad_b = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(grad_a, grad_b, rtol=1e-10, atol=1e-10)
    # Position zero has identity rotation, including the original MHA gradient.
    one = x.detach()[:, :1].requires_grad_()
    valid = torch.ones(2, 1, dtype=torch.bool)
    torch.testing.assert_close(rotated(one, valid), baseline(one, valid), rtol=1e-10, atol=1e-10)


def test_nystromformer_lra_recipe_preserves_published_step_budgets() -> None:
    from experiments.train_lra import RECIPES
    assert {task:(r.steps,r.batch,r.warmup,r.eval_every) for task,r in RECIPES.items()} == {
        'listops':(5000,32,1000,50), 'text':(20000,32,8000,500),
        'retrieval':(30000,32,800,300), 'image':(35000,256,175,175),
        'pathfinder':(62400,256,312,312)}
    assert RECIPES['text'].content_length == RECIPES['retrieval'].content_length == 4000


def test_lra_step_sampler_resumes_across_epoch_and_batch_boundaries() -> None:
    from experiments.train_lra import StepBatchSampler
    all_batches=list(StepBatchSampler(13,5,12,7))
    assert all(len(batch)==5 for batch in all_batches)
    for start in (1,3,7,11):
        assert list(StepBatchSampler(13,5,12,7,start)) == all_batches[start:]
    assert sorted(sum(all_batches,[])[:13]) == list(range(13))
    assert sorted(sum(all_batches,[])[13:26]) == list(range(13))


def test_nystromformer_token_conversion_removes_eos_and_preserves_legacy_masks() -> None:
    from experiments.train_lra import TokenDataset,collate
    text=TokenDataset([(torch.tensor([100,101,2]),1)],'text',4000)
    assert text[0][0].tolist() == [98,99]
    pixels=TokenDataset([(torch.tensor([[0.],[1.],[128/255]]),0)],'image',1024)
    converted=pixels[0]
    assert converted[0].tolist() == [0,255,128]
    batch=collate([converted],length=4,paired=False)
    assert batch['mask'].tolist() == [[False,True,True,False]]


def test_nystromformer_lra_fixed_mean_and_signed_pair_head() -> None:
    from experiments.train_lra import LRAModel
    torch.manual_seed(5)
    model=LRAModel(8,32,2,paired=True,rank=4,implementation='reference').eval()
    inputs=torch.tensor([[3,4,0,0,0,0,0,0]])
    mask=inputs.ne(0)
    with torch.no_grad():
        x=model.word_embeddings(inputs)+model.position_embeddings(torch.arange(8))[None]
        for block in model.blocks:x=block(x,mask)
        pooled=(model.norm(x)*mask[:,:,None]).mean(1)
        torch.testing.assert_close(model.encode(inputs,mask),pooled)
        other_inputs=torch.tensor([[5,6,7,0,0,0,0,0]])
        other=model.encode(other_inputs,other_inputs.ne(0))
        expected=model.classifier(torch.cat((pooled,other,pooled*other,pooled-other),-1))
        torch.testing.assert_close(model(inputs,mask,other_inputs,other_inputs.ne(0)),expected)
    loss=model(inputs,mask,other_inputs,other_inputs.ne(0)).square().sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["single", "paired", "image"])
@pytest.mark.parametrize("mixer", ["memsolve", "mha"])
def test_sequence_graph_updates_accumulation_and_tail(kind, mixer):
    import copy
    import torch._functorch.config as fc
    from memsolve import MemSolve
    from experiments.train_transformers import SequenceTrainingGraph, forward_batch
    fc.backward_pass_autocast = "off"
    cuda.load()
    torch.manual_seed(42)
    encoder = SequenceEncoder(input_kind="values" if kind == "image" else "tokens",
        vocab_size=20, pad_token_id=0, max_length=32, dim=128, depth=2,
        num_heads=4, rank=16, mixer=mixer, implementation="cuda",
        mlp_ratio=2., dropout=0., bias=True,
        grid_shape=(4,8) if kind == "image" else None,
        position_encoding="rove_2d" if kind == "image" else "rove_1d")
    cls = train_transformers.SequencePairClassifier if kind == "paired" else SequenceClassifier
    eager = cls(encoder, 2).cuda().train()
    captured = copy.deepcopy(eager)
    if mixer == "mha":
        for module in captured.modules():
            if isinstance(module, train_transformers.MaskedMultiheadAttention):
                module.enable_flex_attention()
    for module in captured.modules():
        if isinstance(module, MemSolve):
            module.forward = torch.compiler.disable(module.forward)
    captured.compile(dynamic=False, fullgraph=False,
        options={"triton.cudagraphs":False,"emulate_precision_casts":True})
    def batch(size):
        x = (torch.rand(size,32,1,device="cuda") if kind == "image" else
             torch.randint(1,20,(size,32),device="cuda"))
        mask = torch.rand(size,32,device="cuda") > .3
        mask[0] = False
        b = {"inputs":x,"mask":mask,"labels":torch.randint(0,2,(size,),device="cuda")}
        if kind == "paired":
            b = {"first":x,"first_mask":mask,"second":x.flip(1).contiguous(),
                 "second_mask":mask.flip(1).contiguous(),"labels":b['labels']}
        return b
    graph = SequenceTrainingGraph(captured, batch(3), amp=True)
    optimizers = [torch.optim.SGD(m.parameters(),lr=.001) for m in [eager,captured]]
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for size in [3,3,2]:
            b = batch(size)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                reference_loss = F.cross_entropy(forward_batch(eager,b),b['labels'],reduction='sum')
            reference_loss.backward()
            if graph.matches(b):
                actual_loss = graph.backward(b).clone()
            else:
                assert size == 2
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    actual_loss = F.cross_entropy(forward_batch(captured,b),b['labels'],reduction='sum')
                actual_loss.backward()
            torch.testing.assert_close(actual_loss,reference_loss,atol=.01,rtol=.01)
        ga=torch.cat([p.grad.flatten() for p in eager.parameters() if p.grad is not None])
        gb=torch.cat([p.grad.flatten() for p in captured.parameters() if p.grad is not None])
        assert torch.isfinite(gb).all()
        assert (ga-gb).norm()/ga.norm() < .01
        for model,optimizer in zip([eager,captured],optimizers):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(8)
            optimizer.step()
        for a,b in zip(eager.parameters(),captured.parameters()):
            torch.testing.assert_close(a,b,atol=3e-5,rtol=1e-3)
    assert graph.replays == 6


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_model", [False, True])
def test_sequence_graph_dropout_rng_and_grad_overwrite(compile_model):
    from experiments.train_transformers import SequenceTrainingGraph
    cuda.load()
    torch.manual_seed(91)
    encoder=SequenceEncoder(input_kind='tokens',vocab_size=20,pad_token_id=0,
        max_length=32,dim=64,depth=1,num_heads=2,rank=16,mixer='memsolve',
        implementation='cuda',mlp_ratio=2.,dropout=.3,bias=True)
    model=SequenceClassifier(encoder,2).cuda().train()
    if compile_model:
        import torch._functorch.config as fc
        from memsolve import MemSolve
        fc.backward_pass_autocast = "off"
        for module in model.modules():
            if isinstance(module, MemSolve):
                module.forward = torch.compiler.disable(module.forward)
        model.compile(dynamic=False,fullgraph=False,
            options={"triton.cudagraphs":False,"emulate_precision_casts":True})
    batch={'inputs':torch.randint(1,20,(4,32),device='cuda'),
           'mask':torch.ones(4,32,device='cuda',dtype=torch.bool),
           'labels':torch.tensor([0,1,0,1],device='cuda')}
    cpu_rng,gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
    graph=SequenceTrainingGraph(model,batch,amp=True)
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
    losses=[];gradients=[]
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        losses.append(graph.backward(batch).clone())
        gradients.append(model.classifier.weight.grad.clone())
    assert not torch.equal(losses[0],losses[1])
    assert not torch.equal(gradients[0],gradients[1])
    assert all(torch.isfinite(x).all() for x in gradients)


@pytest.mark.parametrize('axial', [False, True])
def test_mha_rove_matches_explicit_pairwise_rotations(axial):
    from memsolve.ball.reference import rotary_position_factors, axial_rotary_position_factors
    from experiments.train_transformers import MaskedMultiheadAttention
    torch.manual_seed(28)
    model = MaskedMultiheadAttention(16, 2, bias=True).double()
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True, False, True, True, False], [False]*6])
    positions = torch.arange(6)[None]
    if axial:
        coords = torch.stack((positions % 3, positions // 3), -1)
        cosine, sine = axial_rotary_position_factors(coords, head_dim=8)
    else:
        cosine, sine = rotary_position_factors(positions, head_dim=8, base=10000.,
            train_length=6, context_length=6, scaling='none')
    cosine, sine = cosine.double(), sine.double()
    actual = model(x, valid, (cosine, sine))
    q, k, v = F.linear(x, model.attention.in_proj_weight, model.attention.in_proj_bias).chunk(3, -1)
    q, k, v = [t.reshape(2, 6, 2, 8).transpose(1, 2) for t in (q, k, v)]
    rotations = torch.zeros(6, 8, 8, dtype=torch.float64)
    for i in range(4):
        rotations[:, 2*i, 2*i] = cosine[0, :, i]
        rotations[:, 2*i+1, 2*i+1] = cosine[0, :, i]
        rotations[:, 2*i, 2*i+1] = -sine[0, :, i]
        rotations[:, 2*i+1, 2*i] = sine[0, :, i]
    weights = ((q[:1] @ k[:1].transpose(-2, -1))/math.sqrt(8)).masked_fill(~valid[:1,None,None,:], -torch.inf).softmax(-1)
    # Each pair transports V_j by R_i^T R_j; Q/K scores remain unchanged.
    relative = rotations.transpose(-2, -1)[:, None] @ rotations[None]
    transported = torch.einsum('ijac,bhjc->bhija', relative, v[:1])
    content = (weights[...,None] * transported).sum(-2)
    expected = model.attention.out_proj(content.transpose(1,2).reshape(1,6,16))
    expected = torch.where(valid[:1,:,None], expected, torch.zeros_like(expected))
    torch.testing.assert_close(actual[:1], expected, rtol=1e-10, atol=1e-10)
    assert actual[1].count_nonzero() == 0
    parameters = (x, *model.parameters())
    ga = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), parameters)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    identity = (torch.ones_like(cosine), torch.zeros_like(sine))
    torch.testing.assert_close(model(x, valid, identity), model(x, valid), rtol=0, atol=0)
