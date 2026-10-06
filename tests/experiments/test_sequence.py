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
from experiments.certificate_diagnostics import summarize
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


def test_certificate_summary_groups_length_layer_and_reports_percentiles() -> None:
    records = []
    for length in (1024, 2048):
        for layer in (1, 2):
            for index, value in enumerate((0.2, 0.4, 0.6)):
                record: dict[str, float | int] = {
                    "length": length,
                    "layer": layer,
                    "checkpoint": 0,
                    "seed": 0,
                    "probe": index,
                    "head": 0,
                }
                for metric in (
                    "q",
                    "contraction_slack",
                    "mu",
                    "state_ratio",
                    "adjoint_ratio",
                    "state_bound_usage",
                    "adjoint_bound_usage",
                ):
                    record[metric] = value
                records.append(record)

    rows = summarize(records)

    assert len(rows) == 4 * 7
    q_row = next(
        row
        for row in rows
        if row["length"] == 1024 and row["layer"] == 1 and row["metric"] == "q"
    )
    assert q_row["count"] == 3
    assert q_row["min"] == pytest.approx(0.2)
    assert q_row["median"] == pytest.approx(0.4)
    assert q_row["max"] == pytest.approx(0.6)


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
    assert metadata["memsolve_cuda_contract"] == cuda._CUDA_CONTRACT_VERSION


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


@pytest.mark.parametrize(
    "mixer", ("linear_transformer", "performer", "nystromformer", "cosformer", "rebased")
)
def test_dna_baseline_forward_backward_is_finite(mixer: str) -> None:
    torch.manual_seed(23)
    model = SequenceClassifier(_encoder(mixer), 3).train()
    inputs = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    valid = inputs.ne(0)
    loss = model(inputs, valid).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mixer", ("linear_transformer", "performer", "nystromformer", "cosformer", "rebased")
)
def test_dna_baseline_accepts_fp16_cuda_autocast(mixer: str) -> None:
    torch.manual_seed(23)
    model = SequenceClassifier(_encoder(mixer), 3).cuda().train()
    inputs = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], device="cuda")
    valid = inputs.ne(0)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(inputs, valid).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mixer", ("linear_transformer", "performer"))
def test_fp32_attention_core_matches_cuda_amp_at_long_length(mixer: str) -> None:
    torch.manual_seed(31)
    constructor = {
        "linear_transformer": train_transformers.MaskedLinearTransformerAttention,
        "performer": train_transformers.MaskedPerformerAttention,
    }[mixer]
    layer = constructor(64, 4, bias=True).cuda().eval()
    inputs = (3 * torch.randn(2, 2048, 64, device="cuda")).requires_grad_()
    valid = torch.ones(2, 2048, dtype=torch.bool, device="cuda")
    valid[1, 1536:] = False
    cotangent = torch.randn_like(inputs)
    reference = layer(inputs, valid)
    reference_gradient = torch.autograd.grad(
        (reference * cotangent).sum(), inputs
    )[0]

    amp_inputs = inputs.detach().clone().requires_grad_()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        actual = layer(amp_inputs.half(), valid)
    actual_gradient = torch.autograd.grad(
        (actual.float() * cotangent).sum(), amp_inputs
    )[0]
    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual.float(), reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        actual_gradient, reference_gradient, rtol=3e-3, atol=3e-3
    )


@pytest.mark.parametrize(
    "mixer", ("linear_transformer", "performer", "nystromformer", "cosformer", "rebased")
)
def test_variable_length_baseline_is_independent_of_batch_padding(mixer: str) -> None:
    torch.manual_seed(29)
    model = SequenceClassifier(_encoder(mixer), 3).eval()
    short = torch.tensor([[2, 3]])
    short_mask = torch.ones_like(short, dtype=torch.bool)
    padded = torch.tensor([[2, 3, 9, 8]])
    padded_mask = torch.tensor([[True, True, False, False]])
    torch.testing.assert_close(
        model(short, short_mask),
        model(padded, padded_mask),
        rtol=2e-5,
        atol=2e-6,
    )


def test_linear_transformer_matches_explicit_quadratic_form() -> None:
    torch.manual_seed(41)
    layer = train_transformers.MaskedLinearTransformerAttention(16, 2, bias=True).double()
    inputs = torch.randn(2, 7, 16, dtype=torch.double)
    valid = torch.tensor(
        [[True, True, True, True, True, False, False],
         [True, True, True, False, False, False, False]]
    )
    actual = layer(inputs, valid)
    batch, length, _ = inputs.shape
    reshape = lambda value: value.reshape(batch, length, 2, 8).transpose(1, 2)
    q = F.elu(reshape(layer.q_proj(inputs))) + 1
    k = F.elu(reshape(layer.k_proj(inputs))) + 1
    v = reshape(layer.v_proj(inputs))
    k = k * valid[:, None, :, None]
    v = v * valid[:, None, :, None]
    weights = torch.einsum("bhid,bhjd->bhij", q, k)
    expected = torch.einsum("bhij,bhjd->bhid", weights, v)
    expected = expected / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    expected = layer.out_proj(expected.transpose(1, 2).reshape(batch, length, -1))
    expected = torch.where(valid[:, :, None], expected, torch.zeros_like(expected))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def _cloned_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone().requires_grad_()
        for name, parameter in module.named_parameters()
    }


def _projection(
    inputs: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    name: str,
) -> torch.Tensor:
    return F.linear(
        inputs,
        parameters[f"{name}.weight"],
        parameters.get(f"{name}.bias"),
    )


def _assert_upstream_forward_and_gradients(
    module: nn.Module,
    oracle: Any,
    inputs: torch.Tensor,
    valid: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    actual_inputs = inputs.detach().clone().requires_grad_()
    actual = module(actual_inputs, valid)
    cotangent = torch.randn_like(actual)
    actual_parameters = tuple(module.parameters())
    actual_gradients = torch.autograd.grad(
        (actual * cotangent).sum(), (actual_inputs, *actual_parameters)
    )

    expected_inputs = inputs.detach().clone().requires_grad_()
    expected_parameters = _cloned_parameters(module)
    expected = oracle(expected_inputs, valid, expected_parameters)
    expected_gradients = torch.autograd.grad(
        (expected * cotangent).sum(),
        (expected_inputs, *expected_parameters.values()),
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=rtol,
            atol=atol,
        )


def _performer_114_oracle(
    inputs: torch.Tensor,
    valid: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    projection_matrix: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    """performer-pytorch 1.1.4, commit fc8b784, noncausal FastAttention."""
    batch, length, dim = inputs.shape
    head_dim = dim // num_heads
    reshape = lambda value: value.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    q = reshape(_projection(inputs, parameters, "q_proj"))
    k = reshape(_projection(inputs, parameters, "k_proj"))
    v = reshape(_projection(inputs, parameters, "v_proj"))
    normalizer = head_dim ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5

    def feature_map(value: torch.Tensor, *, query: bool) -> torch.Tensor:
        projected = torch.einsum(
            "bhnd,md->bhnm", value * normalizer, projection_matrix.to(value)
        )
        diagonal = value.square().sum(dim=-1, keepdim=True) * normalizer**2 / 2
        dimensions = -1 if query else (-2, -1)
        maximum = projected.amax(dim=dimensions, keepdim=True).detach()
        return ratio * (torch.exp(projected - diagonal - maximum) + 1e-4)

    q = feature_map(q, query=True)
    k = feature_map(k, query=False)
    key_mask = valid[:, None, :, None]
    k = k * key_mask
    v = v * key_mask
    inverse = torch.einsum("bhnm,bhm->bhn", q, k.sum(dim=2)).reciprocal()
    context = torch.einsum("bhnm,bhnd->bhmd", k, v)
    output = torch.einsum("bhmd,bhnm,bhn->bhnd", context, q, inverse)
    output = output.transpose(1, 2).reshape(batch, length, dim)
    output = _projection(output, parameters, "out_proj")
    return torch.where(valid[:, :, None], output, torch.zeros_like(output))


def test_performer_matches_upstream_114_forward_and_gradients() -> None:
    torch.manual_seed(53)
    layer = train_transformers.MaskedPerformerAttention(16, 2, bias=True).double()
    inputs = torch.randn(2, 7, 16, dtype=torch.double)
    valid = torch.ones(2, 7, dtype=torch.bool)
    oracle = functools.partial(
        _performer_114_oracle,
        projection_matrix=layer.projection_matrix.detach().clone(),
        num_heads=2,
    )
    _assert_upstream_forward_and_gradients(
        layer, oracle, inputs, valid, rtol=1e-11, atol=1e-12
    )


def _fla_rebased_reference_oracle(
    inputs: torch.Tensor,
    valid: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    num_heads: int,
    feature_dim: int,
    eps: float,
) -> torch.Tensor:
    """FLA ReBasedLinearAttention forward_reference with causal=False."""
    batch, length, dim = inputs.shape
    head_dim = dim // num_heads
    q = _projection(inputs, parameters, "q_proj").reshape(
        batch, length, num_heads, feature_dim
    ).transpose(1, 2)
    k = _projection(inputs, parameters, "k_proj").reshape(
        batch, length, num_heads, feature_dim
    ).transpose(1, 2)
    v = _projection(inputs, parameters, "v_proj").reshape(
        batch, length, num_heads, head_dim
    ).transpose(1, 2)

    def feature_map(value: torch.Tensor) -> torch.Tensor:
        value = F.layer_norm(
            value,
            (feature_dim,),
            parameters["gamma"],
            parameters["beta"],
        )
        outer = torch.einsum("...i,...j->...ij", value, value)
        indices = torch.triu_indices(feature_dim, feature_dim, 1, device=value.device)
        return torch.cat(
            (
                outer.diagonal(dim1=-2, dim2=-1) * feature_dim**-0.5,
                outer[..., indices[0], indices[1]] * (2 / feature_dim) ** 0.5,
            ),
            dim=-1,
        )

    q, k = feature_map(q), feature_map(k)
    key_mask = valid[:, None, :, None]
    k = k * key_mask
    v = v * key_mask
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
    output = (q * (k * v).sum(2, True)).sum(-1)
    output = output / ((q * k.sum(2, True)).sum(-1) + eps)
    output = output.transpose(1, 2).reshape(batch, length, dim)
    output = _projection(output, parameters, "out_proj")
    return torch.where(valid[:, :, None], output, torch.zeros_like(output))


def test_rebased_matches_fla_noncausal_reference_forward_and_gradients() -> None:
    torch.manual_seed(57)
    layer = train_transformers.MaskedReBasedAttention(
        16, 2, bias=True, feature_dim=4
    ).double()
    inputs = torch.randn(2, 7, 16, dtype=torch.double)
    valid = torch.tensor(
        [[True] * 7, [True, True, True, True, False, False, False]]
    )
    oracle = functools.partial(
        _fla_rebased_reference_oracle,
        num_heads=2,
        feature_dim=4,
        eps=1e-5,
    )
    _assert_upstream_forward_and_gradients(
        layer, oracle, inputs, valid, rtol=1e-11, atol=1e-12
    )


def _nystrom_attention_0014_oracle(
    inputs: torch.Tensor,
    valid: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    num_heads: int,
    landmarks: int,
) -> torch.Tensor:
    """nystrom-attention 0.0.14, commit fbe9eba, residual=False."""
    batch, length, dim = inputs.shape
    assert length % landmarks == 0 and bool(valid.all())
    head_dim = dim // num_heads
    q, k, v = _projection(inputs, parameters, "qkv").chunk(3, dim=-1)
    reshape = lambda value: value.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    q, k, v = map(reshape, (q, k, v))
    q = q * head_dim ** -0.5
    chunk = length // landmarks
    q_landmarks = q.reshape(batch, num_heads, landmarks, chunk, head_dim).mean(dim=3)
    k_landmarks = k.reshape(batch, num_heads, landmarks, chunk, head_dim).mean(dim=3)
    similarity = lambda left, right: torch.einsum("bhnd,bhmd->bhnm", left, right)
    attention1 = similarity(q, k_landmarks).softmax(dim=-1)
    attention2 = similarity(q_landmarks, k_landmarks).softmax(dim=-1)
    attention3 = similarity(q_landmarks, k).softmax(dim=-1)
    absolute = attention2.abs()
    estimate = attention2.transpose(-1, -2) / (
        absolute.sum(dim=-1).amax() * absolute.sum(dim=-2).amax()
    )
    identity = torch.eye(landmarks, device=inputs.device, dtype=inputs.dtype)
    for _ in range(6):
        product = attention2 @ estimate
        estimate = 0.25 * estimate @ (
            13 * identity
            - product @ (15 * identity - product @ (7 * identity - product))
        )
    output = (attention1 @ estimate) @ (attention3 @ v)
    output = output.transpose(1, 2).reshape(batch, length, dim)
    output = _projection(output, parameters, "out_proj")
    return output


def test_nystromformer_matches_upstream_0014_forward_and_gradients() -> None:
    torch.manual_seed(59)
    layer = train_transformers.MaskedNystromAttention(
        16, 2, bias=True, landmarks=4
    ).float()
    inputs = torch.randn(2, 8, 16)
    valid = torch.ones(2, 8, dtype=torch.bool)
    oracle = functools.partial(
        _nystrom_attention_0014_oracle, num_heads=2, landmarks=4
    )
    _assert_upstream_forward_and_gradients(
        layer, oracle, inputs, valid, rtol=3e-4, atol=3e-6
    )


def _opennlplab_cosformer_oracle(
    inputs: torch.Tensor,
    valid: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    num_heads: int,
) -> torch.Tensor:
    """OpenNLPLab/cosFormer commit 8e12da1, noncausal forward."""
    batch, length, dim = inputs.shape
    assert bool(valid.all())
    head_dim = dim // num_heads
    reshape = lambda value: value.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    q = F.relu(reshape(_projection(inputs, parameters, "q_proj")))
    k = F.relu(reshape(_projection(inputs, parameters, "k_proj")))
    v = reshape(_projection(inputs, parameters, "v_proj"))
    index = (math.pi / 2) * torch.arange(
        1, length + 1, device=inputs.device, dtype=inputs.dtype
    )[None, None, :, None]
    q = torch.cat((q * (index / length).sin(), q * (index / length).cos()), dim=-1)
    k = torch.cat((k * (index / length).sin(), k * (index / length).cos()), dim=-1)
    context = torch.einsum("bhnm,bhnd->bhmd", k, v)
    inverse = torch.einsum("bhnm,bhm->bhn", q, k.sum(dim=2)).clamp_min(1e-6).reciprocal()
    output = torch.einsum("bhnm,bhmd,bhn->bhnd", q, context, inverse)
    output = output.transpose(1, 2).reshape(batch, length, dim)
    return _projection(output, parameters, "out_proj")


def test_cosformer_matches_upstream_forward_and_gradients() -> None:
    torch.manual_seed(61)
    layer = train_transformers.MaskedCosformerAttention(16, 2, bias=True).float()
    inputs = torch.randn(2, 7, 16)
    valid = torch.ones(2, 7, dtype=torch.bool)
    oracle = functools.partial(_opennlplab_cosformer_oracle, num_heads=2)
    _assert_upstream_forward_and_gradients(
        layer, oracle, inputs, valid, rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize(
    "mixer", ("linear_transformer", "performer", "nystromformer", "cosformer", "rebased")
)
def test_dna_baselines_are_rejected_for_lra(mixer: str) -> None:
    with pytest.raises(ValueError, match="GenomicBenchmarks-only"):
        resolve_args(parse_args(["--suite", "lra", "--task", "listops", "--mixer", mixer]))


@pytest.mark.parametrize("mixer", ("nystromformer", "rebased"))
def test_frozen_dna_baselines_are_rejected_for_formal_runs(mixer: str) -> None:
    with pytest.raises(ValueError, match="frozen baseline"):
        resolve_args(
            parse_args(
                [
                    "--suite",
                    "genomic",
                    "--task",
                    "demo_human_or_worm",
                    "--mixer",
                    mixer,
                    "--formal",
                ]
            )
        )


@pytest.mark.parametrize(
    ("mixer", "implementation"),
    (
        ("linear_transformer", "linear-transformer-elu-plus-one-adapted-fp32-core"),
        ("performer", "performer-pytorch-1.1.4-adapted-fp32-core"),
        ("nystromformer", "nystrom-attention-0.0.14-adapted-no-conv"),
        ("cosformer", "opennlplab-cosformer-official-adapted"),
        ("rebased", "fla-rebased-noncausal-reference-adapted"),
    ),
)
def test_dna_baseline_metadata_records_implementation(mixer: str, implementation: str) -> None:
    args = resolve_args(
        parse_args(["--suite", "genomic", "--task", "demo_human_or_worm", "--mixer", mixer])
    )
    dataset = TensorDataset(torch.ones(1, 4, dtype=torch.long), torch.zeros(1, dtype=torch.long))
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
    )
    model = build_model(args, bundle)
    payload = _build_run_payload(
        args,
        bundle,
        {"train": 1, "validation": 1, "test": 1},
        model,
        torch.device("cpu"),
        cuda_enabled=False,
    )
    assert payload["model"]["mixer"] == mixer
    assert payload["model"]["implementation"] == implementation
    assert "memsolve_cuda_contract" not in payload["runtime"]


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


def test_genomic_folder_protocol_uses_official_test_without_resplitting(tmp_path) -> None:
    task_root = tmp_path / "demo_human_or_worm"
    for split, records in {
        "train": {"negative": ["AC", "ACG"], "positive": ["TTTT", "G"]},
        "test": {"negative": ["A"], "positive": ["GGGGGG"]},
    }.items():
        for label, sequences in records.items():
            directory = task_root / split / label
            directory.mkdir(parents=True)
            for index, sequence in enumerate(sequences):
                (directory / f"{index}.txt").write_text(sequence, encoding="utf-8")
    bundle = prepare_genomic_benchmarks(
        "demo_human_or_worm",
        data_root=tmp_path,
        max_length=0,
        validation_fraction=0.25,
        split_seed=7,
        allow_download=False,
        revision=None,
    )
    assert bundle.max_length == 6
    assert len(bundle.train) + len(bundle.validation) == 4
    assert len(bundle.test) == 2
    assert bundle.metadata["split_protocol"] == "official-train-test-plus-stratified-validation-v1"


def test_formal_local_genomic_provenance_hashes_source_contents(tmp_path) -> None:
    task_root = tmp_path / "demo_human_or_worm"
    for split, records in {
        "train": {"negative": ["AC", "ACG"], "positive": ["TT", "TTT"]},
        "test": {"negative": ["A"], "positive": ["G"]},
    }.items():
        for label, sequences in records.items():
            directory = task_root / split / label
            directory.mkdir(parents=True)
            for index, sequence in enumerate(sequences):
                (directory / f"{index}.txt").write_text(sequence, encoding="utf-8")

    first = prepare_genomic_benchmarks(
        "demo_human_or_worm",
        data_root=tmp_path,
        max_length=0,
        validation_fraction=0.25,
        split_seed=7,
        allow_download=False,
        revision=None,
        formal=True,
    )
    validate_formal_source_provenance(first)
    first_hash = first.metadata["source"]["content_sha256"]
    (task_root / "train" / "negative" / "0.txt").write_text("GG", encoding="utf-8")
    second = prepare_genomic_benchmarks(
        "demo_human_or_worm",
        data_root=tmp_path,
        max_length=0,
        validation_fraction=0.25,
        split_seed=7,
        allow_download=False,
        revision=None,
        formal=True,
    )
    assert second.metadata["source"]["content_sha256"] != first_hash


def test_formal_provenance_rejects_an_unhashed_local_source(tmp_path) -> None:
    task_root = tmp_path / "demo_human_or_worm"
    for split in ("train", "test"):
        for label, sequence in (("negative", "AC"), ("positive", "GT")):
            directory = task_root / split / label
            directory.mkdir(parents=True)
            (directory / "0.txt").write_text(sequence, encoding="utf-8")
            if split == "train":
                (directory / "1.txt").write_text(sequence + "A", encoding="utf-8")
    bundle = prepare_genomic_benchmarks(
        "demo_human_or_worm",
        data_root=tmp_path,
        max_length=0,
        validation_fraction=0.25,
        split_seed=7,
        allow_download=False,
        revision=None,
    )
    with pytest.raises(ValueError, match="content_sha256"):
        validate_formal_source_provenance(bundle)


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
    assert bundle.value_masking == "nonzero"
    assert bundle.metadata["value_masking"] == "nonzero-pixels"
    assert bundle.max_length == 4
    assert values.shape == (4, 1)
    assert label in (0, 1)


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


def test_lra_defaults_preserve_the_requested_learning_rates_and_burn_in() -> None:
    pathfinder = resolve_args(parse_args(["--suite", "lra", "--task", "pathfinder"]))
    pathfinder_custom = resolve_args(
        parse_args(
            [
                "--suite",
                "lra",
                "--task",
                "pathfinder",
                "--pathfinder-resolution",
                "16",
            ]
        )
    )
    pathx = resolve_args(parse_args(["--suite", "lra", "--task", "pathx"]))
    listops = resolve_args(parse_args(["--suite", "lra", "--task", "listops"]))
    text = resolve_args(parse_args(["--suite", "lra", "--task", "text"]))
    assert pathfinder.lr == 2e-4
    assert (
        pathfinder.dim,
        pathfinder.depth,
        pathfinder.heads,
        pathfinder.mlp_ratio,
        pathfinder.dropout,
        pathfinder.batch_size,
        pathfinder.eval_batch_size,
        pathfinder.grad_accum,
        pathfinder.pooling,
    ) == (
        256,
        6,
        4,
        2.0,
        0.0,
        64,
        256,
        2,
        "meanmax",
    )
    assert pathfinder.early_stop_min_epochs == 150
    assert pathfinder.pathfinder_resolution == 32
    assert pathfinder_custom.pathfinder_resolution == 16
    assert pathfinder.max_length is None
    assert pathfinder.validation_fraction is None
    assert pathfinder.split_seed is None
    assert pathx.pathfinder_resolution == 128
    assert pathx.max_length is None
    assert (pathx.dim, pathx.depth, pathx.batch_size, pathx.grad_accum) == (128, 6, 12, 10)
    assert listops.lr == 5e-4
    assert (
        listops.dim,
        listops.depth,
        listops.batch_size,
        listops.eval_batch_size,
        listops.grad_accum,
        listops.mlp_ratio,
        listops.pooling,
    ) == (
        128,
        6,
        25,
        25,
        2,
        4.0,
        "mean",
    )
    assert (
        text.dim,
        text.depth,
        text.batch_size,
        text.eval_batch_size,
        text.grad_accum,
    ) == (256, 6, 16, 16, 2)
    assert listops.early_stop_min_epochs == 30


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
    assert payload["model"]["mlp_ratio"] == 2.0
    assert payload["model"]["dropout"] == 0.0
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


def test_formal_download_requires_an_immutable_revision() -> None:
    args = resolve_args(
        parse_args(
            [
                "--suite",
                "lra",
                "--task",
                "text",
                "--formal",
                "--allow-download",
            ]
        )
    )
    with pytest.raises(ValueError, match="immutable full-SHA"):
        _validate_formal_data_source(args)
    args.data_revision = "release-1"
    with pytest.raises(ValueError, match="immutable full-SHA"):
        _validate_formal_data_source(args)
    args.data_revision = "0123456789abcdef0123456789abcdef01234567"
    _validate_formal_data_source(args)


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


def test_shared_lra_config_resolves_the_selected_task_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    pathfinder = resolve_args(
        parse_args(
            [
                "--config",
                str(root / "experiments/configs/lra.toml"),
                "--task",
                "pathfinder",
            ]
        )
    )
    assert (
        pathfinder.lr,
        pathfinder.epochs,
        pathfinder.dim,
        pathfinder.batch_size,
        pathfinder.grad_accum,
        pathfinder.pooling,
    ) == (
        2e-4,
        200,
        256,
        64,
        2,
        "meanmax",
    )
    retrieval = resolve_args(
        parse_args(
            [
                "--config",
                str(root / "experiments/configs/lra.toml"),
                "--task",
                "retrieval",
            ]
        )
    )
    assert (retrieval.dim, retrieval.depth, retrieval.batch_size, retrieval.grad_accum) == (
        128,
        6,
        16,
        4,
    )


def test_sequence_toml_template_supplies_required_task_selection() -> None:
    root = Path(__file__).resolve().parents[2]
    args = resolve_args(parse_args(["--config", str(root / "experiments/configs/lra.toml")]))
    assert (args.suite, args.task, args.rank, args.lr) == ("lra", "listops", 32, 5e-4)


def test_genomic_toml_template_matches_the_formal_rank() -> None:
    root = Path(__file__).resolve().parents[2]
    args = resolve_args(
        parse_args(["--config", str(root / "experiments/configs/genomic.toml")])
    )
    assert (args.suite, args.dim, args.depth, args.heads, args.rank) == (
        "genomic",
        128,
        4,
        4,
        32,
    )
    assert (args.batch_size, args.grad_accum) == (64, 2)

    ordinary = resolve_args(
        parse_args(
            [
                "--config",
                str(root / "experiments/configs/genomic.toml"),
                "--task",
                "demo_human_or_worm",
            ]
        )
    )
    assert (ordinary.batch_size, ordinary.grad_accum, ordinary.rank) == (128, 1, 32)


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
