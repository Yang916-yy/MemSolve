from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", minversion="4.57.6")

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, BertConfig, BertForMaskedLM

from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM, MemSolveBertModel
from memsolve import MemSolve


pytestmark = pytest.mark.integration


def config(**kwargs):
    values = dict(
        vocab_size=43, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=128,
        max_position_embeddings=32, memsolve_rank=16, hidden_dropout_prob=0,
        memsolve_short_conv_kernel=0,
    )
    return MemSolveBertConfig(**(values | kwargs))


def batch(device="cpu"):
    ids = torch.tensor([[2, 5, 9, 11, 7, 3, 0, 0], [2, 13, 17, 19, 23, 29, 31, 3]], device=device)
    labels = torch.full_like(ids, -100)
    labels[:, 2:4] = ids[:, 2:4]
    masked = ids.clone()
    masked[:, 2:4] = 4
    return dict(input_ids=masked, attention_mask=(ids != 0), labels=labels)


def test_scaffold_initialization_and_weight_tying():
    torch.manual_seed(7)
    model = MemSolveBertForMaskedLM(config(hidden_size=192, num_attention_heads=6))
    assert model.bert.pooler is None
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert len([m for m in model.modules() if isinstance(m, MemSolve)]) == 2
    assert sum(isinstance(m, nn.Conv1d) for m in model.modules()) == 0
    for block in model.bert.encoder.layer:
        assert isinstance(block.attention.output.dense, nn.Identity)
        core = block.attention.mixer
        for w in (core.w_bc.weight, core.w_o.weight):
            assert abs(w.std().item() / 0.02 - 1) < 0.05
        assert torch.count_nonzero(core.core_base_raw) == 0
        assert not hasattr(core, "core_drive_weight")
        torch.testing.assert_close(core.complement(), torch.full((6,), .9))


def test_swiglu_width_and_packed_projection_gradients():
    assert MemSolveBertConfig().intermediate_size == 2048
    assert MemSolveBertConfig(hidden_size=384).intermediate_size == 1024
    model = MemSolveBertForMaskedLM(config()).double()
    block = model.bert.encoder.layer[0]
    intermediate = block.intermediate
    x = torch.randn(2, 7, 64, dtype=torch.float64, requires_grad=True)
    gate_w, up_w = intermediate.gate_up_proj.weight.chunk(2)
    gate_b, up_b = intermediate.gate_up_proj.bias.chunk(2)
    linear = nn.functional.linear
    expected = linear(
        nn.functional.silu(linear(x, gate_w, gate_b)) * linear(x, up_w, up_b),
        block.output.dense.weight, block.output.dense.bias,
    )
    actual = block.output.dense(intermediate(x))
    torch.testing.assert_close(actual, expected)
    inputs = (x, *intermediate.parameters(), *block.output.dense.parameters())
    grad = torch.randn_like(actual)
    for a, b in zip(torch.autograd.grad(actual, inputs, grad), torch.autograd.grad(expected, inputs, grad)):
        torch.testing.assert_close(a, b)
    # Only the encoder FFN changes; the MLM prediction transform keeps GELU.
    assert model.config.hidden_act == "gelu"


def test_centered_short_conv_identity_gradients_padding_and_reload(tmp_path):
    torch.manual_seed(7)
    plain = MemSolveBertForMaskedLM(config()).eval()
    local = MemSolveBertForMaskedLM(config(memsolve_short_conv_kernel=3)).eval()
    missing, unexpected = local.load_state_dict(plain.state_dict(), strict=False)
    assert missing and all("attention.short_conv.conv." in key for key in missing)
    assert not unexpected
    data = batch()
    torch.testing.assert_close(local(**data).logits, plain(**data).logits, rtol=0, atol=0)
    local(**data).loss.backward()
    for layer in local.bert.encoder.layer:
        conv = layer.attention.short_conv.conv
        assert conv.groups == conv.in_channels == conv.out_channels
        assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
        assert conv.weight.grad.abs().sum() > 0
        with torch.no_grad():
            conv.weight.normal_(std=0.1)
            conv.bias.fill_(0.1)
    original = local(**data).logits
    changed = dict(data, input_ids=data["input_ids"].clone())
    changed["input_ids"][0, 6:] = torch.tensor([37, 41])
    torch.testing.assert_close(local(**changed).logits[data["attention_mask"]],
                               original[data["attention_mask"]], rtol=0, atol=0)
    short = {k: v[:1, :6] for k, v in data.items()}
    torch.testing.assert_close(local(**short).logits, original[:1, :6], atol=2e-5, rtol=2e-4)
    local.save_pretrained(tmp_path)
    restored = AutoModelForMaskedLM.from_pretrained(tmp_path)
    assert restored.config.memsolve_short_conv_kernel == 3
    torch.testing.assert_close(restored(**data).logits, original, rtol=0, atol=0)


def test_checkpoint_requires_swiglu_architecture_metadata(tmp_path):
    MemSolveBertForMaskedLM(config()).save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    del saved["ffn_type"]
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="ffn_type"):
        AutoModelForMaskedLM.from_pretrained(tmp_path)


def test_checkpoint_requires_swiglu_weights_even_with_explicit_config(tmp_path):
    from safetensors.torch import load_file, save_file
    model = MemSolveBertForMaskedLM(config())
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    weights = {k: v for k, v in weights.items() if "gate_up_proj" not in k}
    save_file(weights, path, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="missing SwiGLU"):
        MemSolveBertForMaskedLM.from_pretrained(tmp_path, config=model.config)


def test_padding_is_excluded_but_mlm_tokens_are_valid():
    torch.manual_seed(12)
    model = MemSolveBertForMaskedLM(config()).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MemSolve):
                module.core_base_raw.normal_(std=0.1)
    data = batch()
    original = model(**data)
    changed = dict(data, input_ids=data["input_ids"].clone())
    changed["input_ids"][0, 6:] = torch.tensor([37, 41])
    actual = model(**changed)
    torch.testing.assert_close(actual.logits[data["attention_mask"]], original.logits[data["attention_mask"]], rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, original.loss, rtol=0, atol=0)
    assert (data["attention_mask"][data["labels"] != -100]).all()
    # Appending padding must also preserve the normalization count of the memory.
    short = {k: v[:1, :6] for k, v in data.items()}
    torch.testing.assert_close(model(**short).logits, original.logits[:1, :6], atol=2e-5, rtol=2e-4)
    empty = dict(data, attention_mask=torch.zeros_like(data["attention_mask"]))
    assert torch.isfinite(model(**empty).logits).all()


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("position_type", ["absolute", "rove"])
def test_mlm_backward_and_gradient_checkpointing(checkpointing, position_type):
    torch.manual_seed(21)
    model = MemSolveBertForMaskedLM(config(position_embedding_type=position_type)).train()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    output = model(**batch())
    expected = nn.functional.cross_entropy(output.logits.reshape(-1, 43), batch()["labels"].reshape(-1))
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    before = model.bert.encoder.layer[0].attention.mixer.core_base_raw.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, model.bert.encoder.layer[0].attention.mixer.core_base_raw)


@pytest.mark.parametrize("max_shard_size", ["5GB", "20KB"])
@pytest.mark.parametrize("position_type", ["absolute", "rove"])
def test_hf_checkpoint_roundtrip_and_encoder_extraction(tmp_path, max_shard_size, position_type):
    torch.manual_seed(31)
    model = MemSolveBertForMaskedLM(config(memsolve_core_mode="equilibrium",
                                          position_embedding_type=position_type)).eval()
    with torch.no_grad():
        model.bert.encoder.layer[0].attention.mixer.core_base_raw.normal_(std=0.1)
    model.save_pretrained(tmp_path, max_shard_size=max_shard_size)
    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config["model_type"] == "memsolve_bert"
    assert saved_config["architectures"] == ["MemSolveBertForMaskedLM"]
    assert isinstance(AutoConfig.from_pretrained(tmp_path), MemSolveBertConfig)
    loaded, info = AutoModelForMaskedLM.from_pretrained(tmp_path, output_loading_info=True)
    assert loaded.config.memsolve_core_mode == "equilibrium"
    assert loaded.config.position_embedding_type == position_type
    assert info == dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    torch.testing.assert_close(loaded(**batch()).logits, model(**batch()).logits, rtol=0, atol=0)
    encoder = AutoModel.from_pretrained(tmp_path)
    assert isinstance(encoder, MemSolveBertModel)
    inputs = {k: v for k, v in batch().items() if k != "labels"}
    torch.testing.assert_close(encoder(**inputs).last_hidden_state, model.bert(**inputs).last_hidden_state, rtol=0, atol=0)
    # The base encoder itself is also a standalone HF checkpoint.
    encoder.save_pretrained(tmp_path / "encoder")
    assert json.loads((tmp_path / "encoder" / "config.json").read_text())["architectures"] == ["MemSolveBertModel"]
    restored = AutoModel.from_pretrained(tmp_path / "encoder")
    torch.testing.assert_close(restored(**inputs).last_hidden_state, encoder(**inputs).last_hidden_state, rtol=0, atol=0)


@pytest.mark.parametrize("missing", [False, True])
def test_hf_checkpoint_rejects_bad_operator_contract(tmp_path, missing):
    model = MemSolveBertForMaskedLM(config())
    model.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    if missing:
        del saved["memsolve_contract"]
    else:
        saved["memsolve_contract"]["version"] -= 1
    path.write_text(json.dumps(saved))
    with pytest.raises((ValueError, RuntimeError), match="contract"):
        AutoModelForMaskedLM.from_pretrained(tmp_path)


def test_explicit_memsolve_config_cannot_disguise_stock_bert_weights(tmp_path):
    c = config()
    stock = BertForMaskedLM(BertConfig(
        vocab_size=c.vocab_size, hidden_size=c.hidden_size,
        num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
        intermediate_size=c.intermediate_size, max_position_embeddings=c.max_position_embeddings,
    ))
    stock.save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="missing MemSolve mixer weights"):
        MemSolveBertForMaskedLM.from_pretrained(tmp_path, config=c)


def test_unsupported_attention_interfaces_fail_explicitly():
    with pytest.raises(ValueError, match="bidirectional"):
        config(is_decoder=True)
    with pytest.raises(ValueError, match="attention-probability"):
        config(attention_probs_dropout_prob=0.1)
    with pytest.raises(ValueError, match="position_embedding_type"):
        config(position_embedding_type="relative_key")
    model = MemSolveBertForMaskedLM(config())
    with pytest.raises(ValueError, match="probabilities"):
        model(**batch(), output_attentions=True)
    data = batch()
    data["attention_mask"] = torch.ones(2, 8, 8)
    with pytest.raises(ValueError, match="pairwise"):
        model(**data)
    with pytest.raises(ValueError, match="cross attention"):
        model(**batch(), encoder_hidden_states=torch.zeros(2, 8, 64))
    with pytest.raises(ValueError, match="KV caches"):
        model.bert(input_ids=batch()["input_ids"], use_cache=True)
    with pytest.raises(ValueError, match="KV caches"):
        model.bert(input_ids=batch()["input_ids"], cache_position=torch.arange(8))


def test_rove_positions_shift_invariance_and_no_learned_table():
    torch.manual_seed(53)
    model = MemSolveBertForMaskedLM(config(memsolve_short_conv_kernel=3)).eval()
    embeddings = model.bert.embeddings
    for name in ("position_embeddings", "position_ids", "token_type_ids"):
        assert not hasattr(embeddings, name)
    with torch.no_grad():
        for layer in model.bert.encoder.layer:
            layer.attention.mixer.core_base_raw.normal_(std=0.1)
            layer.attention.short_conv.conv.weight.normal_(std=0.05)
    data = batch()
    positions = torch.arange(8).expand(2, -1)
    seen_factors = []
    hooks = [layer.attention.mixer.register_forward_pre_hook(
        lambda module, args, kwargs: seen_factors.append(kwargs["rotary_factors"]), with_kwargs=True)
        for layer in model.bert.encoder.layer]
    actual = model(**data, position_ids=positions).logits
    for hook in hooks:
        hook.remove()
    assert len(seen_factors) == model.config.num_hidden_layers
    assert all(factors is seen_factors[0] for factors in seen_factors)
    expected = model.bert.encoder.layer[0].attention.mixer._rotary_factors(
        embeddings.word_embeddings(data["input_ids"]), data["attention_mask"], positions, None)
    for actual_factor, expected_factor in zip(seen_factors[0], expected):
        torch.testing.assert_close(actual_factor, expected_factor, rtol=0, atol=0)
    shifted = model(**data, position_ids=positions + torch.tensor([[41], [97]])).logits
    torch.testing.assert_close(actual, shifted, atol=3e-6, rtol=3e-5)
    changed = positions * 3
    assert (model(**data, position_ids=changed).logits - actual).abs().max() > 1e-5
    embedded = embeddings.word_embeddings(data["input_ids"])
    torch.testing.assert_close(model(inputs_embeds=embedded, attention_mask=data["attention_mask"],
                                    position_ids=positions).logits, actual, atol=0, rtol=0)


def test_rove_off_has_no_extra_position_table_and_roundtrips(tmp_path):
    torch.manual_seed(61)
    plain = MemSolveBertForMaskedLM(config(position_embedding_type="none")).eval()
    torch.manual_seed(61)
    rotated = MemSolveBertForMaskedLM(config()).eval()
    for (name, p), (other_name, q) in zip(plain.named_parameters(), rotated.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert not hasattr(plain.bert.embeddings, "position_embeddings")
    assert all(not layer.attention.mixer.config.rotary for layer in plain.bert.encoder.layer)
    data = batch()
    actual = plain(**data).logits
    expected = rotated(**data, position_ids=torch.zeros_like(data["input_ids"])).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(plain(**data, position_ids=torch.arange(8) * 7).logits,
                               actual, atol=0, rtol=0)
    plain(**data).loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in plain.parameters())
    plain.save_pretrained(tmp_path)
    restored = AutoModelForMaskedLM.from_pretrained(tmp_path)
    assert restored.config.position_embedding_type == "none"
    torch.testing.assert_close(restored(**data).logits, actual, atol=0, rtol=0)
    with torch.no_grad():
        assert restored(input_ids=torch.randint(5, 43, (1, 513))).logits.shape == (1, 513, 43)


def test_rove_exceeds_512_without_position_buffer_limit():
    model = MemSolveBertForMaskedLM(config(hidden_size=32, num_hidden_layers=1,
        intermediate_size=64, max_position_embeddings=32)).eval()
    ids = torch.randint(5, 43, (1, 513))
    with torch.no_grad():
        output = model(input_ids=ids)
    assert output.logits.shape == (1, 513, 43)
    assert torch.isfinite(output.logits).all()


def test_rove_yarn_padding_does_not_change_per_example_scaling():
    torch.manual_seed(57)
    model = MemSolveBertForMaskedLM(config(hidden_size=32, num_hidden_layers=1,
        intermediate_size=64, memsolve_short_conv_kernel=3)).eval()
    with torch.no_grad():
        block = model.bert.encoder.layer[0].attention
        block.mixer.core_base_raw.normal_(std=0.1)
        block.short_conv.conv.weight.normal_(std=0.05)
        ids = torch.randint(5, 43, (2, 151))
        mask = torch.arange(151).unsqueeze(0) < torch.tensor([[67], [139]])
        batched = model(input_ids=ids, attention_mask=mask).logits
        for row, length in enumerate((67, 139)):
            single = model(input_ids=ids[row:row + 1, :length]).logits
            torch.testing.assert_close(batched[row:row + 1, :length], single,
                                       atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_rove_checkpoint_replay_retains_each_forward_positions(use_reentrant):
    torch.manual_seed(59)
    eager = MemSolveBertForMaskedLM(config()).train()
    with torch.no_grad():
        for layer in eager.bert.encoder.layer:
            layer.attention.mixer.core_base_raw.normal_(std=0.1)
    checkpointed = copy.deepcopy(eager)
    checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    positions = [torch.arange(8).unsqueeze(0), torch.arange(8).unsqueeze(0) * 3 + 17]
    for model in (eager, checkpointed):
        losses = [model(**batch(), position_ids=p).loss for p in positions]
        sum(losses).backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), checkpointed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)


def test_absolute_checkpoint_without_rotary_fields_loads_unchanged(tmp_path):
    model = MemSolveBertForMaskedLM(config(position_embedding_type="absolute")).eval()
    model.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    for name in ("memsolve_rotary_base", "memsolve_rotary_train_length", "memsolve_rotary_scaling"):
        del saved[name]
    path.write_text(json.dumps(saved))
    restored = AutoModelForMaskedLM.from_pretrained(tmp_path)
    assert hasattr(restored.bert.embeddings, "position_embeddings")
    assert not restored.bert.encoder.layer[0].attention.mixer.config.rotary
    torch.testing.assert_close(restored(**batch()).logits, model(**batch()).logits, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rank,conv_kernel", [(32, 0), (48, 0), (32, 3)])
@pytest.mark.parametrize("position_type", ["rove", "none"])
def test_bf16_cuda_mlm_matches_reference_and_updates(rank, conv_kernel, position_type):
    torch.manual_seed(43)
    reference = MemSolveBertForMaskedLM(config(memsolve_rank=rank, memsolve_short_conv_kernel=conv_kernel,
                                             position_embedding_type=position_type)).cuda().train()
    if conv_kernel:
        with torch.no_grad():
            for layer in reference.bert.encoder.layer:
                layer.attention.short_conv.conv.weight.normal_(std=0.02)
    fast = copy.deepcopy(reference)
    for layer in fast.bert.encoder.layer:
        layer.attention.implementation = "cuda"
    data = batch("cuda")
    outputs = []
    for model in (reference, fast):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**data)
        output.loss.backward()
        outputs.append(output)
    def relative(a, b):
        return (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
    assert relative(outputs[1].logits, outputs[0].logits) < 0.005
    ref_grads, fast_grads = [], []
    for (name, p), (other_name, q) in zip(reference.named_parameters(), fast.named_parameters()):
        assert name == other_name
        assert p.dtype == q.dtype == torch.float32
        assert q.grad is not None and torch.isfinite(q.grad).all(), name
        ref_grads.append(p.grad.flatten())
        fast_grads.append(q.grad.flatten())
    assert relative(torch.cat(fast_grads), torch.cat(ref_grads)) < 0.02
    torch.optim.AdamW(fast.parameters(), lr=1e-3, fused=True).step()
    # Plain inference works too, without turning parameters into BF16.
    fast.eval()
    with torch.no_grad():
        assert torch.isfinite(fast(**data).logits).all()


def test_glue_mlm_transfer_and_mnli_pooler_reuse(tmp_path):
    from transformers import AutoModelForSequenceClassification
    from experiments.train_glue import load_classifier

    mlm = MemSolveBertForMaskedLM(config())
    mlm.save_pretrained(tmp_path / "mlm")
    classifier, info = load_classifier(tmp_path / "mlm", "mnli", ["entailment", "neutral", "contradiction"], "reference")
    assert all(key.startswith(("classifier.", "bert.pooler.")) for key in info["missing_keys"])
    for name, value in mlm.bert.state_dict().items():
        torch.testing.assert_close(classifier.bert.state_dict()[name], value)
    data = batch()
    data["labels"] = torch.tensor([0, 2])
    output = classifier(**data)
    assert output.logits.shape == (2, 3) and torch.isfinite(output.loss)
    output.loss.backward()
    assert classifier.bert.encoder.layer[0].attention.mixer.w_bc.weight.grad is not None
    with torch.no_grad():
        classifier.bert.pooler.dense.weight.fill_(0.123)
    classifier.save_pretrained(tmp_path / "mnli")
    regression, _ = load_classifier(tmp_path / "mnli", "stsb", None, "reference")
    torch.testing.assert_close(regression.bert.pooler.dense.weight, classifier.bert.pooler.dense.weight)
    data["labels"] = torch.tensor([0.5, 3.5])
    output = regression(**data)
    torch.testing.assert_close(output.loss, torch.nn.functional.mse_loss(output.logits.flatten(), data["labels"]))
    regression.save_pretrained(tmp_path / "stsb")
    reloaded = AutoModelForSequenceClassification.from_pretrained(tmp_path / "stsb")
    torch.testing.assert_close(reloaded(**data).logits, output.logits)


def test_shared_rotary_tables_are_per_forward_and_recomputed_once(monkeypatch):
    model = MemSolveBertForMaskedLM(config(num_hidden_layers=3)).eval()
    first = model.bert.encoder.layer[0].attention.mixer
    original = first._rotary_factors
    calls = []
    def factors(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(first, "_rotary_factors", factors)
    def redundant(*args, **kwargs):
        raise AssertionError("later layers must reuse the per-forward position factors")
    for layer in model.bert.encoder.layer[1:]:
        monkeypatch.setattr(layer.attention.mixer, "_rotary_factors", redundant)
    for multiplier in (1, 3):
        model(**batch(), position_ids=torch.arange(8)[None] * multiplier)
    assert len(calls) == 2
    assert calls[0][0].data_ptr() != calls[1][0].data_ptr()
    assert not torch.equal(calls[0][0], calls[1][0])
