from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", minversion="4.57.6")

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, BertConfig, BertForMaskedLM

from integrations.transformers import RidgonBertConfig, RidgonBertForMaskedLM, RidgonBertModel
from ridgon import Ridgon


pytestmark = pytest.mark.integration


def config(**kwargs):
    values = dict(
        vocab_size=43, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=128,
        max_position_embeddings=32, ridgon_rank=16, hidden_dropout_prob=0,
    )
    return RidgonBertConfig(**(values | kwargs))


def batch(device="cpu"):
    ids = torch.tensor([[2, 5, 9, 11, 7, 3, 0, 0], [2, 13, 17, 19, 23, 29, 31, 3]], device=device)
    labels = torch.full_like(ids, -100)
    labels[:, 2:4] = ids[:, 2:4]
    masked = ids.clone()
    masked[:, 2:4] = 4
    return dict(input_ids=masked, attention_mask=(ids != 0), labels=labels)


def test_scaffold_initialization_and_weight_tying():
    torch.manual_seed(7)
    model = RidgonBertForMaskedLM(config(hidden_size=192, num_attention_heads=6))
    assert model.bert.pooler is None
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert len([m for m in model.modules() if isinstance(m, Ridgon)]) == 2
    assert not any(isinstance(m, nn.Conv1d) for m in model.modules())
    for block in model.bert.encoder.layer:
        assert isinstance(block.attention.output.dense, nn.Identity)
        core = block.attention.mixer
        offset = core.config.num_heads * core.config.rank
        q, k, v = core.w_qkv.weight.split((offset, offset, 192))
        assert abs(k.std().item() * 192**0.5 - 1) < 0.05
        for w in (q, v, core.w_o.weight):
            assert abs(w.std().item() / 0.02 - 1) < 0.05
        assert torch.count_nonzero(core.core_delta) == 0
        assert torch.equal(core.head_norm_weight, torch.ones_like(core.head_norm_weight))


def test_swiglu_width_and_packed_projection_gradients():
    assert RidgonBertConfig().intermediate_size == 2048
    assert RidgonBertConfig(hidden_size=384).intermediate_size == 1024
    model = RidgonBertForMaskedLM(config()).double()
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


def test_checkpoint_requires_swiglu_architecture_metadata(tmp_path):
    RidgonBertForMaskedLM(config()).save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    del saved["ffn_type"]
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="ffn_type"):
        AutoModelForMaskedLM.from_pretrained(tmp_path)


def test_checkpoint_requires_swiglu_weights_even_with_explicit_config(tmp_path):
    from safetensors.torch import load_file, save_file
    model = RidgonBertForMaskedLM(config())
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    weights = {k: v for k, v in weights.items() if "gate_up_proj" not in k}
    save_file(weights, path, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="missing SwiGLU"):
        RidgonBertForMaskedLM.from_pretrained(tmp_path, config=model.config)


def test_padding_is_excluded_but_mlm_tokens_are_valid():
    torch.manual_seed(12)
    model = RidgonBertForMaskedLM(config()).eval()
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
def test_mlm_backward_and_gradient_checkpointing(checkpointing):
    torch.manual_seed(21)
    model = RidgonBertForMaskedLM(config()).train()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    output = model(**batch())
    expected = nn.functional.cross_entropy(output.logits.reshape(-1, 43), batch()["labels"].reshape(-1))
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    before = model.bert.encoder.layer[0].attention.mixer.core_delta.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, model.bert.encoder.layer[0].attention.mixer.core_delta)


@pytest.mark.parametrize("max_shard_size", ["5GB", "20KB"])
def test_hf_checkpoint_roundtrip_and_encoder_extraction(tmp_path, max_shard_size):
    torch.manual_seed(31)
    model = RidgonBertForMaskedLM(config()).eval()
    with torch.no_grad():
        model.bert.encoder.layer[0].attention.mixer.core_delta.normal_(std=0.1)
    model.save_pretrained(tmp_path, max_shard_size=max_shard_size)
    assert isinstance(AutoConfig.from_pretrained(tmp_path), RidgonBertConfig)
    loaded, info = AutoModelForMaskedLM.from_pretrained(tmp_path, output_loading_info=True)
    assert info == dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    torch.testing.assert_close(loaded(**batch()).logits, model(**batch()).logits, rtol=0, atol=0)
    encoder = AutoModel.from_pretrained(tmp_path)
    assert isinstance(encoder, RidgonBertModel)
    inputs = {k: v for k, v in batch().items() if k != "labels"}
    torch.testing.assert_close(encoder(**inputs).last_hidden_state, model.bert(**inputs).last_hidden_state, rtol=0, atol=0)
    # The base encoder itself is also a standalone HF checkpoint.
    encoder.save_pretrained(tmp_path / "encoder")
    restored = AutoModel.from_pretrained(tmp_path / "encoder")
    torch.testing.assert_close(restored(**inputs).last_hidden_state, encoder(**inputs).last_hidden_state, rtol=0, atol=0)


@pytest.mark.parametrize("missing", [False, True])
def test_hf_checkpoint_rejects_bad_operator_contract(tmp_path, missing):
    model = RidgonBertForMaskedLM(config())
    model.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    if missing:
        del saved["ridgon_contract"]
    else:
        saved["ridgon_contract"]["version"] -= 1
    path.write_text(json.dumps(saved))
    with pytest.raises((ValueError, RuntimeError), match="contract"):
        AutoModelForMaskedLM.from_pretrained(tmp_path)


def test_explicit_ridgon_config_cannot_disguise_stock_bert_weights(tmp_path):
    c = config()
    stock = BertForMaskedLM(BertConfig(
        vocab_size=c.vocab_size, hidden_size=c.hidden_size,
        num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
        intermediate_size=c.intermediate_size, max_position_embeddings=c.max_position_embeddings,
    ))
    stock.save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="missing Ridgon mixer weights"):
        RidgonBertForMaskedLM.from_pretrained(tmp_path, config=c)


def test_unsupported_attention_interfaces_fail_explicitly():
    with pytest.raises(ValueError, match="bidirectional"):
        config(is_decoder=True)
    with pytest.raises(ValueError, match="attention-probability"):
        config(attention_probs_dropout_prob=0.1)
    model = RidgonBertForMaskedLM(config())
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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rank", [32, 48])
def test_bf16_cuda_mlm_matches_reference_and_updates(rank):
    torch.manual_seed(43)
    reference = RidgonBertForMaskedLM(config(ridgon_rank=rank)).cuda().train()
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
