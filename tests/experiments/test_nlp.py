"""Offline contracts for preprocessing, Trainer resume, embedding export and scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", minversion="4.57.6")
pytest.importorskip("datasets")
pytest.importorskip("sentence_transformers", minversion="5.7")
pytest.importorskip("mteb", minversion="2.21")

from datasets import Dataset
from transformers import BertTokenizerFast
from experiments.nlp import (accumulation_steps, freeze_mask, load_prepared,
                             load_sentence_model, masked_loss, masked_metrics,
                             masked_statistics, read_json, write_json)
from experiments.prepare_c4 import prepare

pytestmark = pytest.mark.experiment


@pytest.fixture
def tokenizer(tmp_path):
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    (directory / "vocab.txt").write_text("\n".join(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] +
        ["the", "a", "red", "blue", "green", "bird", "fish", "tree", "water", "sky", "is", "in", "and"]))
    tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
    tokenizer.save_pretrained(directory)
    return tokenizer, directory


def test_fixed_mask_and_token_weighted_loss(tokenizer):
    tokenizer, _ = tokenizer
    example = tokenizer("the red bird is in the blue sky", max_length=16,
                        padding="max_length", return_special_tokens_mask=True)
    first = freeze_mask(example, tokenizer, 0.3, 17)
    torch.rand(11)
    assert first == freeze_mask(example, tokenizer, 0.3, 17)
    assert first["attention_mask"] == example["attention_mask"]
    for label, special, valid in zip(first["labels"], example["special_tokens_mask"], example["attention_mask"]):
        if special or not valid:
            assert label == -100
    logits = torch.randn(3, 5, 19, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, -100, -100], [3, -100, -100, -100, -100], [-100] * 5])
    full = masked_loss(SimpleNamespace(logits=logits), labels)
    split = sum(masked_loss(SimpleNamespace(logits=x), y, 3) for x, y in zip(logits.split(1), labels.split(1)))
    torch.testing.assert_close(full, split)
    grad = torch.autograd.grad(full, logits)[0]
    assert torch.count_nonzero(grad[2]) == 0
    metrics = masked_metrics(SimpleNamespace(predictions=masked_statistics(logits, labels).detach().numpy()))
    assert metrics["masked_nll"] == pytest.approx(full.item())
    assert metrics["masked_tokens"] == 3
    assert accumulation_steps(4096, 128, 2) == 16
    with pytest.raises(ValueError, match="divisible"):
        accumulation_steps(1000, 128, 2)


def test_lion_recipe_schedule_and_decay_conversion():
    from transformers import get_scheduler
    recipe = read_json(Path(__file__).resolve().parents[2] / "experiments/configs/ridgon_bert_mlm.json")
    settings = recipe["training"]
    peak = settings["learning_rate"]
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=peak, weight_decay=settings["weight_decay"])
    scheduler = get_scheduler(settings["lr_scheduler_type"], optimizer=optimizer,
                 num_warmup_steps=4200, num_training_steps=70000,
                 scheduler_specific_kwargs=settings["lr_scheduler_kwargs"])
    assert scheduler.lr_lambdas[0](0) == 0
    assert scheduler.lr_lambdas[0](4200) == pytest.approx(1)
    assert scheduler.lr_lambdas[0](70000) == pytest.approx(.02)
    for fraction in (0., .5, 1.):
        parameter.data.fill_(1.)
        parameter.grad = torch.zeros_like(parameter)
        optimizer.param_groups[0]["lr"] = peak * fraction
        optimizer.step()
        assert parameter.item() == pytest.approx(1. - fraction * 1e-5, abs=1e-7)


@pytest.fixture
def prepared(tmp_path, tokenizer):
    _, directory = tokenizer
    source = tmp_path / "documents.jsonl"
    source.write_text("\n".join(json.dumps({"text": text}) for text in
        ["the red bird is in the blue sky", "a green tree and blue water"] * 16))
    args = argparse.Namespace(output=str(tmp_path / "prepared"), split="train", files=[str(source)],
             dataset="allenai/c4", dataset_config="en", revision="main", tokenizer=str(directory),
             max_length=16, num_shards=1, shard_index=0, max_docs=24,
             eval_mask_probability=0.3, seed=17)
    output = prepare(args)
    assert prepare(args) == output
    args.split, args.max_docs = "validation", 8
    prepare(args)
    return tmp_path / "prepared"


def test_prepared_partition_integrity(prepared):
    data, metadata, _ = load_prepared(prepared)
    assert len(data["train"]) == 24 and len(data["validation"]) == 8
    assert "labels" not in data["train"].column_names
    assert "labels" in data["validation"].column_names
    assert data["train"].features["input_ids"].feature.dtype == "int32"
    assert data["train"].features["attention_mask"].feature.dtype == "int8"
    path = next((prepared / "train").glob("*/manifest.json"))
    manifest = read_json(path)
    manifest["num_shards"] = 2
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Incomplete"):
        load_prepared(prepared)


def test_offline_mlm_retrieval_and_mteb(prepared, tokenizer, tmp_path, monkeypatch):
    from experiments.train_mlm import train as train_mlm
    from experiments.train_retrieval import train as train_retrieval
    from experiments import evaluate_embeddings as evaluation
    torch.set_num_threads(2)
    common = {"max_steps": 2, "per_device_train_batch_size": 4, "per_device_eval_batch_size": 4,
              "learning_rate": 1e-4, "optim": "adamw_torch", "use_cpu": True,
              "bf16": False, "report_to": "none", "dataloader_num_workers": 0,
              "save_steps": 1, "logging_steps": 1, "disable_tqdm": True, "seed": 17}
    recipe = {"model": {"vocab_size": len(tokenizer[0]), "hidden_size": 64,
              "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 128,
              "max_position_embeddings": 32, "ridgon_rank": 16,
              "ridgon_implementation": "reference"}, "global_batch_size": 8,
              "train_mask_probability": 0.3,
              "training": {**common, "remove_unused_columns": False,
                           "average_tokens_across_devices": True}}
    mlm = tmp_path / "mlm"
    metrics = train_mlm(recipe, prepared, mlm)
    assert np.isfinite(metrics["eval_masked_nll"])
    metrics = train_mlm(recipe, prepared, mlm, str(mlm / "checkpoint-1"))
    assert read_json(mlm / "trainer_state.json")["global_step"] == 2
    encoder = load_sentence_model(mlm / "final", 16, "reference", "cpu")
    vectors = encoder.encode(["red bird", "blue fish"], normalize_embeddings=True)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=-1), 1, atol=1e-5)
    with pytest.raises(ValueError, match="position capacity"):
        load_sentence_model(mlm / "final", 64, "reference", "cpu")

    triplets = tmp_path / "triplets"
    Dataset.from_dict({"query": [f"red bird {'a ' * i}" for i in range(16)],
                       "positive": [f"bird sky {'the ' * i}" for i in range(16)],
                       "negative": [f"blue water {'and ' * i}" for i in range(16)]}).save_to_disk(triplets)
    retrieval_recipe = {"training": common, "implementation": "reference", "max_length": 16,
              "cache_mini_batch_size": 2, "scale": 20., "gather_across_devices": False,
              "data": {"validation_triplets": 2, "split_seed": 12, "max_train_triplets": 12}}
    retrieval = tmp_path / "retrieval"
    train_retrieval(retrieval_recipe, str(mlm / "final"), retrieval, str(triplets))
    train_retrieval(retrieval_recipe, str(mlm / "final"), retrieval, str(triplets), str(retrieval / "checkpoint-1"))
    assert (retrieval / "final" / "modules.json").is_file()

    fixture = {"queries": {"q1": "red bird", "q2": "blue fish"},
               "corpus": {"d1": "red bird", "d2": "blue fish", "d3": "green tree"},
               "relevant_docs": {"q1": ["d1"], "q2": ["d2"]}}
    local = tmp_path / "ir.json"
    write_json(local, fixture)
    result = evaluation.evaluate(str(retrieval / "final"), tmp_path / "scores", max_length=16,
             implementation="reference", device="cpu", local_ir=local, bf16=False)
    assert any(value == pytest.approx(1.) for key, value in read_json(result / "scores.json").items()
               if "ndcg@10" in key)

    import mteb
    from mteb.abstasks import AbsTaskRetrieval
    class LocalRetrieval(AbsTaskRetrieval):
        metadata = mteb.get_task("LEMBNarrativeQARetrieval").metadata.model_copy(update={
            "name": "RidgonLocalRetrievalFixture", "dataset": {"path": "local-fixture", "revision": "test"}})

        def load_data(self, **kwargs):
            self.corpus = {"test": {key: {"text": text} for key, text in fixture["corpus"].items()}}
            self.queries = {"test": fixture["queries"]}
            self.relevant_docs = {"test": {key: {doc: 1 for doc in docs} for key, docs in fixture["relevant_docs"].items()}}
            self.data_loaded = True

    monkeypatch.setattr(evaluation, "select_tasks", lambda *args: [LocalRetrieval()])
    result = evaluation.evaluate(str(retrieval / "final"), tmp_path / "mteb", max_length=16,
             implementation="reference", device="cpu", bf16=False)
    assert list((result / "cache").rglob("RidgonLocalRetrievalFixture.json"))
    assert (result / "evaluation.json").is_file()
