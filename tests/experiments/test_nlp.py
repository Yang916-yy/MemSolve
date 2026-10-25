"""Offline NLP checks: MLM, GLUE, retrieval, evaluation, and resume."""
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
    recipe = read_json(Path(__file__).resolve().parents[2] / "experiments/configs/memsolve_bert_mlm.json")
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
             tokenizer_revision="main", document_policy="truncate",
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
    assert metadata["train"][0]["token_counts"]["content_tokens"] > 0
    assert metadata["validation"][0]["token_counts"]["masked_targets"] > 0
    validation, _, _ = load_prepared(prepared, splits=("validation",))
    assert set(validation) == {"validation"}
    path = next((prepared / "train").glob("*/manifest.json"))
    manifest = read_json(path)
    manifest["num_shards"] = 2
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Incomplete"):
        load_prepared(prepared)


def test_document_windows_preserve_content_and_boundaries(tokenizer):
    from experiments.prepare_c4 import tokenized_documents
    tok, _ = tokenizer
    texts = ["the red bird " * 9, "blue fish water " * 7]
    rows = list(tokenized_documents([{"text": t} for t in texts], tok, 8, False, .15, 17,
                                   document_policy="split"))
    flattened = [x for row in rows for x, special, valid in
                 zip(row["input_ids"], row["special_tokens_mask"], row["attention_mask"])
                 if valid and not special]
    expected = sum([tok(t, add_special_tokens=False)["input_ids"] for t in texts], [])
    assert flattened == expected
    red, blue = tok.convert_tokens_to_ids(["red", "blue"])
    assert all(not (red in row["input_ids"] and blue in row["input_ids"]) for row in rows)
    assert all(len(row["input_ids"]) == 8 for row in rows)
    prefixes = list(tokenized_documents([{"text": t} for t in texts], tok, 8, False, .15, 17))
    assert len(prefixes) == 2


def test_prepared_rejects_mislabeled_split(prepared):
    path = next((prepared / "validation").glob("*/manifest.json"))
    manifest = read_json(path)
    manifest["split"] = "train"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="mislabeled"):
        load_prepared(prepared)


@pytest.fixture
def streaming_manifest(prepared, tmp_path):
    metadata = read_json(next((prepared / "validation").glob("*/manifest.json")))
    files = []
    for shard in range(4):
        path = tmp_path / f"stream-{shard}.jsonl"
        texts = [" ".join("red" if (i >> bit) & 1 else "blue" for bit in range(5))
                 for i in range(shard * 8, (shard + 1) * 8)]
        path.write_text("\n".join(json.dumps({"text": text, "url": "unused"}) for text in texts))
        stat = path.stat()
        files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    path = tmp_path / "stream.json"
    write_json(path, {"format": "memsolve-local-text-stream-v1", "files": files,
                     "num_source_files": 4, **{key: metadata[key] for key in
                     ("max_length", "packing", "tokenizer_hash")}})
    return path


def test_streaming_worker_and_rank_shards_are_disjoint(prepared, streaming_manifest):
    from accelerate.data_loader import prepare_data_loader
    from transformers import default_data_collator
    from experiments.nlp import load_mlm_data
    seen = []
    for rank in range(2):
        data, _, _ = load_mlm_data(prepared, streaming_manifest, shuffle_buffer=3)
        loader = torch.utils.data.DataLoader(data["train"], batch_size=2, num_workers=2,
                                             collate_fn=default_data_collator)
        loader = prepare_data_loader(loader, num_processes=2, process_index=rank,
                                     put_on_device=False, dispatch_batches=False, split_batches=False)
        rows = [tuple(row) for batch in loader for row in batch["input_ids"].tolist()]
        assert len(rows) == len(set(rows)) == 16
        seen.append(set(rows))
    assert not seen[0] & seen[1] and len(seen[0] | seen[1]) == 32
    manifest = read_json(streaming_manifest)
    Path(manifest["files"][0]["path"]).write_text('{"text":"changed"}\n')
    with pytest.raises(ValueError, match="file changed"):
        load_mlm_data(prepared, streaming_manifest)


def test_streaming_mlm_train_and_resume(prepared, streaming_manifest, tokenizer, tmp_path):
    from experiments.train_mlm import train, preflight
    torch.set_num_threads(2)
    recipe = {"model": {"vocab_size": len(tokenizer[0]), "hidden_size": 32,
              "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 64,
              "max_position_embeddings": 32, "memsolve_rank": 16,
              "memsolve_implementation": "reference"}, "global_batch_size": 4,
              "train_mask_probability": .3, "streaming_shuffle_buffer": 3,
              "training": {"max_steps": 2, "per_device_train_batch_size": 2,
                  "per_device_eval_batch_size": 4, "learning_rate": 1e-4,
                  "optim": "adamw_torch", "use_cpu": True, "report_to": "none",
                  "dataloader_num_workers": 2, "dataloader_persistent_workers": True,
                  "save_steps": 1, "disable_tqdm": True, "remove_unused_columns": False,
                  "accelerator_config": {"dispatch_batches": False, "split_batches": False}}}
    plan = preflight(recipe, prepared, world_size=2, streaming_manifest=streaming_manifest)
    assert plan["train_rows"] is None and plan["train_source_files"] == 4
    output = tmp_path / "stream-run"
    metrics = train(recipe, prepared, output, streaming_manifest=streaming_manifest)
    assert np.isfinite(metrics["eval_masked_nll"])
    metrics = train(recipe, prepared, output, resume=output / "checkpoint-1",
                    streaming_manifest=streaming_manifest)
    assert read_json(output / "trainer_state.json")["global_step"] == 2
    assert np.isfinite(metrics["eval_masked_nll"])


@pytest.mark.parametrize("invalid", [None, "", 17])
def test_retrieval_rejects_invalid_text(tmp_path, invalid):
    pytest.importorskip("sentence_transformers", minversion="5.7")
    from experiments.train_retrieval import retrieval_data
    path = tmp_path / "triplets"
    Dataset.from_dict({"query": [invalid] * 4,
                       "positive": ["a bird"] * 4,
                       "negative": ["a fish"] * 4}).save_to_disk(str(path))
    with pytest.raises(ValueError, match="strings|null or empty"):
        retrieval_data({"validation_triplets": 1, "split_seed": 12}, str(path))


def test_offline_mlm_retrieval_and_mteb(prepared, tokenizer, tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers", minversion="5.7")
    pytest.importorskip("mteb", minversion="2.21")
    from experiments.train_mlm import train as train_mlm, preflight, evaluate as evaluate_mlm
    from experiments.train_retrieval import train as train_retrieval
    from experiments import evaluate_embeddings as evaluation
    torch.set_num_threads(2)
    common = {"max_steps": 2, "per_device_train_batch_size": 4, "per_device_eval_batch_size": 3,
              "learning_rate": 1e-4, "optim": "adamw_torch", "use_cpu": True,
              "bf16": False, "report_to": "none", "dataloader_num_workers": 0,
              "save_steps": 1, "logging_steps": 1, "disable_tqdm": True, "seed": 17,
              "dataloader_drop_last": True}
    recipe = {"model": {"vocab_size": len(tokenizer[0]), "hidden_size": 64,
              "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 128,
              "max_position_embeddings": 32, "memsolve_rank": 16,
              "memsolve_implementation": "reference"}, "global_batch_size": 8,
              "train_mask_probability": 0.3,
              "training": {**common, "remove_unused_columns": False,
                           "eval_strategy": "steps", "eval_steps": 1,
                           "load_best_model_at_end": True, "metric_for_best_model": "masked_nll",
                           "greater_is_better": False,
                           "average_tokens_across_devices": True}}
    plan = preflight(recipe, prepared, world_size=2)
    assert plan["gradient_accumulation_steps"] == 1
    assert plan["parameters"] > 0
    assert plan["full_batch_token_slots"] == 2 * 8 * 16
    with pytest.raises(ValueError):
        preflight(recipe, prepared, world_size=0)
    mlm = tmp_path / "mlm"
    metrics = train_mlm(recipe, prepared, mlm)
    assert np.isfinite(metrics["eval_masked_nll"])
    validation, _, _ = load_prepared(prepared, splits=("validation",))
    target_count = sum(label != -100 for row in validation["validation"] for label in row["labels"])
    assert metrics["eval_masked_tokens"] == target_count
    run = read_json(mlm / "run.json")["experiment"]
    assert run["dataloader_drop_last"] is True
    assert run["eval_dataloader_drop_last"] is False
    exported = read_json(mlm / "final" / "config.json")
    assert exported["model_type"] == "memsolve_bert"
    assert exported["architectures"] == ["MemSolveBertForMaskedLM"]
    selection = read_json(mlm / "final" / "selection.json")
    assert selection["selection"] == "best_validation"
    assert selection["best_checkpoint"] is not None
    independent = evaluate_mlm(recipe, prepared, mlm / "final", tmp_path / "mlm-eval")
    assert independent["eval_masked_tokens"] == target_count
    assert independent["eval_masked_nll"] == pytest.approx(metrics["eval_masked_nll"], abs=1e-6)
    continued = tmp_path / "continued"
    train_mlm(recipe, prepared, continued, init_checkpoint=mlm / "final")
    assert read_json(continued / "run.json")["experiment"]["initialization"]["sha256"]
    invalid = {**recipe, "model": {**recipe["model"], "memsolve_rank": 32}}
    with pytest.raises(ValueError, match="differs"):
        preflight(invalid, prepared, checkpoint=mlm / "final")
    metrics = train_mlm(recipe, prepared, mlm, str(mlm / "checkpoint-1"))
    assert read_json(mlm / "trainer_state.json")["global_step"] == 2
    encoder = load_sentence_model(mlm / "final", 16, "reference", "cpu")
    vectors = encoder.encode(["red bird", "blue fish"], normalize_embeddings=True)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=-1), 1, atol=1e-5)
    extended = load_sentence_model(mlm / "final", 64, "reference", "cpu")
    assert extended.max_seq_length == 64
    assert extended[0].auto_model.config.position_embedding_type == "rove"

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
            "name": "MemSolveLocalRetrievalFixture", "dataset": {"path": "local-fixture", "revision": "test"}})

        def load_data(self, **kwargs):
            self.corpus = {"test": {key: {"text": text} for key, text in fixture["corpus"].items()}}
            self.queries = {"test": fixture["queries"]}
            self.relevant_docs = {"test": {key: {doc: 1 for doc in docs} for key, docs in fixture["relevant_docs"].items()}}
            self.data_loaded = True

    monkeypatch.setattr(evaluation, "select_tasks", lambda *args: [LocalRetrieval()])
    result = evaluation.evaluate(str(retrieval / "final"), tmp_path / "mteb", max_length=16,
             implementation="reference", device="cpu", bf16=False)
    assert list((result / "cache").rglob("MemSolveLocalRetrievalFixture.json"))
    assert (result / "evaluation.json").is_file()


def test_stock_bert_baseline_evaluation(prepared, tokenizer, tmp_path):
    pytest.importorskip("sentence_transformers", minversion="5.7")
    from transformers import BertConfig, BertForMaskedLM
    from experiments.train_mlm import evaluate
    checkpoint = tmp_path / "bert"
    model = BertForMaskedLM(BertConfig(vocab_size=len(tokenizer[0]), hidden_size=32,
                intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32))
    model.save_pretrained(checkpoint)
    tokenizer[0].save_pretrained(checkpoint)
    recipe = {"model": {"memsolve_implementation": "reference"}, "train_mask_probability": .3,
              "training": {"use_cpu": True, "bf16": False, "report_to": "none",
                           "per_device_eval_batch_size": 3, "remove_unused_columns": False,
                           "dataloader_drop_last": True}}
    metrics = evaluate(recipe, prepared, checkpoint, tmp_path / "baseline-eval")
    assert np.isfinite(metrics["eval_masked_nll"])
    validation, _, _ = load_prepared(prepared, splits=("validation",))
    target_count = sum(label != -100 for row in validation["validation"] for label in row["labels"])
    assert metrics["eval_masked_tokens"] == target_count
    manifest = read_json(next((tmp_path / "baseline-eval").glob("*/evaluation.json")))
    assert manifest["dataloader_drop_last"] is False
    assert manifest["validation_rows"] == 8
    embedding = load_sentence_model(checkpoint, 16, "reference", "cpu")
    assert embedding[0].auto_model.config.model_type == "bert"
    assert not hasattr(embedding[0].auto_model.config, "memsolve_implementation")
    with pytest.raises(ValueError, match="position capacity"):
        load_sentence_model(checkpoint, 64, "reference", "cpu")


@pytest.mark.parametrize("position_type", ["absolute", "rove", "none"])
def test_prepared_length_check_respects_position_type(prepared, tokenizer, position_type):
    from experiments.nlp import load_prepared, position_capacity
    from experiments.train_mlm import check_data
    from integrations.transformers import MemSolveBertConfig
    _, manifests, _ = load_prepared(prepared)
    config = MemSolveBertConfig(vocab_size=len(tokenizer[0]), hidden_size=32,
        num_attention_heads=2, memsolve_rank=16, max_position_embeddings=8,
        position_embedding_type=position_type)
    if position_type == "absolute":
        assert position_capacity(config) == 8
        with pytest.raises(ValueError, match="position capacity"):
            check_data(config, manifests, tokenizer[0])
    else:
        assert position_capacity(config) is None
        check_data(config, manifests, tokenizer[0])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_model", [False, True])
def test_mlm_compile_graph_accumulation_eval_and_reload(tmp_path, compile_model):
    """Graph replay preserves weighted accumulation, updates and the HF export."""
    from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM
    from experiments.nlp import prepare_mlm_graph
    torch.manual_seed(71)
    config = MemSolveBertConfig(vocab_size=32, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, memsolve_rank=16,
        memsolve_implementation="cuda", hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
    eager = MemSolveBertForMaskedLM(config).cuda()
    graph = MemSolveBertForMaskedLM(config).cuda()
    graph.load_state_dict(eager.state_dict())
    batches = []
    for index in range(2):
        ids = torch.randint(5, 32, (2, 160), device="cuda")
        lengths = torch.tensor([[71 + index], [141 + index]], device="cuda")
        mask = (torch.arange(160, device="cuda")[None, :] < lengths).long()
        positions = (torch.arange(160, device="cuda")[None, :] * (index + 1) + 31 * index).expand(2, -1).contiguous()
        labels = ids.clone(); labels[:, :10 * (index + 1)] = -100; labels[mask == 0] = -100
        batches.append(dict(input_ids=ids, attention_mask=mask, token_type_ids=torch.zeros_like(ids),
                            labels=labels, position_ids=positions))
    rng = torch.cuda.get_rng_state().clone()
    prepare_mlm_graph(graph, batches[0], compile_model=compile_model)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    count = sum(batch['labels'].ne(-100).sum() for batch in batches)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True) for m in (eager, graph)]
    for _ in range(2):
        for model, optimizer in zip((eager, graph), optimizers):
            optimizer.zero_grad(set_to_none=True)
            for batch in batches:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if model is eager:
                        output = model(**{k:v for k,v in batch.items() if k != 'labels'})
                        loss = masked_loss(output, batch['labels'], count)
                    else:
                        loss = model(**batch, num_items_in_batch=count).loss
                loss.backward()
        for (name, p), q in zip(eager.named_parameters(), graph.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=.025, atol=2e-4, msg=name)
        for optimizer in optimizers: optimizer.step()
    graph.eval(); eager.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = graph(**batches[0]); expected = eager(**batches[0])
    assert "loss" in result and torch.isfinite(result["loss"])
    torch.testing.assert_close(result.logits, expected.logits, rtol=.025, atol=.003)
    graph.save_pretrained(tmp_path / 'model')
    restored = MemSolveBertForMaskedLM.from_pretrained(tmp_path / 'model').cuda()
    graph.load_state_dict(restored.state_dict())
    graph.train()
    empty = {**batches[0], 'labels': torch.full_like(batches[0]['labels'], -100)}
    graph.zero_grad(set_to_none=True)
    loss = graph(**empty, num_items_in_batch=0).loss
    loss.backward()
    assert loss.item() == 0
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in graph.parameters())
    assert graph._memsolve_graph_info['replays'] == 5
    with pytest.raises(ValueError, match='shape/dtype/device'):
        graph(**{k:v[:1] for k,v in batches[0].items()})


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mlm_compile_graph_dropout_advances():
    from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM
    from experiments.nlp import prepare_mlm_graph
    model = MemSolveBertForMaskedLM(MemSolveBertConfig(vocab_size=32, hidden_size=32,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, memsolve_rank=16,
        memsolve_implementation='cuda', hidden_dropout_prob=.3)).cuda()
    ids = torch.randint(5, 32, (2, 40), device='cuda')
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                 token_type_ids=torch.zeros_like(ids), labels=ids.clone())
    prepare_mlm_graph(model, batch)
    values = []
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        values.append(loss.item())
    assert all(np.isfinite(values)) and len(set(values)) > 1


def test_metrics_and_paper_recipe():
    from experiments.train_glue import glue_metrics

    scores = np.array([[2., 0.], [0., 2.], [2., 0.], [0., 2.]])
    labels = np.array([0, 1, 0, 1])
    assert glue_metrics("mrpc", scores, labels) == {"accuracy": 1., "f1": 1.}
    assert glue_metrics("cola", scores, labels)["matthews_correlation"] == 1.
    corr = glue_metrics("stsb", np.array([[1.], [3.], [5.]]), np.array([0., 2., 4.]))
    assert corr["pearson"] == pytest.approx(1.)
    assert corr["spearmanr"] == pytest.approx(1.)
    recipe = read_json(Path(__file__).resolve().parents[2] / "experiments/configs/memsolve_bert_glue.json")
    assert {task for task, r in recipe["tasks"].items() if r["init"] == "mnli"} == {"mrpc", "rte", "stsb"}
    assert len(recipe["tasks"]) == 8


def test_composer_decay_conversion_at_multiple_learning_rates():
    from experiments.train_glue import torch_weight_decay

    task = {"optimizer": "composer_adamw", "lr": 5e-5, "weight_decay": 5e-6}
    for fraction in (0., .06, 1., .2):
        parameter = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=task["lr"] * fraction, weight_decay=torch_weight_decay(task))
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        torch.testing.assert_close(parameter.detach(), torch.tensor([2 * (1 - fraction * task["weight_decay"])], dtype=torch.float64))
    assert torch_weight_decay({**task, "optimizer": "adamw"}) == task["weight_decay"]
