"""Document boundaries, AT loss, checkpoint round trips and Re-DocRED metrics."""
import copy

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
from transformers import BertConfig, BertModel, BertTokenizerFast, TrainingArguments

from experiments.redocred import (RelationCollator, RelationConfig, RelationModel,
    adaptive_threshold_loss, decode_predictions, prepare_document, relation_metrics,
    training_facts, window_starts, load_encoder, trainer_loss, metric_function)
from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM
from experiments.train_redocred import RelationTrainer


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                              "*", "alice", "bob", "met", "works", "here", "."]))
    return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)


def document(title="doc"):
    return {"title": title, "sents": [["Alice", "met", "Bob", "."], ["Alice", "works", "here", "."]],
            "vertexSet": [[{"name": "Alice", "sent_id": 0, "pos": [0, 1]},
                           {"name": "Alice", "sent_id": 1, "pos": [0, 1]}],
                          [{"name": "Bob", "sent_id": 0, "pos": [2, 3]}]],
            "labels": [{"h": 0, "t": 1, "r": "P1", "evidence": [0]},
                       {"h": 0, "t": 1, "r": "P2", "evidence": [0]}]}


def test_window_coverage_and_entity_alignment(tokenizer):
    for length in (1, 510, 511, 512, 1020, 1024, 1537):
        starts = window_starts(length, 510, 256)
        coverage = np.zeros(length)
        for start in starts:
            coverage[start:min(start + 510, length)] += 1
        assert coverage.min() >= 1
        assert starts == sorted(set(starts))
    with pytest.raises(ValueError):
        window_starts(100, 10, 11)
    row = prepare_document(document(), tokenizer, {"NA": 0, "P1": 1, "P2": 2}, 0)
    assert [row["input_ids"][i] for e in row["entities"] for i in e] == [5, 5, 5]
    collator = RelationCollator(tokenizer, window_size=8, stride=3, num_labels=3)
    batch = collator([row])
    assert len(batch["input_ids"]) > 1
    assert set(batch["token_map"].flatten().tolist()) == set(range(len(row["input_ids"]) + 1))
    assert batch["labels"].tolist() == [[[0, 1, 1], [1, 0, 0]]]
    assert batch["mention_indices"][0, 0, :2].tolist() == [i + 1 for i in row["entities"][0]]
    broken = document()
    broken["vertexSet"][0][0]["pos"] = [0, 10]
    with pytest.raises(ValueError, match="span"):
        prepare_document(broken, tokenizer, {"P1": 1, "P2": 2}, 0)


def test_at_loss_against_published_formula_and_accumulation():
    scores = torch.tensor([[[.1, .5, -.2, .3], [.4, -.5, -.1, .2]],
                           [[-.2, .6, .5, .1], [.3, .4, .2, .1]]], requires_grad=True)
    labels = torch.tensor([[[0, 1, 0, 1], [1, 0, 0, 0]],
                           [[0, 1, 1, 0], [-100, -100, -100, -100]]])
    saved = labels.clone()
    terms = []
    for row, target in zip(scores.flatten(0, 1), labels.flatten(0, 1)):
        if target[0] == -100:
            continue
        pos = torch.where(target[1:] == 1)[0] + 1
        neg = torch.where(target[1:] == 0)[0] + 1
        positive = sum((torch.logsumexp(row[torch.cat((pos, pos.new_tensor([0])))], 0) - row[r]
                        for r in pos), row.new_zeros(()))
        terms.append(positive + torch.logsumexp(row[torch.cat((neg, neg.new_tensor([0])))], 0) - row[0])
    expected = torch.stack(terms).mean()
    actual = adaptive_threshold_loss(scores, labels)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, scores, retain_graph=True)[0],
                               torch.autograd.grad(expected, scores)[0])
    accumulated = sum(adaptive_threshold_loss(scores[i:i+1], labels[i:i+1], 12) for i in range(2))
    torch.testing.assert_close(accumulated, actual)
    assert torch.equal(labels, saved)
    bf16 = scores.detach().bfloat16().requires_grad_()
    loss = adaptive_threshold_loss(bf16, labels)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(bf16.grad).all()


def test_model_batch_padding_gradients_and_save_load(tokenizer, tmp_path):
    torch.manual_seed(4)
    config = BertConfig(vocab_size=12, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=24, max_position_embeddings=32,
                        hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
    encoder = BertModel(config, add_pooling_layer=False)
    model = RelationModel(RelationConfig(encoder_config=config.to_dict(), block_size=4,
                                        pair_chunk_size=1, num_labels=3), encoder).eval()
    rel2id = {"NA": 0, "P1": 1, "P2": 2}
    first = prepare_document(document(), tokenizer, rel2id, 0)
    other = document("other")
    other["vertexSet"].append([{"name": "here", "sent_id": 1, "pos": [2, 3]}])
    second = prepare_document(other, tokenizer, rel2id, 1)
    collator = RelationCollator(tokenizer, window_size=12, stride=6, num_labels=3)
    batch = collator([first, second])
    output = model(**batch)
    separate = model(**collator([first]))["logits"]
    torch.testing.assert_close(output["logits"][0, :2], separate[0], atol=2e-6, rtol=2e-5)
    output["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    target = tmp_path / "export"
    model.save_pretrained(target)
    restored = RelationModel.from_pretrained(target).eval()
    torch.testing.assert_close(restored(**batch)["logits"], output["logits"])


def test_memsolve_mlm_weights_are_loaded_into_relation_encoder(tmp_path, tokenizer):
    config = MemSolveBertConfig(vocab_size=12, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=48, max_position_embeddings=32,
        memsolve_rank=16, memsolve_implementation="reference", memsolve_short_conv_kernel=3)
    mlm = MemSolveBertForMaskedLM(config).eval()
    target = tmp_path / "mlm"
    mlm.save_pretrained(target)
    encoder, info = load_encoder(target, "reference")
    assert not info["missing_keys"]
    for name, value in mlm.bert.state_dict().items():
        torch.testing.assert_close(encoder.state_dict()[name], value)
    row = prepare_document(document(), tokenizer, {"NA": 0, "P1": 1, "P2": 2}, 0)
    model = RelationModel(RelationConfig(encoder_config=config.to_dict(), block_size=8, num_labels=3), encoder)
    result = model(**RelationCollator(tokenizer, window_size=24, stride=12, num_labels=3)([row]))
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_metrics_training_fact_exclusion_and_multilabel_decoding():
    doc = document()
    doc["vertexSet"][1][0]["sent_id"] = 1
    doc["vertexSet"][0] = doc["vertexSet"][0][:1]
    train = copy.deepcopy(doc)
    train["labels"] = train["labels"][:1]
    logits = np.array([[[0., 2., 1.], [0., -1., -2.], [99., 99., 99.]]])
    predictions = decode_predictions(logits, np.array([[0]]), [doc], {0: "NA", 1: "P1", 2: "P2"}, 0)
    assert len(predictions) == 2
    assert all(p["h_idx"] == 0 and p["t_idx"] == 1 for p in predictions)
    metrics = relation_metrics(predictions + predictions[:1], [doc], training_facts([train]))
    assert metrics["f1"] == 1.
    assert metrics["ign_f1"] == pytest.approx(2 / (2 + 1e-5))
    assert metrics["inter_f1"] == 1. and metrics["intra_gold"] == 0
    assert len(decode_predictions(logits, [[0]], [doc], {0: "NA", 1: "P1", 2: "P2"}, 1)) == 1
    assert relation_metrics([], [doc], set())["f1"] == 0.
    no_gold = copy.deepcopy(doc)
    no_gold["labels"] = []
    assert relation_metrics([], [no_gold], set())["ign_f1"] == 0.


def test_trainer_variable_pairs_accumulation_and_prediction(tokenizer, tmp_path):
    config = BertConfig(vocab_size=12, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=24, max_position_embeddings=32)
    model = RelationModel(RelationConfig(encoder_config=config.to_dict(), block_size=4, num_labels=3),
                          BertModel(config, add_pooling_layer=False))
    docs = [document(f"doc-{i}") for i in range(3)]
    docs[1]["vertexSet"].append([{"name": "here", "sent_id": 1, "pos": [2, 3]}])
    rel2id = {"NA": 0, "P1": 1, "P2": 2}
    rows = [prepare_document(d, tokenizer, rel2id, i) for i, d in enumerate(docs)]
    trainer = RelationTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path / "run"), use_cpu=True, per_device_train_batch_size=1,
        per_device_eval_batch_size=2, gradient_accumulation_steps=2, num_train_epochs=1,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", remove_unused_columns=False, label_names=["labels"],
        report_to=[], disable_tqdm=True, dataloader_pin_memory=False),
        train_dataset=rows, eval_dataset=rows,
        data_collator=RelationCollator(tokenizer, window_size=12, stride=6, num_labels=3),
        compute_loss_func=trainer_loss,
        compute_metrics=metric_function(docs, set(), {v: k for k, v in rel2id.items()}, 0))
    result = trainer.train()
    assert np.isfinite(result.training_loss)
    assert trainer.state.global_step == 2
    predictions = trainer.predict(rows)
    scores, ids = predictions.predictions
    assert sorted(ids.flatten().tolist()) == [0, 1, 2]
    assert scores.shape == (3, 6, 3)
    assert "test_ign_f1" in predictions.metrics
    assert trainer.state.best_model_checkpoint is not None
    trainer.args.per_device_eval_batch_size = 1
    single = trainer.predict(rows)
    assert single.metrics["test_loss"] == pytest.approx(predictions.metrics["test_loss"], abs=1e-6)
