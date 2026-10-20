"""Re-DocRED data, hidden-state relation head, and official relation metrics.

ATLOP without localized context pooling. The loss and grouped classifier are
adapted from DREEAM (MIT); metrics follow Re-DocRED (MIT). See THIRD_PARTY_NOTICES.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import urllib.request

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from experiments.nlp import read_json, write_json, tokenizer_digest
import integrations.transformers  # Register MemSolve with Hugging Face AutoModel.


DATA_REVISION = "ccfb54f5ddf5836027c87badda10f6dfc56efaac"
DATA_URL = f"https://raw.githubusercontent.com/tonytan48/Re-DocRED/{DATA_REVISION}/data"
SPLITS = ("train", "dev", "test")


def prepare_document(document, tokenizer, rel2id, doc_id):
    """Mark mention boundaries before WordPiece tokenization; retain every token."""
    entities, sentences = document["vertexSet"], document["sents"]
    if len(entities) < 2 or "labels" not in document:
        raise ValueError(f"{document['title']}: requires entities and revised relation labels")
    starts, ends = set(), set()
    for entity in entities:
        if not entity:
            raise ValueError("Every entity must have at least one mention")
        for mention in entity:
            sid, (start, end) = mention["sent_id"], mention["pos"]
            if not (0 <= sid < len(sentences) and 0 <= start < end <= len(sentences[sid])):
                raise ValueError(f"Invalid mention span in {document['title']}: {mention}")
            starts.add((sid, start))
            ends.add((sid, end))
    marker = tokenizer.convert_tokens_to_ids("*")
    if marker == tokenizer.unk_token_id:
        raise ValueError("The tokenizer must contain the ATLOP '*' entity marker")
    tokens, positions = [], {}
    for sid, sentence in enumerate(sentences):
        for index, word in enumerate(sentence):
            if (sid, index) in starts:
                positions[sid, index] = len(tokens)
                tokens.append(marker)
            pieces = tokenizer.encode(word, add_special_tokens=False)
            tokens.extend(pieces or [tokenizer.unk_token_id])
            if (sid, index + 1) in ends:
                tokens.append(marker)
    mention_positions = [[positions[m["sent_id"], m["pos"][0]] for m in entity] for entity in entities]
    relations = set()
    for label in document["labels"]:
        h, t, relation = label["h"], label["t"], label["r"]
        if not (0 <= h < len(entities) and 0 <= t < len(entities) and h != t):
            raise ValueError(f"Invalid entity pair in {document['title']}")
        if relation not in rel2id or rel2id[relation] == 0:
            raise ValueError(f"Unknown relation: {relation}")
        relations.add((h, t, rel2id[relation]))
    return {"doc_ids": doc_id, "input_ids": tokens, "entities": mention_positions,
            "relations": sorted(relations)}


def prepare_data(root, tokenizer, download=False):
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    documents, hashes = {}, {}
    for split in SPLITS:
        path = raw / f"{split}_revised.json"
        if not path.exists() and download:
            temporary = path.with_suffix(".download")
            with urllib.request.urlopen(f"{DATA_URL}/{path.name}", timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        documents[split] = read_json(path)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        titles = [d["title"] for d in documents[split]]
        if len(set(titles)) != len(titles):
            raise ValueError(f"Duplicate document titles in {split}")
    relations = sorted({label["r"] for d in documents["train"] for label in d["labels"]})
    if len(relations) != 96:
        raise ValueError(f"Re-DocRED requires 96 training relations; found {len(relations)}")
    rel2id = {"NA": 0, **{r: i + 1 for i, r in enumerate(relations)}}
    source = {"format_version": 1, "raw_sha256": hashes, "tokenizer_hash": tokenizer_digest(tokenizer)}
    features = {}
    for split, docs in documents.items():
        rows = [prepare_document(d, tokenizer, rel2id, i) for i, d in enumerate(docs)]
        features[split] = rows
        write_json(root / f"{split}.json", rows)
    manifest = {"source": source, "upstream_revision": DATA_REVISION, "rel2id": rel2id,
                "documents": {s: len(v) for s, v in features.items()},
                "max_tokens": {s: max(len(d["input_ids"]) for d in v) for s, v in features.items()},
                "over_510_tokens": {s: sum(len(d["input_ids"]) > 510 for d in v) for s, v in features.items()}}
    manifest["prepared_sha256"] = {
        s: hashlib.sha256((root / f"{s}.json").read_bytes()).hexdigest() for s in SPLITS}
    write_json(root / "manifest.json", manifest)
    return manifest


def load_data(root, tokenizer):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest["source"]["tokenizer_hash"] != tokenizer_digest(tokenizer):
        raise ValueError("Prepared data use a different tokenizer; prepare into a separate directory")
    features, documents = {}, {}
    for split in SPLITS:
        prepared, raw = root / f"{split}.json", root / "raw" / f"{split}_revised.json"
        for path, expected in ((prepared, manifest["prepared_sha256"][split]),
                               (raw, manifest["source"]["raw_sha256"][split])):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Data changed after preparation: {path}")
        features[split], documents[split] = read_json(prepared), read_json(raw)
    return features, documents, manifest


def window_starts(length, capacity, stride):
    if not 0 < stride <= capacity:
        raise ValueError("Window stride must be positive and no larger than content capacity")
    last = max(0, length - capacity)
    return sorted(set([*range(0, last + 1, stride), last]))


class RelationCollator:
    """Batch documents as overlapping BERT windows, plus global mention indices."""

    def __init__(self, tokenizer, window_size=512, stride=256, num_labels=97):
        if tokenizer.num_special_tokens_to_add(pair=False) != 2:
            raise ValueError("This pipeline expects BERT [CLS]/[SEP] framing")
        if window_size < 4:
            raise ValueError("window_size must leave room for content and two special tokens")
        window_starts(1, window_size - 2, stride)
        self.tokenizer, self.window_size, self.stride = tokenizer, window_size, stride
        self.num_labels = num_labels

    def __call__(self, examples):
        windows, maps, offsets = [], [], []
        total = 0
        for row in examples:
            offsets.append(total)
            ids = row["input_ids"]
            for start in window_starts(len(ids), self.window_size - 2, self.stride):
                stop = min(start + self.window_size - 2, len(ids))
                windows.append(self.tokenizer.build_inputs_with_special_tokens(ids[start:stop]))
                maps.append([0, *range(total + start + 1, total + stop + 1), 0])
            total += len(ids)
        length = max(map(len, windows))
        input_ids = torch.full((len(windows), length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        token_map = torch.zeros_like(input_ids)
        for i, (ids, mapping) in enumerate(zip(windows, maps)):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attention_mask[i, :len(ids)] = 1
            token_map[i, :len(ids)] = torch.tensor(mapping)
        counts = torch.bincount(token_map.flatten(), minlength=total + 1).float().clamp_min(1)
        max_entities = max(len(row["entities"]) for row in examples)
        max_mentions = max(len(e) for row in examples for e in row["entities"])
        max_pairs = max_entities * (max_entities - 1)
        mention_indices = torch.zeros((len(examples), max_entities, max_mentions), dtype=torch.long)
        mention_mask = torch.zeros_like(mention_indices, dtype=torch.bool)
        # Padded entities pool a single zero vector, avoiding logsumexp(-inf, ...).
        mention_mask[:, :, 0] = True
        pair_indices = torch.zeros((len(examples), max_pairs, 2), dtype=torch.long)
        labels = torch.full((len(examples), max_pairs, self.num_labels), -100, dtype=torch.long)
        for b, (row, offset) in enumerate(zip(examples, offsets)):
            for e, mentions in enumerate(row["entities"]):
                mention_indices[b, e, :len(mentions)] = torch.tensor(mentions) + offset + 1
                mention_mask[b, e, :len(mentions)] = True
            pairs = [(h, t) for h in range(len(row["entities"])) for t in range(len(row["entities"])) if h != t]
            pair_indices[b, :len(pairs)] = torch.tensor(pairs)
            labels[b, :len(pairs)] = 0
            lookup = {pair: i for i, pair in enumerate(pairs)}
            for h, t, r in row["relations"]:
                labels[b, lookup[h, t], r] = 1
            labels[b, :len(pairs), 0] = labels[b, :len(pairs), 1:].sum(-1).eq(0)
        # Trainer infers batch size from the first tensor: documents, not windows.
        return {"doc_ids": torch.tensor([row["doc_ids"] for row in examples]).unsqueeze(-1),
                "input_ids": input_ids, "attention_mask": attention_mask, "token_map": token_map,
                "token_counts": counts, "mention_indices": mention_indices, "mention_mask": mention_mask,
                "pair_indices": pair_indices, "labels": labels}


def adaptive_threshold_loss(logits, labels, num_items_in_batch=None):
    """DREEAM AT loss with FP32 reductions, pair padding, and accumulation scaling."""
    valid = labels[..., 0].ne(-100)
    scores = logits.float()[valid]
    positives = labels[valid].bool().clone()
    positives[:, 0] = False
    positive_or_threshold = positives.clone()
    positive_or_threshold[:, 0] = True
    positive_logp = scores.masked_fill(~positive_or_threshold, -1e30).log_softmax(-1)
    negative_logp = scores.masked_fill(positives, -1e30).log_softmax(-1)
    loss = -(positive_logp * positives).sum(-1) - negative_logp[:, 0]
    # HF counts non--100 label entries across microbatches and DDP ranks.
    count = valid.sum() if num_items_in_batch is None else num_items_in_batch / labels.shape[-1]
    return loss.sum() / torch.as_tensor(count, device=loss.device).clamp_min(1)


def trainer_loss(outputs, labels, num_items_in_batch=None):
    return adaptive_threshold_loss(outputs["logits"], labels, num_items_in_batch)


class RelationConfig(PretrainedConfig):
    model_type = "memsolve_docre"

    def __init__(self, encoder_config=None, block_size=64, pair_chunk_size=256, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        self.block_size, self.pair_chunk_size = block_size, pair_chunk_size


class RelationModel(PreTrainedModel):
    """Shared BERT/MemSolve head; no localized attention pooling."""
    config_class = RelationConfig
    base_model_prefix = "encoder"

    def __init__(self, config, encoder=None):
        super().__init__(config)
        if encoder is None:
            specification = dict(config.encoder_config)
            model_type = specification.pop("model_type")
            encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **specification),
                                            add_pooling_layer=False)
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        if hidden % config.block_size or config.pair_chunk_size < 1:
            raise ValueError("block_size must divide hidden_size; pair_chunk_size must be positive")
        self.head_extractor = nn.Linear(hidden, hidden)
        self.tail_extractor = nn.Linear(hidden, hidden)
        self.bilinear = nn.Linear(hidden * config.block_size, config.num_labels)

    def forward(self, doc_ids, input_ids, attention_mask, token_map, token_counts,
                mention_indices, mention_mask, pair_indices, labels=None):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              output_attentions=False, return_dict=True).last_hidden_state
        # Average overlaps in FP32; index zero collects padding and special tokens.
        hidden = states.shape[-1]
        merged = states.new_zeros((token_counts.shape[0], hidden), dtype=torch.float32)
        merged = merged.index_add(0, token_map.flatten(), states.float().flatten(0, 1))
        merged = merged / token_counts[:, None]
        merged = torch.cat((merged.new_zeros((1, hidden)), merged[1:]))
        mentions = merged[mention_indices].masked_fill(~mention_mask[..., None], -torch.inf)
        entities = mentions.logsumexp(dim=2)
        heads, tails = self.head_extractor(entities).tanh(), self.tail_extractor(entities).tanh()
        batch = torch.arange(heads.shape[0], device=heads.device)[:, None]
        logits = []
        block = self.config.block_size
        for pairs in pair_indices.split(self.config.pair_chunk_size, dim=1):
            h = heads[batch, pairs[..., 0]].reshape(*pairs.shape[:2], hidden // block, block)
            t = tails[batch, pairs[..., 1]].reshape_as(h)
            products = (h.unsqueeze(-1) * t.unsqueeze(-2)).flatten(-3)
            logits.append(self.bilinear(products))
        scores = torch.cat(logits, dim=1).float()
        output = {"logits": scores, "doc_ids": doc_ids}
        if labels is not None:
            output = {"loss": adaptive_threshold_loss(scores, labels), **output}
        return output


def load_encoder(checkpoint, implementation):
    config = AutoConfig.from_pretrained(checkpoint)
    if config.model_type not in ("bert", "memsolve_bert"):
        raise ValueError("Use a BERT or MemSolve-BERT encoder checkpoint")
    if config.model_type == "memsolve_bert":
        config.memsolve_implementation = implementation
    encoder, info = AutoModel.from_pretrained(checkpoint, config=config, add_pooling_layer=False,
                                            output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError(f"Encoder checkpoint mismatch: {info}")
    if any(not key.startswith(("cls.", "pooler.")) for key in info["unexpected_keys"]):
        raise ValueError(f"Unexpected non-MLM weights in encoder checkpoint: {info}")
    return encoder, info


def decode_predictions(logits, doc_ids, documents, id2rel, max_relations=4):
    """TH decoding with the ATLOP recipe's top-four cap (0 means uncapped)."""
    predictions = []
    for scores, identifier in zip(logits, doc_ids):
        doc = documents[int(np.asarray(identifier).item())]
        pairs = [(h, t) for h in range(len(doc["vertexSet"])) for t in range(len(doc["vertexSet"])) if h != t]
        scores = scores[:len(pairs)]
        selected = scores > scores[:, :1]
        if max_relations > 0:
            cutoff = np.sort(scores, axis=-1)[:, -max_relations]
            selected &= scores >= cutoff[:, None]
        selected[:, 0] = False
        for pair_index, r in zip(*np.where(selected)):
            h, t = pairs[pair_index]
            predictions.append({"title": doc["title"], "h_idx": h, "t_idx": t, "r": id2rel[int(r)]})
    return predictions


def training_facts(documents):
    return {(h["name"], t["name"], label["r"])
            for doc in documents for label in doc["labels"]
            for h in doc["vertexSet"][label["h"]] for t in doc["vertexSet"][label["t"]]}


def relation_metrics(predictions, documents, facts):
    """Official F1/Ign F1 (annotated-train facts), plus sentence-distance subsets."""
    by_title = {doc["title"]: doc for doc in documents}
    gold = {(doc["title"], label["h"], label["t"], label["r"])
            for doc in documents for label in doc["labels"]}
    predicted = {(p["title"], p["h_idx"], p["t_idx"], p["r"]) for p in predictions}
    correct = gold & predicted
    seen = sum(any((a["name"], b["name"], r) in facts
                   for a in by_title[title]["vertexSet"][h] for b in by_title[title]["vertexSet"][t])
               for title, h, t, r in correct)
    precision = len(correct) / len(predicted) if predicted else 0.
    recall = len(correct) / len(gold) if gold else 0.
    ign_precision = (len(correct) - seen) / (len(predicted) - seen + 1e-5)

    def harmonic(p, r):
        return 2 * p * r / (p + r) if p + r else 0.

    metrics = {"f1": harmonic(precision, recall), "ign_f1": harmonic(ign_precision, recall),
               "precision": precision, "recall": recall}
    def distance(key):
        title, h, t, _ = key
        entities = by_title[title]["vertexSet"]
        return min(abs(a["sent_id"] - b["sent_id"]) for a in entities[h] for b in entities[t])

    for name, condition in (("intra", lambda d: d == 0), ("inter", lambda d: d >= 1),
                            ("inter_long", lambda d: d > 5)):
        truth = {k for k in gold if condition(distance(k))}
        guesses = {k for k in predicted if k[0] in by_title and condition(distance(k))}
        hits = len(truth & guesses)
        metrics[f"{name}_f1"] = 2 * hits / (len(truth) + len(guesses)) if truth or guesses else 0.
        metrics[f"{name}_gold"] = len(truth)
    return metrics


def metric_function(documents, facts, id2rel, max_relations):
    def compute(result):
        logits, ids = result.predictions
        predictions = decode_predictions(logits, ids, documents, id2rel, max_relations)
        metrics = relation_metrics(predictions, documents, facts)
        total_loss, pairs = 0., 0
        for scores, targets in zip(logits, result.label_ids):
            count = np.count_nonzero(targets[:, 0] != -100)
            total_loss += adaptive_threshold_loss(
                torch.from_numpy(scores[:count]), torch.from_numpy(targets[:count]),
                num_items_in_batch=targets.shape[-1]).item()
            pairs += count
        metrics["at_loss"] = total_loss / max(pairs, 1)
        return metrics
    return compute
