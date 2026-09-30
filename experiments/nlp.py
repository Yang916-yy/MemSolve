"""Shared experiment plumbing; model mathematics lives in ridgon.ball."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling, default_data_collator


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkpoint_digest(path):
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.suffix in (".json", ".safetensors", ".bin", ".txt"))
    if not files:
        raise ValueError("A local exported checkpoint is required")
    result = hashlib.sha256()
    for item in files:
        result.update(str(item.relative_to(path)).encode())
        with item.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                result.update(block)
    return result.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "datasets", "accelerate",
             "sentence-transformers", "mteb")
            if importlib.util.find_spec(name.replace("-", "_")) is not None}


def tokenizer_digest(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend["padding"] = backend["truncation"] = None
    return digest({"vocab": tokenizer.get_vocab(),
                   "special_tokens": tokenizer.special_tokens_map,
                   "backend": backend})


def accumulation_steps(global_batch, micro_batch, world_size):
    if min(global_batch, micro_batch, world_size) < 1 or global_batch % (micro_batch * world_size):
        raise ValueError("global_batch_size must be divisible by micro batch × world size")
    return global_batch // (micro_batch * world_size)


def check_training_precision(settings, implementation):
    if settings.get("fp16") or settings.get("bf16_full_eval") or settings.get("fp16_full_eval"):
        raise ValueError("Use bf16 AMP with FP32 parameters, not fp16/full_eval casts")
    if implementation == "cuda" and (not torch.cuda.is_available() or not settings.get("bf16")):
        raise ValueError("The cuda implementation requires a CUDA device and bf16 AMP")


def record_run(output, manifest, resume=None):
    """Rank zero validates before Trainer writes. Never overwrite an unrelated run."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output = Path(output)
    path = output / "run.json"
    if resume:
        if Path(resume).resolve().parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this output directory")
        if not (Path(resume) / "trainer_state.json").is_file():
            raise ValueError("Resume requires a Trainer checkpoint, not just exported weights")
        if not path.is_file() or read_json(path)["experiment"] != manifest["experiment"]:
            raise ValueError("Resume recipe/data differ from output/run.json")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; use --resume")
    if int(os.environ.get("RANK", "0")) == 0:
        write_json(path, manifest)


class MLMCollator:
    """HF stochastic training masks; prepared validation already has fixed labels."""
    def __init__(self, tokenizer, probability):
        self.training = DataCollatorForLanguageModeling(tokenizer, mlm_probability=probability)

    def __call__(self, examples):
        if "labels" in examples[0]:
            return default_data_collator(examples)
        return self.training(examples)


def freeze_mask(example, tokenizer, probability, seed):
    # Content-addressed masks survive different worker/batch/shard boundaries.
    row_seed = (int(digest(dict(example))[0:16], 16) + seed) % (2**63 - 1)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=probability, seed=row_seed)
    return {key: value[0].tolist() for key, value in collator([example]).items()}


def masked_loss(outputs, labels, num_items_in_batch=None):
    count = labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    count = torch.as_tensor(count, device=labels.device).clamp_min(1)
    return F.cross_entropy(outputs.logits.float().flatten(0, 1), labels.flatten(),
                           ignore_index=-100, reduction="sum") / count


def masked_statistics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    valid = labels.ne(-100)
    nll = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                          ignore_index=-100, reduction="none").reshape_as(labels)
    return torch.stack((nll.sum(-1), ((logits.argmax(-1) == labels) & valid).sum(-1),
                        valid.sum(-1)), dim=-1)


def masked_metrics(prediction):
    nll, correct, count = prediction.predictions.sum(axis=0, dtype="float64")
    if count == 0:
        raise ValueError("Validation contains no masked targets")
    nll = float(nll / count)
    return {"masked_nll": nll, "masked_accuracy": float(correct / count),
            "masked_perplexity": math.exp(nll), "masked_tokens": int(count)}


def load_prepared(root):
    from datasets import concatenate_datasets, load_from_disk
    root = Path(root)
    result, manifests, tokenizer_path = {}, {}, None
    for split in ("train", "validation"):
        paths = sorted(path for path in (root / split).glob("part-*/manifest.json")
                       if not path.parent.name.endswith(".incomplete"))
        if not paths:
            raise ValueError(f"No complete {split} partitions in {root}")
        metadata = [read_json(path) for path in paths]
        total = metadata[0]["num_shards"]
        if len(paths) != total or sorted(m["shard_index"] for m in metadata) != list(range(total)):
            raise ValueError(f"Incomplete or duplicate {split} partitions")
        common = [{k: v for k, v in m.items() if k not in
                   ("shard_index", "rows", "fingerprint")} for m in metadata]
        if any(m != common[0] for m in common):
            raise ValueError(f"Inconsistent {split} preparation settings")
        parts = [load_from_disk(str(path.parent / "dataset")) for path in paths]
        if any(len(part) != meta["rows"] or part._fingerprint != meta["fingerprint"]
               for part, meta in zip(parts, metadata)):
            raise ValueError("Prepared data fingerprint or row count changed")
        result[split] = concatenate_datasets(parts)
        manifests[split] = metadata
        tokenizer_path = tokenizer_path or paths[0].parent / "tokenizer"
    train, val = manifests["train"][0], manifests["validation"][0]
    for key in ("tokenizer_hash", "max_length"):
        if train[key] != val[key]:
            raise ValueError(f"Train/validation {key} mismatch")
    return result, manifests, tokenizer_path


def load_sentence_model(checkpoint, max_length, implementation, device):
    import integrations.transformers  # registers the model in HF Auto classes
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Transformer, Pooling
    if max_length < 3:
        raise ValueError("max_length must leave room for text and BERT special tokens")
    path = Path(checkpoint)
    overrides = {"ridgon_implementation": implementation} if implementation else {}
    if (path / "modules.json").is_file():
        model = SentenceTransformer(str(path), device=device, config_kwargs=overrides)
    else:
        encoder = Transformer(str(path), config_kwargs=overrides, max_seq_length=max_length)
        model = SentenceTransformer(modules=[encoder, Pooling(
            encoder.get_embedding_dimension(), pooling_mode="mean")], device=device)
    capacity = model[0].auto_model.config.max_position_embeddings
    if max_length > capacity:
        raise ValueError(f"Requested {max_length} tokens but position capacity is {capacity}; "
                         "position extension and longer-context training are separate work")
    model.max_seq_length = max_length
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Expected FP32 model parameters")
    return model
