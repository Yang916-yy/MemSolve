"""Shared experiment plumbing; model mathematics lives in memsolve.ball."""
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


def source_digest():
    """Fingerprint the operator, native source and NLP pipeline, including local edits."""
    root = Path(__file__).resolve().parents[1]
    files = list((root / "memsolve").rglob("*.py"))
    files += [p for p in (root / "csrc").rglob("*") if p.is_file()
              and p.suffix in (".cpp", ".cu", ".cuh", ".h", ".txt")]
    files += [root / "integrations/transformers.py", root / "integrations/distributed.py", root / "pyproject.toml"]
    files += [root / "experiments" / name for name in
              ("nlp.py", "prepare_c4.py", "train_mlm.py", "train_retrieval.py", "evaluate_embeddings.py")]
    return digest({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(files)})


def versions():
    packages = {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "datasets", "accelerate",
             "sentence-transformers", "mteb")
            if importlib.util.find_spec(name.replace("-", "_")) is not None}
    return {**packages, "memsolve_source_sha256": source_digest()}


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
    if implementation not in ("cuda", "reference"):
        raise ValueError("implementation must be cuda or reference")
    if settings.get("fp16") or settings.get("bf16_full_eval") or settings.get("fp16_full_eval"):
        raise ValueError("Use bf16 AMP with FP32 parameters, not fp16/full_eval casts")
    if implementation == "cuda" and (settings.get("use_cpu") or not torch.cuda.is_available()
                                      or not settings.get("bf16")):
        raise ValueError("The cuda implementation requires a CUDA device and bf16 AMP")


def record_run(output, manifest, resume=None):
    """Validate/write on rank zero and propagate failure before Trainer starts."""
    manifest = {**manifest, "experiment": {**manifest["experiment"],
                                         "source_sha256": source_digest()}}
    output = Path(output)
    path = output / "run.json"
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    error = [None]
    if not distributed or torch.distributed.get_rank() == 0:
        try:
            if resume:
                if Path(resume).resolve().parent != output.resolve():
                    raise ValueError("Resume checkpoint must belong to this output directory")
                if not (Path(resume) / "trainer_state.json").is_file():
                    raise ValueError("Resume requires a Trainer checkpoint, not just exported weights")
                if not path.is_file() or read_json(path)["experiment"] != manifest["experiment"]:
                    raise ValueError("Resume recipe/data differ from output/run.json")
            elif output.exists() and any(output.iterdir()):
                raise ValueError(f"Output directory is not empty: {output}; use --resume")
            write_json(path, manifest)
        except (ValueError, OSError, KeyError) as exc:
            error[0] = str(exc)
    if distributed:
        torch.distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise ValueError(error[0])


class MLMCollator:
    """HF stochastic training masks; prepared validation already has fixed labels."""
    def __init__(self, tokenizer, probability):
        if not 0 < probability <= 1:
            raise ValueError("MLM mask probability must be in (0, 1]")
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


class _MLMSumLoss(torch.nn.Module):
    """Tensor-only capture surface; the original HF model still owns all weights."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.original_forward = model.forward

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            output = self.original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                           token_type_ids=token_type_ids, return_dict=True)
            return masked_loss(output, labels, labels.new_ones(()))


def prepare_mlm_graph(model, sample_batch, *, compile_model=True):
    """Capture forward/loss/backward before DDP using PyTorch graphed callables.

    HF Trainer retains accumulation, global target normalization, DDP reduction,
    optimizer and checkpoints. Only training with labels uses the fixed-shape
    graph; validation retains full logits and its ordinary model forward.
    """
    from transformers.modeling_outputs import MaskedLMOutput
    from memsolve import MemSolve
    from memsolve.ball import cuda
    keys = ("input_ids", "attention_mask", "token_type_ids", "labels")
    samples = tuple(sample_batch[key].clone() for key in keys)
    device = next(model.parameters()).device
    if device.type != "cuda" or any(t.device != device or t.dtype != torch.int64 for t in samples):
        raise ValueError("MLM graph requires CUDA int64 IDs, masks, token types and labels")
    if any(t.shape != samples[0].shape for t in samples) or samples[0].ndim != 2:
        raise ValueError("MLM graph samples must share [batch, length]")
    if model.is_gradient_checkpointing:
        raise ValueError("MLM graph does not support gradient checkpointing")
    cuda.load(device=device)
    model.train()
    loss_module = _MLMSumLoss(model)
    if compile_model:
        import torch._functorch.config as functorch_config
        import torch._dynamo.config as dynamo_config
        functorch_config.backward_pass_autocast = "off"
        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 2 * model.config.num_hidden_layers + 4)
        for module in model.modules():
            if isinstance(module, MemSolve):
                module.forward = torch.compiler.disable(module.forward)
        loss_module.forward = torch.compile(loss_module.forward, backend="inductor", dynamic=False,
            fullgraph=False, options={"triton.cudagraphs": False, "emulate_precision_casts": True})
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    try:
        # No DDP exists during capture: its hooks run outside these graphs.
        graphed = torch.cuda.make_graphed_callables(loss_module, samples, num_warmup_iters=3)
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
    original_forward = loss_module.original_forward
    # make_graphed_callables returns reusable gradient storage. AccumulateGrad
    # may otherwise adopt it when .grad is None and overwrite prior microsteps
    # on the next replay (pytorch/pytorch#181723). Keep separate accumulators.
    parameters = tuple(model.parameters())
    gradient_buffers = tuple(torch.zeros_like(p) for p in parameters)
    info = {"execution": "compile-graph" if compile_model else "graph",
            "shape": list(samples[0].shape), "captures": 1, "replays": 0,
            "ddp_in_graph": False, "optimizer_in_graph": False}

    def forward(input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                num_items_in_batch=None, **kwargs):
        if not model.training or labels is None:
            output = original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, **kwargs)
            if labels is not None:
                output["loss"] = masked_loss(output, labels, num_items_in_batch)
            return output
        if kwargs:
            raise ValueError(f"MLM training graph does not accept extra inputs: {tuple(kwargs)}")
        tensors = (input_ids, attention_mask, token_type_ids, labels)
        if any(t is None or t.shape != sample.shape or t.dtype != sample.dtype or t.device != sample.device
               for t, sample in zip(tensors, samples)):
            raise ValueError("MLM training graph input shape/dtype/device changed")
        missing = [(p, buffer) for p, buffer in zip(parameters, gradient_buffers) if p.grad is None]
        if missing:
            torch._foreach_zero_([buffer for _, buffer in missing])
            for parameter, buffer in missing:
                parameter.grad = buffer
        total = graphed(*tensors)
        count = labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
        count = torch.as_tensor(count, device=labels.device).clamp_min(1)
        info["replays"] += 1
        return MaskedLMOutput(loss=total / count)

    model.forward = forward
    model.accepts_loss_kwargs = True
    model._memsolve_graph_info = info
    return model


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


def load_prepared(root, splits=("train", "validation")):
    from datasets import concatenate_datasets, load_from_disk
    root = Path(root)
    result, manifests, tokenizer_path = {}, {}, None
    if not splits or any(split not in ("train", "validation") for split in splits):
        raise ValueError("Choose train and/or validation prepared splits")
    for split in splits:
        paths = sorted(path for path in (root / split).glob("part-*/manifest.json")
                       if not path.parent.name.endswith(".incomplete"))
        if not paths:
            raise ValueError(f"No complete {split} partitions in {root}")
        metadata = [read_json(path) for path in paths]
        if any(m.get("format") != 1 or m.get("split") != split for m in metadata):
            raise ValueError(f"Unsupported format or mislabeled {split} partition")
        total = metadata[0]["num_shards"]
        if len(paths) != total or sorted(m["shard_index"] for m in metadata) != list(range(total)):
            raise ValueError(f"Incomplete or duplicate {split} partitions")
        common = [{k: v for k, v in m.items() if k not in
                   ("shard_index", "rows", "fingerprint", "token_counts")} for m in metadata]
        if any(m != common[0] for m in common):
            raise ValueError(f"Inconsistent {split} preparation settings")
        parts = [load_from_disk(str(path.parent / "dataset")) for path in paths]
        if any(len(part) != meta["rows"] or part._fingerprint != meta["fingerprint"]
               for part, meta in zip(parts, metadata)):
            raise ValueError("Prepared data fingerprint or row count changed")
        required = {"input_ids", "attention_mask",
                    "labels" if split == "validation" else "special_tokens_mask"}
        if any(not len(part) or not required.issubset(part.column_names) for part in parts):
            raise ValueError(f"Empty {split} data or missing MLM columns")
        for part, meta in zip(parts, metadata):
            if any(getattr(part.features[key], "length", None) != meta["max_length"]
                   for key in required):
                raise ValueError(f"Prepared {split} sequence length differs from its manifest")
        result[split] = concatenate_datasets(parts)
        manifests[split] = metadata
        tokenizer_path = tokenizer_path or paths[0].parent / "tokenizer"
    if "train" in manifests and "validation" in manifests:
        train, val = manifests["train"][0], manifests["validation"][0]
        for key in ("tokenizer_hash", "max_length", "packing"):
            if train[key] != val[key]:
                raise ValueError(f"Train/validation {key} mismatch")
    return result, manifests, tokenizer_path


def _stream_tokenize(examples, tokenizer, max_length):
    # Share the exact per-document truncation/filtering with Arrow preparation.
    from experiments.prepare_c4 import tokenized_documents
    rows = list(tokenized_documents(({"text": text} for text in examples["text"]),
                tokenizer, max_length, False, 0.15, 17))
    keys = [*tokenizer.model_input_names, "special_tokens_mask"]
    return {key: [row[key] for row in rows] for key in keys}


def load_mlm_data(root, streaming_manifest=None, *, seed=17, shuffle_buffer=10000):
    """Prepared validation plus either prepared training or full local JSONL streaming.

    HF Datasets owns file/worker sharding and buffered shuffle. Accelerate must
    shard the returned HF IterableDataset exactly once across DDP ranks, with
    dispatch_batches=False and split_batches=False. Never pre-shard here.
    """
    if streaming_manifest is None:
        return load_prepared(root)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    data, manifests, tokenizer_path = load_prepared(root, splits=("validation",))
    manifest = read_json(streaming_manifest)
    if manifest.get("format") != "memsolve-local-text-stream-v1":
        raise ValueError("Unsupported local streaming manifest")
    files = manifest["files"]
    if not files or len(files) != manifest["num_source_files"]:
        raise ValueError("Incomplete local streaming source")
    paths = [str(Path(item["path"]).resolve()) for item in files]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate local streaming files")
    for item, path in zip(files, paths):
        stat = Path(path).stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Local streaming file changed: {path}")
    validation = manifests["validation"][0]
    for key in ("tokenizer_hash", "max_length", "packing"):
        if manifest[key] != validation[key]:
            raise ValueError(f"Streaming train/validation {key} mismatch")
    if manifest["packing"] != "one_document" or shuffle_buffer < 1:
        raise ValueError("Streaming requires per-document truncation and a positive shuffle buffer")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    source = load_dataset("json", data_files=paths, split="train", streaming=True).select_columns(["text"])
    source = source.shuffle(seed=seed, buffer_size=shuffle_buffer)
    source = source.map(_stream_tokenize, batched=True, batch_size=256,
                        fn_kwargs={"tokenizer": tokenizer, "max_length": manifest["max_length"]},
                        remove_columns=["text"])
    data["train"] = source
    manifests["train"] = [{**manifest, "shuffle_seed": seed, "shuffle_buffer": shuffle_buffer,
                           "tokenize_batch_size": 256, "row_count": None,
                           "resume": "HF Trainer replays/skips consumed batches; worker mask RNG is not bitwise restored"}]
    return data, manifests, tokenizer_path


def load_sentence_model(checkpoint, max_length, implementation, device):
    import integrations.transformers  # registers the model in HF Auto classes
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Transformer, Pooling
    if max_length < 3:
        raise ValueError("max_length must leave room for text and BERT special tokens")
    path = Path(checkpoint)
    from transformers import AutoConfig
    transformer_path = path
    if (path / "modules.json").is_file():
        modules = read_json(path / "modules.json")
        transformer_path = path / modules[0].get("path", "")
    config = AutoConfig.from_pretrained(str(transformer_path))
    overrides = ({"memsolve_implementation": implementation}
                 if implementation and config.model_type == "memsolve_bert" else {})
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
