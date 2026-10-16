"""MemSolve-BERT C4 MLM training with HF Trainer, DDP, AMP, token-weighted loss and resume."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM
from experiments.nlp import (MLMCollator, accumulation_steps, check_training_precision, checkpoint_digest, digest,
    load_prepared, load_mlm_data, masked_loss, masked_metrics, masked_statistics, read_json,
    record_run, tokenizer_digest, versions, write_json, prepare_mlm_graph)


def check_data(config, manifests, tokenizer, checkpoint=None):
    first = next(iter(manifests.values()))[0]
    if tokenizer_digest(tokenizer) != first["tokenizer_hash"]:
        raise ValueError("Saved tokenizer does not match data preparation")
    if len(tokenizer) != config.vocab_size or tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError("Tokenizer vocabulary/PAD id and model configuration differ")
    if first["max_length"] > config.max_position_embeddings:
        raise ValueError("Prepared sequence length exceeds position capacity")
    if checkpoint is not None:
        saved_tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        if tokenizer_digest(saved_tokenizer) != tokenizer_digest(tokenizer):
            raise ValueError("Checkpoint tokenizer differs from prepared data")


def initialization_config(recipe, checkpoint=None):
    desired = MemSolveBertConfig(**recipe["model"])
    if checkpoint is None:
        return desired
    saved = AutoConfig.from_pretrained(str(checkpoint))
    if not isinstance(saved, MemSolveBertConfig):
        raise ValueError("Continued MLM requires a MemSolve-BERT checkpoint")
    # Never load weights under an invented contract or resize positions silently.
    for key in recipe["model"]:
        if key != "memsolve_implementation" and getattr(saved, key) != getattr(desired, key):
            raise ValueError(f"Initial checkpoint/model configuration differs: {key}")
    saved.memsolve_implementation = desired.memsolve_implementation
    return saved


def training_data(recipe, data_root, streaming_manifest=None):
    settings = recipe["training"]
    if streaming_manifest is not None:
        accelerator = settings.get("accelerator_config", {})
        if accelerator.get("dispatch_batches") is not False or accelerator.get("split_batches", False):
            raise ValueError("Streaming requires accelerator_config dispatch_batches=false, split_batches=false")
        if settings.get("max_steps", -1) <= 0 or settings.get("ignore_data_skip", False):
            raise ValueError("Streaming requires explicit max_steps and resume data skipping")
    return load_mlm_data(data_root, streaming_manifest, seed=settings.get("data_seed", settings.get("seed", 17)),
                         shuffle_buffer=recipe.get("streaming_shuffle_buffer", 10000))


def preflight(recipe, data_root, checkpoint=None, world_size=None, streaming_manifest=None):
    """Inspect real data and model geometry without allocating CUDA tensors."""
    data, manifests, tokenizer_path = training_data(recipe, data_root, streaming_manifest)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    config = initialization_config(recipe, checkpoint)
    check_data(config, manifests, tokenizer, checkpoint)
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    settings = recipe["training"]
    accum = accumulation_steps(recipe["global_batch_size"], settings["per_device_train_batch_size"], world_size)
    if not 0 < recipe["train_mask_probability"] <= 1:
        raise ValueError("train_mask_probability must be in (0, 1]")
    with torch.device("meta"):
        model = MemSolveBertForMaskedLM(config)
    steps = settings.get("max_steps", -1)
    rows = steps * recipe["global_batch_size"] if steps > 0 else None
    length = manifests["train"][0]["max_length"]
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "position_capacity": config.max_position_embeddings, "sequence_length": length,
            "world_size": world_size, "gradient_accumulation_steps": accum,
            "global_batch_size": recipe["global_batch_size"], "max_steps": steps,
            "full_batch_sequence_presentations": rows,
            "full_batch_token_slots": rows * length if rows is not None else None,
            "train_rows": len(data["train"]) if streaming_manifest is None else None,
            "train_source_files": len(manifests["train"][0]["files"]) if streaming_manifest else None,
            "validation_rows": len(data["validation"]),
            "prepared_token_counts": {split: [m.get("token_counts") for m in mm]
                                      for split, mm in manifests.items()},
            "batch_note": "Partial batches can occur at dataset boundaries; token slots include padding and special tokens",
            "core_mode": config.memsolve_core_mode, "contract": config.memsolve_contract,
            "initialization": str(Path(checkpoint).resolve()) if checkpoint else "random",
            "versions": versions()}


def train(recipe, data_root, output, resume=None, init_checkpoint=None, streaming_manifest=None):
    data, manifests, tokenizer_path = training_data(recipe, data_root, streaming_manifest)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    config = initialization_config(recipe, init_checkpoint)
    check_data(config, manifests, tokenizer, init_checkpoint)
    settings = dict(recipe["training"])
    settings["gradient_accumulation_steps"] = accumulation_steps(
        recipe["global_batch_size"], settings["per_device_train_batch_size"],
        int(os.environ.get("WORLD_SIZE", "1")))
    check_training_precision(settings, config.memsolve_implementation)
    args = TrainingArguments(output_dir=str(output), **settings)
    if streaming_manifest and data["train"].num_shards < args.world_size * max(1, args.dataloader_num_workers):
        raise ValueError("Streaming source needs at least one file per rank/worker")
    set_seed(args.seed)
    model = (MemSolveBertForMaskedLM.from_pretrained(str(init_checkpoint), config=config)
             if init_checkpoint else MemSolveBertForMaskedLM(config))
    execution = recipe.get("execution", "eager")
    if execution not in ("eager", "graph", "compile-graph"):
        raise ValueError("execution must be eager, graph or compile-graph")
    if execution != "eager" and (config.memsolve_implementation != "cuda" or args.torch_compile
                                  or args.gradient_checkpointing):
        raise ValueError("MLM graph requires CUDA without nested Trainer compile or checkpointing")
    record_run(output, {"experiment": {"kind": "mlm", "recipe": recipe,
               "data": manifests, "world_size": args.world_size,
               "dataloader_drop_last": args.dataloader_drop_last,
               "initialization": ({"path": str(Path(init_checkpoint).resolve()),
                                    "sha256": checkpoint_digest(init_checkpoint)} if init_checkpoint else "random"),
               "gradient_accumulation_steps": args.gradient_accumulation_steps},
               "versions": versions()}, resume)
    if execution != "eager":
        batch = args.per_device_train_batch_size
        rows = [data["validation"][i % len(data["validation"])] for i in range(batch)]
        sample = MLMCollator(tokenizer, recipe["train_mask_probability"])(rows)
        sample = {key: value.to(args.device) for key, value in sample.items()}
        model.to(args.device)
        prepare_mlm_graph(model, sample, compile_model=execution == "compile-graph")
        print(f"Rank {args.process_index}: MLM execution ready: {model._memsolve_graph_info}", flush=True)
    trainer = Trainer(model=model, args=args, train_dataset=data["train"],
                      eval_dataset=data["validation"], processing_class=tokenizer,
                      data_collator=MLMCollator(tokenizer, recipe["train_mask_probability"]),
                      compute_loss_func=masked_loss if execution == "eager" else None, compute_metrics=masked_metrics,
                      preprocess_logits_for_metrics=masked_statistics)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(Path(output) / "final"))
    trainer.save_state()
    if trainer.is_world_process_zero():
        write_json(Path(output) / "final" / "selection.json", {
            "selection": "best_validation" if args.load_best_model_at_end else "last",
            "global_step": trainer.state.global_step,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "metric": args.metric_for_best_model, "best_metric": trainer.state.best_metric})
    trainer.save_metrics("train", result.metrics)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    return metrics


def evaluate(recipe, data_root, checkpoint, output):
    """Score a local MLM checkpoint on the same frozen targets, including HF baselines."""
    data, manifests, tokenizer_path = load_prepared(data_root, splits=("validation",))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    config = AutoConfig.from_pretrained(str(checkpoint))
    implementation = recipe["model"]["memsolve_implementation"]
    if isinstance(config, MemSolveBertConfig):
        config.memsolve_implementation = implementation
    check_data(config, manifests, tokenizer, checkpoint)
    settings = dict(recipe["training"])
    check_training_precision(settings, implementation)
    manifest = {"checkpoint_sha256": checkpoint_digest(checkpoint), "data": manifests,
                "model_type": config.model_type, "precision": "bf16_amp" if settings.get("bf16") else "fp32",
                "per_device_eval_batch_size": settings["per_device_eval_batch_size"],
                "world_size": int(os.environ.get("WORLD_SIZE", "1")), "versions": versions()}
    destination = Path(output) / digest(manifest)[:20]
    args = TrainingArguments(output_dir=str(destination), **settings)
    model = AutoModelForMaskedLM.from_pretrained(str(checkpoint), config=config)
    trainer = Trainer(model=model, args=args, eval_dataset=data["validation"], processing_class=tokenizer,
                      data_collator=MLMCollator(tokenizer, recipe["train_mask_probability"]),
                      compute_loss_func=masked_loss, compute_metrics=masked_metrics,
                      preprocess_logits_for_metrics=masked_statistics)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    if trainer.is_world_process_zero():
        write_json(destination / "evaluation.json", manifest)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/memsolve_bert_mlm.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume", help="Explicit checkpoint-N directory in this run")
    parser.add_argument("--streaming-train-manifest", help="Verified local JSONL file manifest; --data supplies frozen validation")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint", help="Start a new training stage from local MLM weights; new optimizer/schedule")
    initialization.add_argument("--evaluate-checkpoint", help="Only evaluate a local MLM checkpoint on frozen validation masks")
    parser.add_argument("--check-only", action="store_true", help="Print data/model/budget checks without training or CUDA allocation")
    parser.add_argument("--world-size", type=int, help="Planned GPU count, only with --check-only")
    args = parser.parse_args()
    recipe = read_json(args.config)
    if args.check_only:
        if args.resume or args.evaluate_checkpoint:
            parser.error("--check-only is a training preflight, not resume/evaluation")
        print(json.dumps(preflight(recipe, args.data, args.init_checkpoint, args.world_size,
                                   args.streaming_train_manifest), indent=2))
    else:
        if not args.output or args.world_size is not None:
            parser.error("--output is required; --world-size is only for --check-only")
        if args.evaluate_checkpoint:
            if args.resume or args.streaming_train_manifest:
                parser.error("--resume applies to training, not standalone evaluation")
            evaluate(recipe, args.data, args.evaluate_checkpoint, args.output)
        else:
            train(recipe, args.data, args.output, args.resume, args.init_checkpoint,
                  args.streaming_train_manifest)


if __name__ == "__main__":
    main()
