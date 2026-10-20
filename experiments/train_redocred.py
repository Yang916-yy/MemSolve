"""Fine-tune BERT or MemSolve-BERT on Re-DocRED with the shared no-LOP head."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from experiments.nlp import (accumulation_steps, checkpoint_digest, position_capacity,
                             read_json, record_run, versions, write_json)
from experiments.redocred import (RelationConfig, RelationModel, RelationCollator,
    prepare_data, load_data, load_encoder, training_facts, metric_function,
    decode_predictions, trainer_loss)


class RelationTrainer(Trainer):
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        result = super().evaluation_loop(dataloader, description, prediction_loss_only,
                                         ignore_keys, metric_key_prefix)
        # Gathered predictions exclude DDP padding; batch-mean losses do not
        # preserve the pair weighting when document/entity counts differ.
        key = f"{metric_key_prefix}_at_loss"
        if key in result.metrics:
            result.metrics[f"{metric_key_prefix}_loss"] = result.metrics.pop(key)
        return result


def run(args, recipe):
    set_seed(recipe["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    features, documents, manifest = load_data(args.data, tokenizer)
    id2rel = {v: k for k, v in manifest["rel2id"].items()}
    if args.evaluate_only:
        config = RelationConfig.from_pretrained(args.checkpoint)
        if config.encoder_config["model_type"] == "memsolve_bert":
            config.encoder_config["memsolve_implementation"] = args.implementation
        model = RelationModel.from_pretrained(args.checkpoint, config=config)
        expected = read_json(Path(args.checkpoint) / "recipe.json")
        if recipe != expected or model.config.id2label != id2rel:
            raise ValueError("Evaluation recipe/relations differ from the exported model")
        load_info = {}
    else:
        encoder, load_info = load_encoder(args.checkpoint, args.implementation)
        # Different backbone constructors must not change the shared head's seed.
        set_seed(recipe["seed"])
        model = RelationModel(RelationConfig(encoder_config=encoder.config.to_dict(),
            block_size=recipe["block_size"], pair_chunk_size=recipe["pair_chunk_size"],
            id2label=id2rel, label2id=manifest["rel2id"]), encoder=encoder)
    capacity = position_capacity(model.encoder.config)
    if capacity is not None and recipe["window_size"] > capacity:
        raise ValueError("Window size exceeds the encoder's learned position capacity")
    use_cuda = not args.cpu
    if use_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --cpu --implementation reference for CPU checks")
    if model.encoder.config.model_type == "memsolve_bert":
        implementation = model.encoder.config.memsolve_implementation
        if implementation == "cuda" and not use_cuda:
            raise ValueError("CPU checks require a reference encoder")
    settings = dict(
        output_dir=args.output, seed=recipe["seed"], data_seed=recipe["seed"],
        per_device_train_batch_size=args.micro_batch, per_device_eval_batch_size=args.micro_batch,
        num_train_epochs=recipe["epochs"], max_steps=args.max_steps,
        learning_rate=recipe["encoder_lr"], weight_decay=recipe["weight_decay"],
        warmup_ratio=recipe["warmup_ratio"], lr_scheduler_type="linear",
        max_grad_norm=recipe["max_grad_norm"], bf16=use_cuda, use_cpu=args.cpu,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        eval_accumulation_steps=16,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=10, logging_first_step=True, logging_nan_inf_filter=False,
        dataloader_num_workers=args.workers, dataloader_persistent_workers=args.workers > 0,
        dataloader_drop_last=False, dataloader_pin_memory=use_cuda,
        remove_unused_columns=False, label_names=["labels"],
        report_to=[], disable_tqdm=True, ddp_find_unused_parameters=False,
        average_tokens_across_devices=True,
    )
    training_args = TrainingArguments(**settings)
    if training_args.n_gpu > 1:
        raise ValueError("Use torchrun or one visible GPU: DataParallel cannot split document/window batches")
    training_args.gradient_accumulation_steps = (1 if args.evaluate_only else accumulation_steps(
        recipe["global_batch_size"], args.micro_batch, training_args.world_size))
    if model.encoder.config.model_type == "memsolve_bert" and implementation == "cuda":
        from memsolve.ball import cuda
        cuda.load(device=training_args.device)
    settings["gradient_accumulation_steps"] = training_args.gradient_accumulation_steps
    if not args.evaluate_only:
        record_run(args.output, {"experiment": {"recipe": recipe, "training": settings,
            "data": manifest, "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_digest(args.checkpoint),
            "encoder": model.encoder.config.to_dict()}, "versions": versions(),
            "loading": load_info}, args.resume)
    collator = RelationCollator(tokenizer, recipe["window_size"], recipe["window_stride"], len(id2rel))
    groups = [{"params": list(model.encoder.parameters()), "lr": recipe["encoder_lr"]},
              {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")],
               "lr": recipe["head_lr"]}]
    optimizer = torch.optim.AdamW(groups, betas=tuple(recipe["betas"]), eps=recipe["epsilon"],
                                  weight_decay=recipe["weight_decay"], fused=use_cuda)
    facts = training_facts(documents["train"])
    trainer = RelationTrainer(model=model, args=training_args, processing_class=tokenizer,
        train_dataset=features["train"], eval_dataset=features["dev"], data_collator=collator,
        compute_loss_func=trainer_loss, optimizers=(optimizer, None),
        compute_metrics=metric_function(documents["dev"], facts, id2rel, recipe["max_relations"]))
    if not args.evaluate_only:
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_state()
        trainer.save_model(str(Path(args.output) / "best"))
        if trainer.is_world_process_zero():
            write_json(Path(args.output) / "best" / "recipe.json", recipe)
            write_json(Path(args.output) / "selection.json", {
                "selected_by": "dev_f1", "checkpoint": trainer.state.best_model_checkpoint,
                "dev_f1": trainer.state.best_metric, "diagnostic_max_steps": args.max_steps})
    # The test set is evaluated only after dev-based checkpoint selection.
    for split in (("dev", "test") if args.evaluate_only else ("dev",)):
        trainer.compute_metrics = metric_function(documents[split], facts, id2rel, recipe["max_relations"])
        result = trainer.predict(features[split], metric_key_prefix=split)
        if trainer.is_world_process_zero():
            logits, ids = result.predictions
            predictions = decode_predictions(logits, ids, documents[split], id2rel, recipe["max_relations"])
            write_json(Path(args.output) / f"{split}_predictions.json", predictions)
            write_json(Path(args.output) / f"{split}_metrics.json", result.metrics)
            print(f"{split}: {result.metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/memsolve_bert_redocred.json")
    parser.add_argument("--checkpoint", required=True, help="Local MLM/encoder checkpoint or exported DocRE best")
    parser.add_argument("--data", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="Fetch pinned Re-DocRED JSONs during preparation")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate an exported best model on dev/test")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--implementation", choices=("cuda", "reference"), default="cuda")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        print(prepare_data(args.data, AutoTokenizer.from_pretrained(args.checkpoint), args.download))
    elif args.output:
        run(args, read_json(args.config))
    else:
        parser.error("Training/evaluation requires --output")


if __name__ == "__main__":
    main()
