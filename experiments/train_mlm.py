"""MemSolve-BERT C4 MLM training with HF Trainer, DDP, AMP, token-weighted loss and resume."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM
from experiments.nlp import (MLMCollator, accumulation_steps, check_training_precision,
    load_prepared, masked_loss, masked_metrics, masked_statistics, read_json,
    record_run, tokenizer_digest, versions)


def train(recipe, data_root, output, resume=None):
    data, manifests, tokenizer_path = load_prepared(data_root)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tokenizer_digest(tokenizer) != manifests["train"][0]["tokenizer_hash"]:
        raise ValueError("Saved tokenizer does not match data preparation")
    config = MemSolveBertConfig(**recipe["model"])
    if len(tokenizer) != config.vocab_size or tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError("Tokenizer vocabulary/PAD id and model configuration differ")
    if manifests["train"][0]["max_length"] > config.max_position_embeddings:
        raise ValueError("Prepared sequence length exceeds position capacity")
    settings = dict(recipe["training"])
    settings["gradient_accumulation_steps"] = accumulation_steps(
        recipe["global_batch_size"], settings["per_device_train_batch_size"],
        int(os.environ.get("WORLD_SIZE", "1")))
    check_training_precision(settings, config.memsolve_implementation)
    args = TrainingArguments(output_dir=str(output), **settings)
    set_seed(args.seed)
    model = MemSolveBertForMaskedLM(config)
    record_run(output, {"experiment": {"kind": "mlm", "recipe": recipe,
               "data": manifests, "world_size": args.world_size,
               "dataloader_drop_last": args.dataloader_drop_last,
               "gradient_accumulation_steps": args.gradient_accumulation_steps},
               "versions": versions()}, resume)
    trainer = Trainer(model=model, args=args, train_dataset=data["train"],
                      eval_dataset=data["validation"], processing_class=tokenizer,
                      data_collator=MLMCollator(tokenizer, recipe["train_mask_probability"]),
                      compute_loss_func=masked_loss, compute_metrics=masked_metrics,
                      preprocess_logits_for_metrics=masked_statistics)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(Path(output) / "final"))
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/memsolve_bert_mlm.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Explicit checkpoint-N directory in this run")
    args = parser.parse_args()
    train(read_json(args.config), args.data, args.output, args.resume)


if __name__ == "__main__":
    main()
