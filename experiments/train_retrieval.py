"""MemSolve-BERT mean-pooled retrieval fine-tuning with Sentence Transformers cached MNR loss."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import set_seed
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator

from experiments.nlp import (checkpoint_digest, check_training_precision, load_sentence_model, read_json,
                             record_run, versions)


class AMPTripletEvaluator(TripletEvaluator):
    """ST calls evaluators outside Trainer's loss autocast context."""
    def __init__(self, *, bf16, **kwargs):
        super().__init__(**kwargs)
        self.bf16 = bf16

    def __call__(self, model, *args, **kwargs):
        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            return super().__call__(model, *args, **kwargs)


def retrieval_data(settings, local_data=None):
    if local_data:
        dataset = load_from_disk(local_data)
        source = {"path": str(Path(local_data).resolve())}
    else:
        revision = HfApi().dataset_info(settings["dataset"], revision=settings.get("revision", "main")).sha
        dataset = load_dataset(settings["dataset"], settings["subset"], split="train", revision=revision)
        source = {"dataset": settings["dataset"], "subset": settings["subset"], "revision": revision}
    columns = ["query", "positive", "negative"]
    if set(columns) - set(dataset.column_names):
        raise ValueError("Retrieval data must contain query, positive, negative text columns")
    dataset = dataset.select_columns(columns)
    size = settings["validation_triplets"]
    if not 0 < size < len(dataset):
        raise ValueError("validation_triplets must leave nonempty train and validation sets")
    split = dataset.train_test_split(test_size=size, seed=settings["split_seed"])
    training, validation = split["train"], split["test"]
    limit = settings.get("max_train_triplets")
    if limit is not None:
        training = training.select(range(min(limit, len(training))))
    # Ettin's row-wise split can share queries with training. Remove those rows
    # from the bounded training subset without materializing the full corpus.
    heldout = set(validation["query"])
    training = training.filter(lambda queries: [q not in heldout for q in queries],
                               input_columns="query", batched=True)
    if not len(training) or not len(validation):
        raise ValueError("Empty retrieval split")
    return training, validation, {**source, "train_rows": len(training), "validation_rows": len(validation),
                                 "train_fingerprint": training._fingerprint,
                                 "validation_fingerprint": validation._fingerprint}


def train(recipe, checkpoint, output, local_data=None, resume=None):
    settings = dict(recipe["training"])
    check_training_precision(settings, recipe["implementation"])
    settings["batch_sampler"] = BatchSamplers.NO_DUPLICATES
    args = SentenceTransformerTrainingArguments(output_dir=str(output), **settings)
    set_seed(args.seed)
    training, validation, source = retrieval_data(recipe["data"], local_data)
    model = load_sentence_model(resume or checkpoint, recipe["max_length"],
                                recipe["implementation"], str(args.device))
    record_run(output, {"experiment": {"kind": "retrieval", "recipe": recipe,
               "checkpoint": str(Path(checkpoint).resolve()), "data": source,
               "checkpoint_sha256": checkpoint_digest(checkpoint),
               "dataloader_drop_last": args.dataloader_drop_last,
               "world_size": args.world_size}, "versions": versions()}, resume)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=recipe["cache_mini_batch_size"],
             scale=recipe["scale"], gather_across_devices=recipe["gather_across_devices"])
    evaluator = AMPTripletEvaluator(bf16=args.bf16, anchors=validation["query"], positives=validation["positive"],
             negatives=validation["negative"], batch_size=args.per_device_eval_batch_size,
             name="heldout_queries_triplet", write_csv=False)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=training,
             eval_dataset=validation, loss=loss, evaluator=evaluator)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(Path(output) / "final"))
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/memsolve_bert_retrieval.json")
    parser.add_argument("--checkpoint", required=True, help="Exported MLM or Sentence Transformer directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", help="Optional HF save_to_disk triplet dataset")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(read_json(args.config), args.checkpoint, args.output, args.data, args.resume)


if __name__ == "__main__":
    main()
