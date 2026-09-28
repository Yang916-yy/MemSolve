"""GLUE fine-tuning with HF Trainer and the LION paper's M2-BASE recipe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import SequentialSampler
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, default_data_collator, set_seed

from experiments.nlp import (checkpoint_digest, prepare_nlp_compile, read_json, record_run, tokenizer_digest,
                             versions, write_json, position_capacity)
from integrations.transformers import MemSolveBertConfig, MemSolveBertForSequenceClassification

COLUMNS = {
    "cola": ("sentence", None), "sst2": ("sentence", None),
    "mnli": ("premise", "hypothesis"), "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"), "rte": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"), "stsb": ("sentence1", "sentence2"),
}


def prepare(recipe, checkpoint, root):
    """Cache pinned train/dev data; hidden-label test sets are not loaded."""
    root = Path(root)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    signature = {"dataset": recipe["dataset"], "revision": recipe["dataset_revision"],
                 "max_length": recipe["max_length"], "tokenizer_hash": tokenizer_digest(tokenizer)}
    for task in recipe["tasks"]:
        target = root / task
        if (target / "manifest.json").exists():
            if read_json(target / "manifest.json")["source"] != signature:
                raise ValueError(f"Prepared {task} data use a different source or tokenizer")
            continue
        splits = ["train", "validation_matched", "validation_mismatched"] if task == "mnli" else ["train", "validation"]
        first, second = COLUMNS[task]

        def tokenize(rows):
            return tokenizer(rows[first], rows[second] if second else None,
                             truncation=True, padding="max_length", max_length=recipe["max_length"])

        data = DatasetDict()
        label_names = None
        for split in splits:
            # Explicit Parquet paths avoid remote dataset scripts and test downloads.
            url = f"hf://datasets/{recipe['dataset']}@{recipe['dataset_revision']}/{task}/{split}-00000-of-00001.parquet"
            raw = load_dataset("parquet", data_files={split: url}, split=split)
            label_names = getattr(raw.features["label"], "names", None)
            data[split] = raw.map(tokenize, batched=True, num_proc=4,
                                  remove_columns=[key for key in raw.column_names if key != "label"])
        data.save_to_disk(str(target / "dataset"))
        write_json(target / "manifest.json", {"source": signature, "task": task,
                   "label_names": label_names, "rows": {k: len(v) for k, v in data.items()},
                   "fingerprints": {k: v._fingerprint for k, v in data.items()}})
        print(f"Prepared {task}: { {k: len(v) for k, v in data.items()} }", flush=True)


def glue_metrics(task, predictions, labels):
    if task == "stsb":
        predictions = predictions.reshape(-1)
        return {"pearson": float(pearsonr(predictions, labels).statistic),
                "spearmanr": float(spearmanr(predictions, labels).statistic)}
    predictions = predictions.argmax(axis=-1)
    if task == "cola":
        return {"matthews_correlation": float(matthews_corrcoef(labels, predictions))}
    metrics = {"accuracy": float(accuracy_score(labels, predictions))}
    if task in ("mrpc", "qqp"):
        metrics["f1"] = float(f1_score(labels, predictions, zero_division=0))
    return metrics


def torch_weight_decay(task_recipe):
    """Composer decays by wd * lr/current_peak; PyTorch decays by wd * lr."""
    if task_recipe["optimizer"] == "composer_adamw":
        return task_recipe["weight_decay"] / task_recipe["lr"]
    if task_recipe["optimizer"] == "adamw":
        return task_recipe["weight_decay"]
    raise ValueError("Unknown optimizer convention")


class GlueTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        # The public LION GLUE loader uses shuffle=False in every epoch.
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)


class GlueProgress(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.previous_time = time.monotonic()
        self.previous_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step == self.previous_step:
            return
        now = time.monotonic()
        step_seconds = (now - self.previous_time) / (state.global_step - self.previous_step)
        progress = {"utc": datetime.now(timezone.utc).isoformat(), "step": state.global_step,
                    "max_steps": state.max_steps, "epoch": state.epoch, "seconds_per_step": step_seconds,
                    "remaining_hours": (state.max_steps - state.global_step) * step_seconds / 3600,
                    "metrics": logs}
        write_json(Path(args.output_dir) / "progress.json", progress)
        print(f"PROGRESS {progress}", flush=True)
        self.previous_time, self.previous_step = now, state.global_step


def load_classifier(checkpoint, task, labels, implementation):
    config = MemSolveBertConfig.from_pretrained(checkpoint)
    config.memsolve_implementation = implementation
    config.num_labels = 1 if task == "stsb" else len(labels)
    config.problem_type = "regression" if task == "stsb" else "single_label_classification"
    config.finetuning_task = task
    if labels:
        config.id2label = dict(enumerate(labels))
        config.label2id = {name: i for i, name in enumerate(labels)}
    model, info = MemSolveBertForSequenceClassification.from_pretrained(
        checkpoint, config=config, ignore_mismatched_sizes=True, output_loading_info=True)
    allowed = ("classifier.", "bert.pooler.")
    if any(not key.startswith(allowed) for key in info["missing_keys"] + info["mismatched_keys"]):
        raise ValueError(f"Backbone loading mismatch: {info}")
    if any(not key.startswith("cls.") for key in info["unexpected_keys"]) or info["error_msgs"]:
        raise ValueError(f"Unexpected checkpoint contents: {info}")
    # Every task gets a fresh label head; an MNLI-trained pooler is retained.
    model._init_weights(model.classifier)
    return model, info


def train(args, recipe):
    if torch.cuda.device_count() != 1:
        raise ValueError("Run one GLUE job per GPU using CUDA_VISIBLE_DEVICES")
    if recipe["train_order"] != "sequential":
        raise ValueError("This LION recipe uses sequential training data")
    task_recipe = recipe["tasks"][args.task]
    if task_recipe["init"] == "mnli":
        source_config = MemSolveBertConfig.from_pretrained(args.checkpoint)
        if source_config.finetuning_task != "mnli":
            raise ValueError(f"{args.task} requires the completed MNLI checkpoint")
    set_seed(recipe["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    manifest = read_json(Path(args.data) / args.task / "manifest.json")
    if manifest["source"]["tokenizer_hash"] != tokenizer_digest(tokenizer):
        raise ValueError("Checkpoint tokenizer differs from prepared GLUE data")
    for key, expected in (("max_length", recipe["max_length"]), ("revision", recipe["dataset_revision"]),
                          ("dataset", recipe["dataset"])):
        if manifest["source"][key] != expected:
            raise ValueError(f"Prepared data differ from recipe: {key}")
    data = load_from_disk(str(Path(args.data) / args.task / "dataset"))
    validation = {key: value for key, value in data.items() if key != "train"}
    model, load_info = load_classifier(args.checkpoint, args.task, manifest["label_names"], "cuda")
    capacity = position_capacity(model.config)
    if capacity is not None and recipe["max_length"] > capacity:
        raise ValueError("Sequence length exceeds learned position capacity")
    from memsolve.ball import cuda
    cuda.load(device=torch.device("cuda", 0))
    if args.compile:
        prepare_nlp_compile(model)
    wd = torch_weight_decay(task_recipe)
    settings = dict(
        output_dir=args.output, seed=recipe["seed"], data_seed=recipe["seed"],
        per_device_train_batch_size=task_recipe["batch_size"], per_device_eval_batch_size=64,
        num_train_epochs=task_recipe["epochs"], max_steps=args.max_steps,
        learning_rate=task_recipe["lr"], weight_decay=wd, warmup_ratio=recipe["warmup_ratio"],
        lr_scheduler_type="linear", max_grad_norm=0.0, bf16=True,
        eval_strategy="steps", eval_steps=task_recipe["eval_steps"],
        save_strategy="epoch", save_total_limit=1, load_best_model_at_end=False,
        logging_steps=50, logging_first_step=True, logging_nan_inf_filter=False,
        dataloader_drop_last=False, dataloader_num_workers=args.workers,
        dataloader_persistent_workers=args.workers > 0,
        report_to=[], disable_tqdm=True, torch_compile=args.compile,
        torch_compile_backend="inductor" if args.compile else None,
        torch_compile_mode="reduce-overhead" if args.compile else None,
    )
    record_run(args.output, {"experiment": {"recipe": recipe, "task": args.task,
        "init_checkpoint": str(Path(args.checkpoint).resolve()), "init_sha256": checkpoint_digest(args.checkpoint),
        "data": manifest, "training": settings}, "versions": versions(), "loading": load_info}, args.resume)
    # LION create_param_groups(None, model) applies decay to all parameters.
    optimizer = torch.optim.AdamW(model.parameters(), lr=task_recipe["lr"], betas=tuple(recipe["betas"]),
                                  eps=recipe["epsilon"], weight_decay=wd, fused=True)
    trainer = GlueTrainer(model=model, args=TrainingArguments(**settings), processing_class=tokenizer,
        train_dataset=data["train"], eval_dataset=validation, data_collator=default_data_collator,
        compute_metrics=lambda result: glue_metrics(args.task, result.predictions, result.label_ids),
        optimizers=(optimizer, None), callbacks=[GlueProgress()])
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_state()
    trainer.save_model(str(Path(args.output) / "final"))
    metrics = trainer.evaluate()
    write_json(Path(args.output) / "results.json", {"task": args.task, "seed": recipe["seed"],
        "selection": "final", "diagnostic_max_steps": args.max_steps,
        "train": result.metrics, "validation": metrics})
    print(f"FINAL {args.task}: {metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/memsolve_bert_glue.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--task", choices=tuple(COLUMNS))
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    recipe = read_json(args.config)
    if args.prepare_only:
        prepare(recipe, args.checkpoint, args.data)
    elif args.task and args.output:
        train(args, recipe)
    else:
        parser.error("Training requires --task and --output")


if __name__ == "__main__":
    main()
