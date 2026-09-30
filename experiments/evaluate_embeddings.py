"""MTEB v2 / LongEmbed evaluation, plus an offline local retrieval entrypoint."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from experiments.nlp import checkpoint_digest, digest, load_sentence_model, read_json, versions, write_json


LONGEMBED = ["LEMBSummScreenFDRetrieval", "LEMBQMSumRetrieval",
             "LEMBWikimQARetrieval", "LEMBNarrativeQARetrieval"]


def select_tasks(suite, names=None):
    import mteb
    if names:
        return mteb.get_tasks(tasks=names)
    if suite == "longembed":
        return mteb.get_tasks(tasks=LONGEMBED)
    if suite == "mteb-eng-v2":
        return mteb.get_benchmark("MTEB(eng, v2)").tasks
    raise ValueError("Choose a suite or explicit MTEB task names")


def evaluate(checkpoint, output, max_length=512, implementation="cuda", device="cuda",
             suite="mteb-eng-v2", tasks=None, batch_size=32, local_ir=None, bf16=True):
    model = load_sentence_model(checkpoint, max_length, implementation, device)
    if implementation == "cuda" and (not str(device).startswith("cuda") or not bf16):
        raise ValueError("CUDA evaluation requires bf16 AMP with FP32 weights")
    selected = [] if local_ir else list(select_tasks(suite, tasks))
    metadata = {"checkpoint_sha256": checkpoint_digest(checkpoint), "max_length": max_length,
                "precision": "bf16_amp" if bf16 else "fp32", "implementation": implementation,
                "normalize_embeddings": True, "batch_size": batch_size,
                "length_policy": "truncate_to_max_length; no position extension or chunking",
                "suite": "local_ir" if local_ir else ("custom" if tasks else suite),
                "tasks": [{"name": t.metadata.name, "dataset": t.metadata.dataset,
                           "eval_splits": t.metadata.eval_splits} for t in selected],
                "versions": versions()}
    if local_ir:
        fixture = read_json(local_ir)
        metadata["local_ir_sha256"] = digest(fixture)
    destination = Path(output) / digest(metadata)[:20]
    write_json(destination / "evaluation.json", metadata)
    context = torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16) if bf16 else nullcontext()
    with context:
        if local_ir:
            from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
            evaluator = InformationRetrievalEvaluator(fixture["queries"], fixture["corpus"],
                         {k: set(v) for k, v in fixture["relevant_docs"].items()},
                         batch_size=batch_size, name="local", write_csv=False)
            scores = evaluator(model)
            write_json(destination / "scores.json", scores)
        else:
            import mteb
            # Official framework owns task loading, retrieval and scoring. Successful
            # task results are cached; restarting the same command skips those tasks.
            scores = mteb.evaluate(model, tasks=selected, cache=mteb.ResultCache(destination / "cache"),
                      encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
                      overwrite_strategy="only-missing", raise_error=True)
            scores.to_disk(destination / "results.json")
    print(f"Evaluation results: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--suite", choices=("mteb-eng-v2", "longembed"), default="mteb-eng-v2")
    selection.add_argument("--tasks", nargs="+")
    selection.add_argument("--local-ir", help="JSON with queries, corpus and relevant_docs dictionaries")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--implementation", choices=("cuda", "reference"), default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true", help="Reference CPU evaluation without AMP")
    parser.add_argument("--list-tasks", action="store_true", help="Resolve task metadata without loading data/model")
    args = parser.parse_args()
    if args.list_tasks:
        for task in select_tasks(args.suite, args.tasks):
            print(task.metadata.name, task.metadata.eval_splits)
        return
    if not args.checkpoint or not args.output:
        parser.error("--checkpoint and --output are required for evaluation")
    evaluate(args.checkpoint, args.output, args.max_length, args.implementation,
             args.device, args.suite, args.tasks, args.batch_size, args.local_ir, not args.fp32)


if __name__ == "__main__":
    main()
