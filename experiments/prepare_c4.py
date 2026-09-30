"""Stream C4 into restartable, independently prepared HF Arrow partitions."""
from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value, load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from experiments.nlp import digest, freeze_mask, read_json, tokenizer_digest, write_json


def tokenized_documents(source, tokenizer, max_length, validation, probability, seed):
    source = iter(source)
    while batch := list(itertools.islice(source, 256)):
        encoded = tokenizer([row["text"] for row in batch], truncation=True,
                            max_length=max_length, padding="max_length",
                            return_special_tokens_mask=True)
        for index in range(len(batch)):
            row = {key: value[index] for key, value in encoded.items()}
            if not any(a and not s for a, s in zip(row["attention_mask"], row["special_tokens_mask"])):
                continue
            yield freeze_mask(row, tokenizer, probability, seed) if validation else row


def prepare(args):
    if args.max_length < 3 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid sequence length or shard selection")
    if args.max_docs is not None and args.max_docs < 1:
        raise ValueError("max_docs must be positive (per partition)")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast or tokenizer.mask_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("A fast tokenizer with MASK and PAD tokens is required")
    if args.files:
        source_id = {"files": [{"path": str(Path(p).resolve()), "size": Path(p).stat().st_size,
                               "mtime_ns": Path(p).stat().st_mtime_ns} for p in args.files]}
        source = load_dataset("json", data_files=args.files, split="train", streaming=True)
    else:
        revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
        source_id = {"dataset": args.dataset, "config": args.dataset_config, "revision": revision}
        source = load_dataset(args.dataset, args.dataset_config, revision=revision,
                              split=args.split, streaming=True)
    if args.num_shards > source.num_shards:
        raise ValueError(f"Source exposes {source.num_shards} file shards; reduce --num-shards")
    source = source.shard(num_shards=args.num_shards, index=args.shard_index)
    settings = {"format": 1, "source": source_id, "split": args.split,
                "num_shards": args.num_shards, "shard_index": args.shard_index,
                "max_docs": args.max_docs, "max_length": args.max_length,
                "tokenizer_hash": tokenizer_digest(tokenizer), "packing": "one_document",
                "eval_mask_probability": args.eval_mask_probability, "mask_seed": args.seed}
    output = Path(args.output) / args.split / f"part-{args.shard_index:05d}-of-{args.num_shards:05d}"
    if output.exists():
        previous = read_json(output / "manifest.json")
        if {k: previous[k] for k in settings} != settings:
            raise ValueError("Existing partition uses different settings")
        print(f"Already complete: {output} ({previous['rows']} rows)")
        return output
    temporary = output.with_name(output.name + ".incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)  # Only this job's explicitly incomplete partition.
    temporary.mkdir(parents=True)

    def generate():
        rows = tokenized_documents(source, tokenizer, args.max_length,
                                   args.split == "validation", args.eval_mask_probability, args.seed)
        yield from itertools.islice(rows, args.max_docs) if args.max_docs is not None else rows

    columns = {name: Sequence(Value("int32" if name == "input_ids" else "int8"),
                             length=args.max_length) for name in tokenizer.model_input_names}
    columns["labels" if args.split == "validation" else "special_tokens_mask"] = Sequence(
        Value("int32" if args.split == "validation" else "int8"), length=args.max_length)
    dataset = Dataset.from_generator(generate, features=Features(columns),
                                     cache_dir=str(temporary / "arrow-cache"),
                                     fingerprint=digest(settings))
    dataset.save_to_disk(str(temporary / "dataset"), max_shard_size="1GB")
    stored = load_from_disk(str(temporary / "dataset"))
    tokenizer.save_pretrained(str(temporary / "tokenizer"))
    write_json(temporary / "manifest.json", {**settings, "rows": len(dataset),
                                           "fingerprint": stored._fingerprint})
    del dataset, stored
    shutil.rmtree(temporary / "arrow-cache")
    temporary.rename(output)
    print(f"Prepared: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--dataset", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--files", nargs="+", help="Local JSONL text files instead of Hub C4")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-docs", type=int, help="Prepared document limit PER partition")
    parser.add_argument("--eval-mask-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
