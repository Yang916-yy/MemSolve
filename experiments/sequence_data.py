"""Data contracts for the Long Range Arena (LRA) sequence runner.

Tokenization, source metadata, deterministic validation splits, and
variable-length collation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset


LRA_TASKS = ("listops", "text", "retrieval", "image", "pathfinder", "pathx")
PATHFINDER_TASKS = ("pathfinder", "pathx")
LRA_SOURCE_REVISION = "google-research/long-range-arena@cd31e5c6"
PATHFINDER_SPLIT_PROTOCOL = "tfds-v4.0.1-hard-md5-order-v1"
PATHX_RESOLUTION = 128


@dataclass(frozen=True)
class DatasetBundle:
    """One immutable task partition and the model-facing data contract."""

    train: Dataset
    validation: Dataset
    test: Dataset
    input_kind: Literal["tokens", "values"]
    num_classes: int
    max_length: int
    metadata: dict[str, Any]
    vocab_size: int | None = None
    pad_token_id: int | None = None
    paired: bool = False
    value_masking: Literal["length", "nonzero"] = "length"


class TokenVocabulary:
    """Fixed token vocabulary with reserved padding, unknown, and EOS IDs."""

    PAD = "<pad>"
    UNK = "<unk>"
    EOS = "<eos>"

    def __init__(self, tokens: Sequence[str]) -> None:
        ordered = [self.PAD, self.UNK, self.EOS]
        ordered.extend(token for token in tokens if token not in ordered)
        self.tokens = tuple(ordered)
        self.lookup = {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_token_id(self) -> int:
        return self.lookup[self.PAD]

    @property
    def unk_token_id(self) -> int:
        return self.lookup[self.UNK]

    @property
    def eos_token_id(self) -> int:
        return self.lookup[self.EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.tokens, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def encode_tokens(self, tokens: Sequence[str], max_length: int) -> list[int]:
        _require_positive_length(max_length)
        room = max_length - 1
        encoded = [self.lookup.get(token, self.unk_token_id) for token in tokens[:room]]
        encoded.append(self.eos_token_id)
        return encoded

    def encode_bytes(self, text: str, max_length: int) -> list[int]:
        _require_positive_length(max_length)
        room = max_length - 1
        encoded = [
            self.lookup[chr(value)]
            for value in text.encode("utf-8")[:room]
        ]
        encoded.append(self.eos_token_id)
        return encoded

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.tokens, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TokenVocabulary":
        tokens = json.loads(path.read_text(encoding="utf-8"))
        if tokens[:3] != [cls.PAD, cls.UNK, cls.EOS]:
            raise ValueError(f"invalid vocabulary at {path}")
        return cls(tokens[3:])

    @classmethod
    def byte_level(cls) -> "TokenVocabulary":
        return cls([chr(value) for value in range(256)])

    @classmethod
    def listops(cls, rows: Iterable[tuple[str, int]]) -> "TokenVocabulary":
        tokens: set[str] = set()
        for expression, _label in rows:
            tokens.update(_normalise_listops(expression).split())
        return cls(sorted(tokens))


class PackedTokenDataset(Dataset):
    """Variable-length token storage backed by immutable NumPy memmaps."""

    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix
        self.offsets = np.load(prefix.with_suffix(".offsets.npy"), mmap_mode="r")
        self.labels_array = np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r")
        self.tokens = np.memmap(
            prefix.with_suffix(".tokens.bin"), dtype=np.uint16, mode="r"
        )
        self.labels = self.labels_array.astype(np.int64).tolist()
        self.lengths = np.diff(self.offsets).astype(np.int64).tolist()

    def __len__(self) -> int:
        return len(self.labels_array)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        values = np.asarray(self.tokens[start:end]).copy()
        return torch.from_numpy(values).long(), int(self.labels_array[index])


class PackedPairTokenDataset(Dataset):
    """Paired variable-length token storage for LRA Retrieval."""

    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix
        self.first_offsets = np.load(
            prefix.with_suffix(".first.offsets.npy"), mmap_mode="r"
        )
        self.second_offsets = np.load(
            prefix.with_suffix(".second.offsets.npy"), mmap_mode="r"
        )
        self.labels_array = np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r")
        self.first = np.memmap(
            prefix.with_suffix(".first.tokens.bin"), dtype=np.uint16, mode="r"
        )
        self.second = np.memmap(
            prefix.with_suffix(".second.tokens.bin"), dtype=np.uint16, mode="r"
        )
        self.labels = self.labels_array.astype(np.int64).tolist()
        self.lengths = np.maximum(
            np.diff(self.first_offsets), np.diff(self.second_offsets)
        ).astype(np.int64).tolist()

    def __len__(self) -> int:
        return len(self.labels_array)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        first_start, first_end = (
            int(self.first_offsets[index]),
            int(self.first_offsets[index + 1]),
        )
        second_start, second_end = (
            int(self.second_offsets[index]),
            int(self.second_offsets[index + 1]),
        )
        first = torch.from_numpy(np.asarray(self.first[first_start:first_end]).copy()).long()
        second = torch.from_numpy(
            np.asarray(self.second[second_start:second_end]).copy()
        ).long()
        return first, second, int(self.labels_array[index])


class IndexedDataset(Dataset):
    """A deterministic subset that preserves sequence-length metadata."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        source_lengths = getattr(dataset, "lengths", None)
        source_labels = getattr(dataset, "labels", None)
        self.lengths = (
            [int(source_lengths[index]) for index in self.indices]
            if source_lengths is not None
            else None
        )
        self.labels = (
            [int(source_labels[index]) for index in self.indices]
            if source_labels is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class PixelSequenceDataset(Dataset):
    """Grayscale arrays in raster order, including black background pixels."""

    def __init__(self, images: Path, labels: Path, *, resolution: int = 32) -> None:
        self.resolution = resolution
        self.images = np.load(images, mmap_mode="r")
        self.labels_array = np.load(labels, mmap_mode="r")
        if self.images.shape != (len(self.labels_array), resolution, resolution):
            raise ValueError("image arrays must be uint8 [samples, resolution, resolution]")
        if self.images.dtype != np.uint8:
            raise TypeError("image arrays must store uint8 grayscale pixels")
        self.labels = self.labels_array.astype(np.int64).tolist()
        self.lengths = [resolution**2] * len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        pixels = np.asarray(self.images[index], dtype=np.float32) / 255.0
        return torch.from_numpy(pixels.reshape(self.resolution**2, 1)), self.labels[index]


class PathfinderDataset(Dataset):
    """LRA Pathfinder images as one scalar token per pixel."""

    def __init__(self, data_dir: Path) -> None:
        try:
            from PIL import Image  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Pillow is required for LRA Pathfinder") from error
        self.data_dir = data_dir
        self.resolution = int(data_dir.name.removeprefix("pathfinder"))
        if self.resolution <= 0:
            raise ValueError(f"cannot infer Pathfinder resolution from {data_dir}")
        difficulty = data_dir / "curv_contour_length_14"
        metadata_paths = sorted(
            (difficulty / "metadata").glob("*.npy"), key=lambda path: int(path.stem)
        )
        if not metadata_paths:
            raise FileNotFoundError(f"no Pathfinder metadata found under {difficulty}")
        self.samples: list[tuple[Path, int]] = []
        self.sample_keys: list[str] = []
        for metadata_path in metadata_paths:
            for example_id, line in enumerate(
                metadata_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                columns = line.split()
                if len(columns) < 4:
                    raise ValueError(f"invalid Pathfinder metadata line in {metadata_path}")
                relative = Path("curv_contour_length_14") / columns[0] / columns[1]
                self.samples.append((relative, int(columns[3])))
                self.sample_keys.append("_".join((columns[0], columns[1], str(example_id))))
        self.labels = [label for _path, label in self.samples]
        self.lengths = [self.resolution**2] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(self.data_dir / path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(pixels.reshape(-1, 1)), label


class LengthBucketBatchSampler(BatchSampler):
    """Length-aware train batches without a deterministic token order."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        seed: int,
        *,
        bucket_multiplier: int = 20,
    ) -> None:
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_size = max(self.batch_size, self.batch_size * bucket_multiplier)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])


def _pad_tokens_to_length(tokens: torch.Tensor, length: int | None, pad_id: int) -> torch.Tensor:
    if length is None:
        return tokens
    if tokens.shape[1] > length:
        raise ValueError("padding length must cover every sequence in the batch")
    return torch.nn.functional.pad(tokens, (0, length - tokens.shape[1]), value=pad_id)


def collate_tokens(
    rows: Sequence[tuple[torch.Tensor, int]], *, pad_token_id: int, pad_length: int | None = None
) -> dict[str, torch.Tensor | float]:
    sequences, labels = zip(*rows)
    lengths = torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long)
    token_ids = pad_sequence(
        sequences, batch_first=True, padding_value=pad_token_id
    )
    token_ids = _pad_tokens_to_length(token_ids, pad_length, pad_token_id)
    positions = torch.arange(token_ids.shape[1]).unsqueeze(0)
    return {
        "inputs": token_ids,
        "mask": positions < lengths.unsqueeze(1),
        "labels": torch.tensor(labels, dtype=torch.long),
        "padding_ratio": 1.0 - float(lengths.sum()) / max(token_ids.numel(), 1),
    }


def collate_token_pairs(
    rows: Sequence[tuple[torch.Tensor, torch.Tensor, int]], *, pad_token_id: int,
    pad_length: int | None = None
) -> dict[str, torch.Tensor | float]:
    first, second, labels = zip(*rows)
    first_lengths = torch.tensor([tokens.numel() for tokens in first], dtype=torch.long)
    second_lengths = torch.tensor([tokens.numel() for tokens in second], dtype=torch.long)
    first_ids = pad_sequence(first, batch_first=True, padding_value=pad_token_id)
    second_ids = pad_sequence(second, batch_first=True, padding_value=pad_token_id)
    first_ids = _pad_tokens_to_length(first_ids, pad_length, pad_token_id)
    second_ids = _pad_tokens_to_length(second_ids, pad_length, pad_token_id)
    first_positions = torch.arange(first_ids.shape[1]).unsqueeze(0)
    second_positions = torch.arange(second_ids.shape[1]).unsqueeze(0)
    return {
        "first": first_ids,
        "first_mask": first_positions < first_lengths.unsqueeze(1),
        "second": second_ids,
        "second_mask": second_positions < second_lengths.unsqueeze(1),
        "labels": torch.tensor(labels, dtype=torch.long),
        "padding_ratio": 1.0
        - float(first_lengths.sum() + second_lengths.sum())
        / max(first_ids.numel() + second_ids.numel(), 1),
    }


def collate_values(
    rows: Sequence[tuple[torch.Tensor, int]],
    *,
    value_masking: Literal["length", "nonzero"] = "length",
) -> dict[str, torch.Tensor | float]:
    if value_masking not in ("length", "nonzero"):
        raise ValueError(f"unsupported value masking contract: {value_masking}")
    values, labels = zip(*rows)
    lengths = torch.tensor([value.shape[0] for value in values], dtype=torch.long)
    inputs = pad_sequence(values, batch_first=True)
    positions = torch.arange(inputs.shape[1]).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    if value_masking == "nonzero":
        mask = mask & inputs.ne(0).any(dim=-1)
    return {
        "inputs": inputs,
        "mask": mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "padding_ratio": 1.0 - float(mask.sum()) / max(mask.numel(), 1),
    }


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    collate_fn: Callable,
    train: bool,
    seed: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "collate_fn": collate_fn,
    }
    lengths = getattr(dataset, "lengths", None)
    if train and lengths is not None:
        return DataLoader(
            dataset,
            batch_sampler=LengthBucketBatchSampler(lengths, batch_size, seed),
            **kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        generator=torch.Generator().manual_seed(seed) if train else None,
        **kwargs,
    )


def prepare_lra(
    task: str,
    *,
    data_root: Path,
    cache_root: Path,
    max_length: int | None,
    validation_fraction: float | None,
    split_seed: int | None,
    pathfinder_resolution: int | None,
    allow_download: bool,
    revision: str | None,
    formal: bool = False,
) -> DatasetBundle:
    """Prepare the public LRA tasks with pinned preprocessing semantics."""

    if task not in LRA_TASKS:
        raise ValueError(f"unsupported LRA task: {task}")
    if task == "image":
        if max_length not in (None, 1024) or pathfinder_resolution is not None:
            raise ValueError("LRA Image uses the complete 32x32 pixel grid")
        if validation_fraction is None or split_seed is None:
            raise ValueError("LRA Image requires validation_fraction and split_seed")
        return _prepare_lra_image(data_root, validation_fraction, split_seed, formal)
    if task in PATHFINDER_TASKS:
        task_name = "Path-X" if task == "pathx" else "Pathfinder"
        if max_length is not None:
            raise ValueError(f"{task_name} derives max_length from its resolution")
        if validation_fraction is not None or split_seed is not None:
            raise ValueError(f"{task_name} uses the official hard 80/10/10 split")
        if task == "pathx":
            if pathfinder_resolution is not None:
                raise ValueError(
                    "Path-X fixes its resolution at 128; "
                    "use Pathfinder for a configurable resolution"
                )
            resolution = PATHX_RESOLUTION
        else:
            if pathfinder_resolution is None or pathfinder_resolution <= 0:
                raise ValueError("Pathfinder requires a positive resolution")
            resolution = pathfinder_resolution
        return _prepare_lra_pathfinder(
            data_root=data_root,
            task=task,
            resolution=resolution,
            formal=formal,
        )
    if pathfinder_resolution is not None:
        raise ValueError("pathfinder_resolution is only valid for generic Pathfinder")
    if max_length is None:
        raise ValueError(f"LRA {task} requires max_length")
    _require_positive_length(max_length)
    if task == "text":
        if validation_fraction is None or split_seed is None:
            raise ValueError("LRA Text requires validation_fraction and split_seed")
    elif validation_fraction is not None or split_seed is not None:
        raise ValueError(f"LRA {task} uses its official validation split")
    if task == "listops":
        files = _resolve_listops_files(data_root)
        source = {
            split: source_signature(path, include_content_hash=formal)
            for split, path in files.items()
        }
        vocabulary = _load_or_build_listops_vocabulary(
            cache_root,
            files["train"],
            include_content_hash=formal,
        )
        source_provenance: dict[str, Any] = {
            "splits": source,
            "pinned_manifest": _optional_source_manifest(data_root),
        }
        if formal:
            source_provenance["content_sha256"] = _named_file_content_sha256(files)
        datasets = {
            split: build_packed_tokens(
                cache_root / "lra" / "listops" / f"{split}-l{max_length}",
                _iter_listops(path),
                lambda text: vocabulary.encode_tokens(
                    _normalise_listops(text).split(), max_length
                ),
                {
                    "schema": 1,
                    "suite": "lra",
                    "task": task,
                    "split": split,
                    "max_length": max_length,
                    "vocabulary": vocabulary.fingerprint,
                    "source": source[split],
                },
            )
            for split, path in files.items()
        }
        return DatasetBundle(
            train=datasets["train"],
            validation=datasets["val"],
            test=datasets["test"],
            input_kind="tokens",
            num_classes=10,
            max_length=max_length,
            vocab_size=vocabulary.vocab_size,
            pad_token_id=vocabulary.pad_token_id,
        metadata={
            "suite": "lra",
            "task": task,
            "source_definition": LRA_SOURCE_REVISION,
            "source": source_provenance,
                "tokenization": "official-listops-token-stream-with-eos",
                "vocabulary_fingerprint": vocabulary.fingerprint,
                "split_protocol": "official-files-v1",
            },
        )
    if task == "text":
        return _prepare_lra_text(
            cache_root=cache_root,
            max_length=max_length,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
            allow_download=allow_download,
            revision=revision,
            formal=formal,
        )
    if task == "retrieval":
        return _prepare_lra_retrieval(
            data_root=data_root,
            cache_root=cache_root,
            max_length=max_length,
            formal=formal,
        )
    raise AssertionError(f"unhandled LRA task: {task}")


def stratified_split_indices(
    labels: Sequence[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministically reserve validation samples without touching test data."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    rng = random.Random(seed)
    train, validation = [], []
    for indices in groups.values():
        rng.shuffle(indices)
        if len(indices) <= 1:
            train.extend(indices)
            continue
        count = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    if not validation:
        raise ValueError("cannot form a validation split from singleton classes")
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def stratified_subset_indices(labels: Sequence[int], count: int, seed: int) -> list[int]:
    """Choose a deterministic, label-balanced pilot subset."""

    if count <= 0 or count >= len(labels):
        return list(range(len(labels)))
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    quotas = {label: count * len(indices) / len(labels) for label, indices in groups.items()}
    minimum = 1 if count >= len(groups) else 0
    allocation = {
        label: min(len(indices), max(minimum, int(quotas[label])))
        for label, indices in groups.items()
    }
    while sum(allocation.values()) < count:
        choices = [label for label, indices in groups.items() if allocation[label] < len(indices)]
        label = max(choices, key=lambda item: (quotas[item] - allocation[item], -item))
        allocation[label] += 1
    while sum(allocation.values()) > count:
        choices = [label for label in groups if allocation[label] > minimum]
        label = min(choices, key=lambda item: (quotas[item] - allocation[item], item))
        allocation[label] -= 1
    selected = [
        index
        for label, indices in groups.items()
        for index in indices[: allocation[label]]
    ]
    rng.shuffle(selected)
    return selected


def limit_dataset(dataset: Dataset, count: int, seed: int) -> Dataset:
    if count <= 0 or count >= len(dataset):
        return dataset
    labels = getattr(dataset, "labels", None)
    if labels is None:
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        return IndexedDataset(dataset, indices[:count])
    return IndexedDataset(dataset, stratified_subset_indices(labels, count, seed))


def index_fingerprint(dataset: Dataset) -> str | None:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return None
    digest = hashlib.sha256()
    for index in indices:
        digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def pathfinder_split_indices(sample_keys: Sequence[str]) -> tuple[list[int], list[int], list[int]]:
    """Reproduce TFDS v4.0.1's ``hard`` hash order before 80/10/10 slicing."""

    if len(sample_keys) < 10:
        raise ValueError("Pathfinder needs at least ten examples for an 80/10/10 split")

    def key(index: int) -> int:
        # BeamWriter uses Hasher("hard").hash_key: integer MD5 ordering is the
        # record order underlying LRA's ``hard[:80%]`` split expression.
        try:
            digest = hashlib.md5(b"hard", usedforsecurity=False)
        except TypeError:  # pragma: no cover - old Python builds
            digest = hashlib.md5(b"hard")
        digest.update(sample_keys[index].encode("utf-8"))
        return int.from_bytes(digest.digest(), byteorder="big")

    order = sorted(range(len(sample_keys)), key=key)
    train_end = int(round(0.8 * len(order)))
    validation_end = int(round(0.9 * len(order)))
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def _file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _named_file_content_sha256(files: dict[str, Path]) -> str:
    """Hash a labelled collection of raw source files without path dependence."""

    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_content_sha256(path)))
    return digest.hexdigest()


def _directory_content_sha256(directory: Path) -> str:
    """Hash every source file relative to its dataset root."""

    digest = hashlib.sha256()
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_content_sha256(path)))
    return digest.hexdigest()


def source_signature(
    path: Path,
    *,
    include_content_hash: bool = False,
) -> dict[str, int | str]:
    stat = path.stat()
    signature: dict[str, int | str] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_content_hash:
        signature["content_sha256"] = _file_content_sha256(path)
    return signature


def is_immutable_revision(revision: str | None) -> bool:
    """Return whether a Hugging Face revision is a full content-addressed SHA."""

    return isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is not None


def validate_formal_source_provenance(bundle: DatasetBundle) -> None:
    """Require replayable source identity before a result is marked formal."""

    source = bundle.metadata.get("source")
    if not isinstance(source, dict):
        raise ValueError("--formal requires structured source provenance metadata")
    content_hash = source.get("content_sha256")
    if isinstance(content_hash, str) and re.fullmatch(r"[0-9a-f]{64}", content_hash):
        return
    if source.get("transport") == "huggingface" and is_immutable_revision(
        source.get("revision")
    ):
        return
    raise ValueError(
        "--formal requires local source content_sha256 metadata or an immutable "
        "download revision"
    )


@contextlib.contextmanager
def _cache_lock(prefix: Path) -> Iterator[None]:
    """Serialize construction of one immutable packed-data cache entry."""

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - sequence runs use Linux CUDA
        raise RuntimeError("packed sequence caches require POSIX file locking") from error
    lock_path = prefix.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_packed_tokens(
    prefix: Path,
    rows: Iterable[tuple[str, int]],
    encode: Callable[[str], Sequence[int]],
    manifest: dict[str, Any],
) -> PackedTokenDataset:
    required = (
        prefix.with_suffix(".tokens.bin"),
        prefix.with_suffix(".offsets.npy"),
        prefix.with_suffix(".labels.npy"),
    )
    manifest_path = prefix.with_suffix(".manifest.json")
    with _cache_lock(prefix):
        if _cache_matches(required, manifest_path, manifest):
            return PackedTokenDataset(prefix)
        _clear_cache_files((*required, manifest_path))
        temporary = prefix.with_suffix(".tokens.bin.tmp")
        offsets, labels = [0], []
        with temporary.open("wb") as stream:
            for text, label in rows:
                values = np.asarray(encode(text), dtype=np.uint16)
                if not values.size:
                    raise ValueError("tokenization produced an empty sequence")
                values.tofile(stream)
                offsets.append(offsets[-1] + len(values))
                labels.append(int(label))
        temporary.replace(required[0])
        np.save(required[1], np.asarray(offsets, dtype=np.int64))
        np.save(required[2], np.asarray(labels, dtype=np.int64))
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return PackedTokenDataset(prefix)


def build_packed_pairs(
    prefix: Path,
    rows: Iterable[tuple[str, str, int]],
    encode: Callable[[str], Sequence[int]],
    manifest: dict[str, Any],
) -> PackedPairTokenDataset:
    required = (
        prefix.with_suffix(".first.tokens.bin"),
        prefix.with_suffix(".second.tokens.bin"),
        prefix.with_suffix(".first.offsets.npy"),
        prefix.with_suffix(".second.offsets.npy"),
        prefix.with_suffix(".labels.npy"),
    )
    manifest_path = prefix.with_suffix(".manifest.json")
    with _cache_lock(prefix):
        if _cache_matches(required, manifest_path, manifest):
            return PackedPairTokenDataset(prefix)
        _clear_cache_files((*required, manifest_path))
        first_temporary = prefix.with_suffix(".first.tokens.bin.tmp")
        second_temporary = prefix.with_suffix(".second.tokens.bin.tmp")
        first_offsets, second_offsets, labels = [0], [0], []
        with first_temporary.open("wb") as first_stream, second_temporary.open(
            "wb"
        ) as second_stream:
            for first, second, label in rows:
                first_values = np.asarray(encode(first), dtype=np.uint16)
                second_values = np.asarray(encode(second), dtype=np.uint16)
                if not first_values.size or not second_values.size:
                    raise ValueError("tokenization produced an empty Retrieval document")
                first_values.tofile(first_stream)
                second_values.tofile(second_stream)
                first_offsets.append(first_offsets[-1] + len(first_values))
                second_offsets.append(second_offsets[-1] + len(second_values))
                labels.append(int(label))
        first_temporary.replace(required[0])
        second_temporary.replace(required[1])
        np.save(required[2], np.asarray(first_offsets, dtype=np.int64))
        np.save(required[3], np.asarray(second_offsets, dtype=np.int64))
        np.save(required[4], np.asarray(labels, dtype=np.int64))
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return PackedPairTokenDataset(prefix)


def _prepare_lra_image(data_root: Path, validation_fraction: float,
                       split_seed: int, formal: bool) -> DatasetBundle:
    root = data_root / "cifar10"
    paths = {name: root / f"{name}.npy" for name in
             ("train_images", "train_labels", "test_images", "test_labels")}
    train_full = PixelSequenceDataset(paths["train_images"], paths["train_labels"])
    test = PixelSequenceDataset(paths["test_images"], paths["test_labels"])
    train_ids, validation_ids = stratified_split_indices(
        train_full.labels, validation_fraction, split_seed)
    train, validation = IndexedDataset(train_full, train_ids), IndexedDataset(train_full, validation_ids)
    source = {"transport": "local-cifar10-grayscale-arrays",
              "files": {name: source_signature(path, include_content_hash=formal)
                        for name, path in paths.items()},
              "preparation": _optional_source_manifest(root)}
    if formal:
        source["content_sha256"] = _named_file_content_sha256(paths)
    return DatasetBundle(train=train, validation=validation, test=test,
        input_kind="values", num_classes=10, max_length=1024,
        metadata={"suite": "lra", "task": "image", "source": source,
                  "source_definition": LRA_SOURCE_REVISION, "resolution": 32,
                  "tokenization": "grayscale-float-0-1-raster-all-pixels",
                  "split_protocol": "stratified-validation-v1", "split_seed": split_seed,
                  "validation_fraction": validation_fraction,
                  "split_fingerprints": {"train": index_fingerprint(train),
                                         "validation": index_fingerprint(validation)}})


def _prepare_lra_text(
    *,
    cache_root: Path,
    max_length: int,
    validation_fraction: float,
    split_seed: int,
    allow_download: bool,
    revision: str | None,
    formal: bool,
) -> DatasetBundle:
    rows, source = _load_imdb_rows(
        cache_root=cache_root,
        allow_download=allow_download,
        revision=revision,
        formal=formal,
    )
    vocabulary = TokenVocabulary.byte_level()
    train_full = build_packed_tokens(
        cache_root / "lra" / "text" / f"train-l{max_length}",
        ((str(row["text"]), int(row["label"])) for row in rows["train"]),
        lambda text: vocabulary.encode_bytes(text, max_length),
        {
            "schema": 1,
            "suite": "lra",
            "task": "text",
            "split": "train",
            "max_length": max_length,
            "vocabulary": vocabulary.fingerprint,
            "source": source,
        },
    )
    test = build_packed_tokens(
        cache_root / "lra" / "text" / f"test-l{max_length}",
        ((str(row["text"]), int(row["label"])) for row in rows["test"]),
        lambda text: vocabulary.encode_bytes(text, max_length),
        {
            "schema": 1,
            "suite": "lra",
            "task": "text",
            "split": "test",
            "max_length": max_length,
            "vocabulary": vocabulary.fingerprint,
            "source": source,
        },
    )
    train_indices, validation_indices = stratified_split_indices(
        train_full.labels, validation_fraction, split_seed
    )
    train = IndexedDataset(train_full, train_indices)
    validation = IndexedDataset(train_full, validation_indices)
    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        input_kind="tokens",
        num_classes=2,
        max_length=max_length,
        vocab_size=vocabulary.vocab_size,
        pad_token_id=vocabulary.pad_token_id,
        metadata={
            "suite": "lra",
            "task": "text",
            "source_definition": LRA_SOURCE_REVISION,
            "source": source,
            "tokenization": "utf-8-bytes-with-eos",
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "split_protocol": "stratified-validation-v1",
            "split_seed": split_seed,
            "validation_fraction": validation_fraction,
            "split_fingerprints": {
                "train": index_fingerprint(train),
                "validation": index_fingerprint(validation),
            },
        },
    )


def _load_imdb_rows(
    *, cache_root: Path, allow_download: bool, revision: str | None, formal: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_base = cache_root / "hf" / "stanfordnlp___imdb" / "plain_text" / "0.0.0"
    candidates = []
    if revision:
        candidates.append(cache_base / revision)
    elif cache_base.is_dir():
        candidates.extend(sorted((path for path in cache_base.iterdir() if path.is_dir()), reverse=True))
    for candidate in candidates:
        train_files = tuple(candidate.glob("imdb-train.arrow"))
        test_files = tuple(candidate.glob("imdb-test.arrow"))
        if len(train_files) != 1 or len(test_files) != 1:
            continue
        try:
            from datasets import Dataset as ArrowDataset
        except ImportError as error:
            raise RuntimeError("install the sequence extra to read cached LRA Text") from error
        train_rows = ArrowDataset.from_file(str(train_files[0]))
        test_rows = ArrowDataset.from_file(str(test_files[0]))
        return (
            {"train": train_rows, "test": test_rows},
            {
                "repository": "stanfordnlp/imdb",
                "revision": candidate.name,
                "transport": "local-hf-arrow",
                "train_fingerprint": train_rows._fingerprint,
                "test_fingerprint": test_rows._fingerprint,
                **(
                    {
                        "content_sha256": _named_file_content_sha256(
                            {"train": train_files[0], "test": test_files[0]}
                        )
                    }
                    if formal
                    else {}
                ),
            },
        )
    if not allow_download:
        raise FileNotFoundError(
            f"no cached LRA Text data under {cache_root}; pass --allow-download"
        )
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the sequence extra to prepare LRA Text") from error
    rows = load_dataset(
        "stanfordnlp/imdb",
        revision=revision or "main",
        cache_dir=str(cache_root / "hf"),
    )
    source = {
        "repository": "stanfordnlp/imdb",
        "revision": revision or "main",
        "transport": "huggingface",
        "train_fingerprint": rows["train"]._fingerprint,
        "test_fingerprint": rows["test"]._fingerprint,
    }
    return {"train": rows["train"], "test": rows["test"]}, source


def _prepare_lra_retrieval(
    *, data_root: Path, cache_root: Path, max_length: int, formal: bool
) -> DatasetBundle:
    source = _find_directory(
        data_root, ("aan", "tsv_data", "opennlplab/data/aan")
    )
    files = {
        "train": source / "new_aan_pairs.train.tsv",
        "val": source / "new_aan_pairs.eval.tsv",
        "test": source / "new_aan_pairs.test.tsv",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    vocabulary = TokenVocabulary.byte_level()
    source_metadata = {
        split: source_signature(path, include_content_hash=formal)
        for split, path in files.items()
    }
    source_provenance: dict[str, Any] = {
        "splits": source_metadata,
        "pinned_manifest": _optional_source_manifest(data_root),
        "source_manifest": _optional_source_manifest(source),
    }
    if formal:
        source_provenance["content_sha256"] = _named_file_content_sha256(files)
    datasets = {
        split: build_packed_pairs(
            cache_root / "lra" / "retrieval" / f"{split}-l{max_length}",
            _iter_aan(path),
            lambda text: vocabulary.encode_bytes(text, max_length),
            {
                "schema": 1,
                "suite": "lra",
                "task": "retrieval",
                "split": split,
                "max_length": max_length,
                "vocabulary": vocabulary.fingerprint,
                "source": source_metadata[split],
            },
        )
        for split, path in files.items()
    }
    return DatasetBundle(
        train=datasets["train"],
        validation=datasets["val"],
        test=datasets["test"],
        input_kind="tokens",
        num_classes=2,
        max_length=max_length,
        vocab_size=vocabulary.vocab_size,
        pad_token_id=vocabulary.pad_token_id,
        paired=True,
        metadata={
            "suite": "lra",
            "task": "retrieval",
            "source_definition": LRA_SOURCE_REVISION,
            "source": source_provenance,
            "tokenization": "utf-8-bytes-with-eos",
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "split_protocol": "official-files-v1",
        },
    )


def _prepare_pathfinder_arrays(
    source: Path, task: str, resolution: int, formal: bool
) -> DatasetBundle:
    manifest = _optional_source_manifest(source)
    if manifest is None or manifest.get("difficulty") != "hard" or manifest.get("resolution") != resolution:
        raise ValueError("Pathfinder arrays require a source manifest identifying hard difficulty and resolution")
    files = {f"{split}_{kind}": source / f"{split}_{kind}.npy"
             for split in ("train", "validation", "test") for kind in ("images", "labels")}
    datasets = {split: PixelSequenceDataset(files[f"{split}_images"], files[f"{split}_labels"],
                                            resolution=resolution)
                for split in ("train", "validation", "test")}
    provenance = {"transport": "local-grayscale-arrays", "preparation": manifest,
                  "files": {name: source_signature(path, include_content_hash=formal)
                            for name, path in files.items()}}
    if formal:
        provenance["content_sha256"] = _named_file_content_sha256(files)
    return DatasetBundle(
        train=datasets["train"], validation=datasets["validation"], test=datasets["test"],
        input_kind="values", num_classes=2, max_length=resolution**2,
        metadata={"suite": "lra", "task": task, "source_definition": LRA_SOURCE_REVISION,
                  "source": provenance, "resolution": resolution, "value_masking": "all-pixels",
                  "split_protocol": manifest.get("split_protocol", "provided-splits")})


def _prepare_lra_pathfinder(
    *, data_root: Path, task: str, resolution: int, formal: bool
) -> DatasetBundle:
    source = _resolve_pathfinder_directory(data_root, resolution)
    if (source / "train_images.npy").is_file():
        return _prepare_pathfinder_arrays(source, task, resolution, formal)
    full = PathfinderDataset(source)
    source_signature = _pathfinder_source_signature(
        source,
        include_content_hash=formal,
    )
    source_manifest = _optional_source_manifest(data_root)
    _verify_pathfinder_manifest(
        source_signature,
        len(full),
        source_manifest,
        resolution=resolution,
    )
    train_indices, validation_indices, test_indices = pathfinder_split_indices(
        full.sample_keys
    )
    train = IndexedDataset(full, train_indices)
    validation = IndexedDataset(full, validation_indices)
    test = IndexedDataset(full, test_indices)
    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        input_kind="values",
        num_classes=2,
        max_length=resolution**2,
        value_masking="length",
        metadata={
            "suite": "lra",
            "task": task,
            "source_definition": LRA_SOURCE_REVISION,
            "source": {
                **source_signature,
                "pinned_manifest": source_manifest,
            },
            "split_protocol": PATHFINDER_SPLIT_PROTOCOL,
            "split_fingerprints": {
                "train": index_fingerprint(train),
                "validation": index_fingerprint(validation),
                "test": index_fingerprint(test),
            },
            "resolution": resolution,
            "value_masking": "all-pixels",
        },
    )


def _normalise_listops(text: str) -> str:
    return text.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None})


def _iter_listops(path: Path) -> Iterator[tuple[str, int]]:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        try:
            source_index, target_index = header.index("Source"), header.index("Target")
        except ValueError as error:
            raise ValueError(f"invalid ListOps header in {path}: {header}") from error
        for line in stream:
            columns = line.rstrip("\n").split("\t")
            if len(columns) > max(source_index, target_index):
                yield columns[source_index], int(columns[target_index])


def _iter_aan(path: Path) -> Iterator[tuple[str, str, int]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            columns = line.rstrip("\n").split("\t", 4)
            if len(columns) != 5:
                continue
            label, _id1, _id2, first, second = columns
            try:
                numeric = float(label)
            except ValueError:
                continue
            if numeric.is_integer() and int(numeric) in (0, 1):
                yield first, second, int(numeric)


def _resolve_listops_files(root: Path) -> dict[str, Path]:
    for relative in (
        "listops",
        "listops-1000",
        "listops/listops",
        "_kaggle/listops/listops",
    ):
        source = root / relative
        files = {split: source / f"{split}.tsv" for split in ("train", "val", "test")}
        if all(path.is_file() for path in files.values()):
            return files
        basic = {
            split: source / f"basic_{split}.tsv" for split in ("train", "val", "test")
        }
        if all(path.is_file() for path in basic.values()):
            return basic
    raise FileNotFoundError(f"no complete ListOps split found under {root}")


def _resolve_pathfinder_directory(root: Path, resolution: int) -> Path:
    for relative in (
        f"pathfinder/pathfinder{resolution}",
        f"pathfinder{resolution}",
        f"pathfinder/pathfinder/pathfinder{resolution}",
        f"_kaggle/pathfinder/pathfinder/pathfinder{resolution}",
    ):
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no Pathfinder-{resolution} directory found under {root}")


def _find_directory(root: Path, alternatives: Sequence[str]) -> Path:
    for relative in alternatives:
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"none of {[str(root / value) for value in alternatives]} exists")


def _load_or_build_listops_vocabulary(
    cache_root: Path,
    train_file: Path,
    *,
    include_content_hash: bool,
) -> TokenVocabulary:
    path = cache_root / "lra" / "listops" / "vocab.json"
    manifest_path = path.with_suffix(".manifest.json")
    manifest = {
        "schema": 1,
        "source": source_signature(
            train_file,
            include_content_hash=include_content_hash,
        ),
    }
    with _cache_lock(path):
        if path.is_file() and manifest_path.is_file():
            if json.loads(manifest_path.read_text(encoding="utf-8")) == manifest:
                return TokenVocabulary.load(path)
        vocabulary = TokenVocabulary.listops(_iter_listops(train_file))
        vocabulary.save(path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return vocabulary


def _optional_source_manifest(data_root: Path) -> dict[str, Any] | None:
    path = data_root / "source-manifest.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid LRA source manifest at {path}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"LRA source manifest at {path} must be an object")
    return loaded


def _pathfinder_source_signature(
    source: Path,
    *,
    include_content_hash: bool,
) -> dict[str, Any]:
    metadata_dir = source / "curv_contour_length_14" / "metadata"
    paths = sorted(metadata_dir.glob("*.npy"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    signature = {
        "path": str(source.resolve()),
        "metadata_files": len(paths),
        "metadata_sha256": digest.hexdigest(),
    }
    if include_content_hash:
        signature["content_sha256"] = _directory_content_sha256(source)
    return signature


def _verify_pathfinder_manifest(
    signature: dict[str, Any],
    examples: int,
    manifest: dict[str, Any] | None,
    *,
    resolution: int,
) -> None:
    if manifest is None:
        return
    expected = manifest.get(f"pathfinder{resolution}_hard")
    if not isinstance(expected, dict):
        return
    for key, actual in (
        ("metadata_files", signature["metadata_files"]),
        ("metadata_sha256", signature["metadata_sha256"]),
        ("usable_rows", examples),
    ):
        if key in expected and expected[key] != actual:
            raise ValueError(
                "Pathfinder source does not match its pinned manifest: "
                f"{key}={actual!r}, expected {expected[key]!r}"
            )


def _cache_matches(
    required: Sequence[Path], manifest_path: Path, manifest: dict[str, Any]
) -> bool:
    if not all(path.is_file() for path in required) or not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    except json.JSONDecodeError:
        return False


def _clear_cache_files(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _require_positive_length(length: int) -> None:
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("max_length must be a positive integer")


__all__ = [
    "DatasetBundle",
    "IndexedDataset",
    "LRA_TASKS",
    "PATHFINDER_TASKS",
    "PATHFINDER_SPLIT_PROTOCOL",
    "PATHX_RESOLUTION",
    "TokenVocabulary",
    "collate_token_pairs",
    "collate_tokens",
    "collate_values",
    "index_fingerprint",
    "is_immutable_revision",
    "limit_dataset",
    "make_loader",
    "pathfinder_split_indices",
    "prepare_lra",
    "stratified_split_indices",
    "validate_formal_source_provenance",
]
