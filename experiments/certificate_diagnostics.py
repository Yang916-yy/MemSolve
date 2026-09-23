"""Summarize historical reflected-operator certificate measurements.

These metrics do not certify the current independent-QKV readout. Generate
old measurements with their recorded source revision; this utility only
aggregates already recorded CSV rows and never loads a model checkpoint.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any, Iterable
import numpy as np

METRICS = (
    "q",
    "contraction_slack",
    "mu",
    "state_ratio",
    "adjoint_ratio",
    "state_bound_usage",
    "adjoint_bound_usage",
)


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def summarize(
    records: list[dict[str, float | int]],
) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[int, int], list[dict[str, float | int]]] = {}
    for record in records:
        groups.setdefault((int(record["length"]), int(record["layer"])), []).append(
            record
        )
    rows: list[dict[str, float | int | str]] = []
    for (length, layer), group in sorted(groups.items()):
        for metric in METRICS:
            row: dict[str, float | int | str] = {
                "length": length,
                "layer": layer,
                "metric": metric,
                "count": len(group),
            }
            row.update(_percentiles(float(item[metric]) for item in group))
            rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        records = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    _write_csv(summarize(records), args.output)


if __name__ == "__main__":
    main()
