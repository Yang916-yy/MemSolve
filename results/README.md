# Published Results

This directory contains the compact, machine-readable evidence behind the
results reported in the README and paper. It publishes final per-seed test
metrics and enough provenance to identify the code, data, model shell, and
runtime used by each formal panel.

| Result | Files | Scope |
| --- | --- | --- |
| GenomicBenchmarks | [`genomic/per_seed.csv`](genomic/per_seed.csv), [`genomic/provenance.csv`](genomic/provenance.csv), [`genomic/metadata.json`](genomic/metadata.json) | Matched MHA/LSSO shell, eight tasks, three seeds |
| Long Range Arena | [`lra/per_seed.csv`](lra/per_seed.csv), [`lra/provenance.csv`](lra/provenance.csv), [`lra/metadata.json`](lra/metadata.json) | LSSO ListOps, Text, Retrieval, and Pathfinder-32, three seeds |
| Contraction certificates | [`certificates/summary.csv`](certificates/summary.csv), [`certificates/metadata.json`](certificates/metadata.json) | Layerwise 1K/2K/4K/8K operator stress test |
| CUDA wall clock | [`cuda/wall_clock.csv`](cuda/wall_clock.csv), [`cuda/metadata.json`](cuda/metadata.json) | Complete mixer forward and forward-backward timing |

Accuracies in the CSV files are percentages. `selected_epoch` is the epoch of
the validation-selected checkpoint evaluated once on the held-out test split.
Each `config_digest` is the SHA-256 digest stored by the formal runner for the
fully resolved run configuration. Dataset content hashes and recorded protocol
metadata are included in the corresponding provenance table; split
fingerprints are included where the formal artifact records them.

The repository intentionally omits checkpoints, caches, and verbose training
logs. Those files are large and are not needed to audit the reported numbers.
The LRA CSV contains only results trained by this repository; published
comparison values in the paper remain attributed to their original sources.

## Historical evidence versus current source

The published CUDA metadata records native contract **6**. Current source uses
native ABI **10** and model contract **13**, including a revised mixed-precision
policy, biased projection fusion and removal of internal feature rotation. These source updates do not rewrite the
CSV measurements or establish new task results. Consult each panel's metadata
for its actual runtime and protocol; do not label historical tables as current
source benchmarks.

The September 2026 optimization measurements are exploratory operator checks.
They do not provide new ImageNet accuracy, COCO AP or ADE20K mIoU. New formal
panels require their own matched runs and provenance, rather than copying the
old metrics into a new runtime description.

Archived CSV and JSON configuration labels remain unchanged: they describe
the measured models, including preprocessing absent from current source.
Local ablation queues, intermediate results and machine status are not published here.
