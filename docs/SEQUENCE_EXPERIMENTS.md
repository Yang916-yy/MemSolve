# Sequence Experiments

> Current source uses model contract 13 and native ABI 10. External position
> embeddings belong to the surrounding model. Historical measurements retain
> their recorded source versions and are not new-source results.


`python -m experiments.train_transformers` is the single PyTorch entrypoint
for the current GenomicBenchmarks and Long Range Arena (LRA) experiments. It
owns data preparation, the shared encoder shell, validation-based checkpoint
selection, and the one-time held-out test evaluation.

The default LSSO core is DYNAMIC. DYNAMIC, STATIC and ZERO support native
CUDA; no-skew and no-complement require the reference backend. The shared
encoder supplies learned absolute position embeddings initialized from
`Normal(0, 0.02)`. Native ABI 10 accepts FP16 and BF16 inputs. Training recipes
still default to FP16; current model checkpoints use contract 13.

Historical results below used earlier operator and numerical contracts,
including internal relation-feature preprocessing absent from current source.
Use the recorded source revision to reproduce them. Old checkpoints are not
automatically migrated, and current recipes do not reproduce old model semantics.

All accuracies below are held-out test percentages. Three-seed summaries use
the arithmetic mean and sample standard deviation (`n - 1`) over seeds 0, 1,
and 2.

## Data Protocol

- GenomicBenchmarks uses each dataset's official train/test partition. A
  deterministic stratified validation subset is derived only from train with
  `--split-seed` (2026 in the formal runs). The test split is not used for
  model or checkpoint selection. `--max-length 0` preserves the longest source
  sequence, including Mouse examples; it does not apply a legacy fixed
  truncation.
- LRA ListOps uses the provided train/validation/test files. Text and Retrieval
  use UTF-8 bytes plus EOS. Text derives validation from its train split;
  Retrieval retains its supplied validation file.
- Generic `--task pathfinder` accepts `--pathfinder-resolution` (32 by
  default). `--task pathx` is a distinct Pathfinder-128 identity. Both loaders
  reproduce the TFDS 4.0.1 `hard` record order before the 80/10/10 slice:
  TFDS `BeamWriter` sorts `md5(b"hard" + example_key)`. The recorded protocol
  is `tfds-v4.0.1-hard-md5-order-v1`, with an index fingerprint for every
  derived partition. As in the archived LRA Transformer input pipeline,
  zero-valued background pixels are excluded by the attention/readout mask;
  retained foreground pixels keep their original row-major raster positions.

These decisions follow the archived official
[Long Range Arena repository](https://github.com/google-research/long-range-arena)
and the official
[GenomicBenchmarks repository](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks).
Formal runs hash the local source tree or cached Arrow split into
`content_sha256` metadata. Downloads require an immutable full commit SHA in
`--data-revision`; mutable branches and tags are rejected. Packed token caches
serialize the first build of each prefix, so independent seeds may safely
share one cache root.

## Run Contract

Every run writes its expanded configuration, data provenance, split
fingerprints, runtime information, model parameter count, source commit, and
configuration digest to its output directory. `--formal` requires a clean
committed worktree and a content-addressed data source. Checkpoints reject a
configuration mismatch on explicit `--resume`; a prior run is never resumed
implicitly.

`best.pt` is selected by highest validation accuracy, with lower validation
loss breaking an exact tie. Early-stop deltas determine staleness only and do
not suppress a strictly better validation checkpoint. The held-out test split
is evaluated exactly once after checkpoint selection, and `last.pt` records
the final result. An explicit resume of a completed run returns that result
without another training or test pass.

## GenomicBenchmarks

### Matched protocol

The formal comparison uses one `D128/L4/H4` encoder shell, MLP ratio 4,
dropout 0.1, learned absolute position encoding, mean pooling, AdamW, weight
decay 0.01, FP16 AMP, 5% linear warmup followed by cosine decay, gradient clip
1.0, and a 40-epoch budget. Both mixers use the same seed, split, optimizer,
schedule, checkpoint rule, and data order. The usual physical/effective batch
is `128/128`; Mouse Enhancers Ensembl uses physical batch 64 and accumulation
2, retaining effective batch 128. LSSO uses DYNAMIC CUDA at rank
32. The baseline mixer is PyTorch MHA with learned absolute positions.
The mixer parameterizations are not forced to have identical parameter counts:
for every task, LSSO has 33,264 fewer parameters than MHA. For example, the
200-position Human-or-Worm models contain 786,834 and 820,098 parameters,
respectively; task-dependent position-table sizes shift both totals equally.

The runner also exposes matched-shell global-mixer implementations for the DNA
panel: `linear_transformer`, `performer`, `nystromformer`, `cosformer`, and
`rebased`.
Linear Transformer uses the original bidirectional ELU+1 feature map. Performer
follows the FAVOR+ random-feature implementation in `performer-pytorch==1.1.4`
with a fixed projection matrix. Nystrom attention follows
`nystrom-attention==0.0.14` with
64 landmarks and six pseudoinverse iterations; its optional depthwise residual
convolution is disabled so it does not add a local prior. cosFormer follows the
official OpenNLPLab bidirectional ReLU/cosine formulation. The adaptations live
in the experiment runner to enforce its boolean padding contract and, for
variable-length batches, compute position- or landmark-dependent quantities at
each sample's true length. ReBased follows FLA's noncausal reference equation
with its normalized, learnable quadratic feature map; its projections retain
the matched shell's bias setting. Linear Transformer, Performer, and ReBased
run their complete attention cores in FP32 under CUDA AMP; this changes only
evaluation precision, not their real-valued formulas. All five execute as
ordinary PyTorch CUDA tensor programs; none has a separate vendor CUDA
extension.

Nyströmformer and ReBased are frozen as numerical references and rejected by
`--formal` until suitable bidirectional Triton implementations are available.

Some MHA `config.json` files retain generic parser fields such as
`core_mode=dynamic`, `rank=32`, and
`implementation=cuda` under `resolved_arguments`. Those fields are inactive
for MHA. The authoritative `model` record is `mixer=mha`,
`implementation=torch-mha`, `position_encoding=learned-absolute`; the MHA
runs do not execute LSSO.

### Formal results

| Task | MHA (mean +/- sample SD) | LSSO (mean +/- sample SD) | LSSO - MHA |
| --- | ---: | ---: | ---: |
| Coding vs intergenomic | 80.34 +/- 1.99 | 83.23 +/- 2.46 | +2.89 |
| Human or worm | 65.87 +/- 0.35 | 81.57 +/- 1.18 | +15.70 |
| Mouse enhancers Ensembl | 71.07 +/- 1.65 | 70.25 +/- 0.83 | -0.83 |
| Human enhancers Cohn | 66.83 +/- 0.17 | 67.24 +/- 0.26 | +0.41 |
| Human enhancers Ensembl | 76.37 +/- 0.20 | 78.91 +/- 0.05 | +2.54 |
| Human Ensembl regulatory | 92.14 +/- 0.05 | 93.10 +/- 0.02 | +0.96 |
| Human non-TATA promoters | 84.71 +/- 0.15 | 85.41 +/- 0.17 | +0.70 |
| Human OCR Ensembl | 65.61 +/- 0.54 | 68.55 +/- 0.31 | +2.93 |
| Unweighted 8-task macro | 75.37 | 78.53 | +3.16 |

LSSO is higher on 7 of 8 tasks. The macro row first averages the three seeds
within each task and then averages the eight task means, giving 75.37 for MHA
and 78.53 for LSSO. This is a matched mixer comparison, not a
GenomicBenchmarks state-of-the-art claim; no per-task significance claim is
made from three seeds.

The underlying test accuracies are:

| Task | MHA s0 | MHA s1 | MHA s2 | LSSO s0 | LSSO s1 | LSSO s2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coding vs intergenomic | 82.384 | 78.416 | 80.208 | 86.072 | 81.788 | 81.828 |
| Human or worm | 65.624 | 66.264 | 65.720 | 82.112 | 80.216 | 82.376 |
| Mouse enhancers Ensembl | 72.727 | 71.074 | 69.421 | 70.248 | 69.421 | 71.074 |
| Human enhancers Cohn | 66.940 | 66.911 | 66.638 | 67.199 | 66.998 | 67.516 |
| Human enhancers Ensembl | 76.367 | 76.571 | 76.174 | 78.854 | 78.928 | 78.957 |
| Human Ensembl regulatory | 92.189 | 92.127 | 92.101 | 93.121 | 93.088 | 93.099 |
| Human non-TATA promoters | 84.547 | 84.724 | 84.846 | 85.588 | 85.256 | 85.377 |
| Human OCR Ensembl | 66.239 | 65.315 | 65.278 | 68.194 | 68.791 | 68.651 |

### Provenance

- Seeds: `0, 1, 2` for every task and mixer.
- Source commit: `bec1f18750ca4ad364b9d2a5b2ac63359b5cf6c9`.
- Internal run-set identifier: `sequence-r32-v063`. Compact per-seed metrics
  and sanitized provenance are published in [`results/genomic`](../results/genomic/);
  checkpoints, caches, and verbose logs are omitted.
- Every included run records `formal=true`, `git_dirty=false`, and
  `validation_only=false`.
- Recorded runtime: Python 3.12.3, PyTorch 2.11.0+cu128, CUDA runtime 12.8,
  and LSSO CUDA contract 6 on an NVIDIA GeForce RTX 5070 Ti.

## Long Range Arena

### Frozen LSSO recipes

LRA uses task-specific shells rather than forcing one shape on tasks with very
different sequence and input structure. The four formal recipes are:

| Task | LSSO shell | Physical x accumulation | Effective batch | LR / epochs |
| --- | --- | ---: | ---: | --- |
| ListOps | `D128/L6/H8`, MLP 4, dropout .1, learned absolute PE + mean | `25 x 2` | 50 | `5e-4` / 40 |
| Text | `D256/L6/H8`, MLP 4, dropout .1, learned absolute PE + mean | `16 x 2` | 32 | `5e-4` / 32 |
| Retrieval | `D128/L6/H8`, MLP 4, dropout .1, learned absolute PE + mean | `16 x 4` | 64 | `5e-4` / 20 |
| Pathfinder-32 | `D256/L6/H4`, MLP 2, dropout 0, factorized row/column learned absolute PE (no PEG) + meanmax | `64 x 2` | 128 | `2e-4` / 200 |

All four use rank-32 DYNAMIC CUDA, AdamW, weight decay 0.01,
5% linear warmup followed by cosine decay, and gradient clip 1.0. Pathfinder
adds the factorized row/column position features before applying the official
nonzero-pixel mask. Its earliest possible stop is epoch 150 with
patience 10. The other tasks cannot stop before 75% of their declared budget.

### Formal LSSO results

| Task | s0 | s1 | s2 | Mean +/- sample SD |
| --- | ---: | ---: | ---: | ---: |
| ListOps | 38.350 | 37.250 | 37.200 | 37.60 +/- 0.65 |
| Text | 65.712 | 65.004 | 64.860 | 65.19 +/- 0.46 |
| Retrieval | 82.887 | 82.944 | 82.795 | 82.88 +/- 0.08 |
| Pathfinder-32 | 78.585 | 78.595 | 79.175 | 78.79 +/- 0.34 |

### Published reference context

The following published values provide recognizable LRA context. They are not
results from the LSSO shared-shell protocol:

| Model and source | ListOps | Text | Retrieval | Pathfinder |
| --- | ---: | ---: | ---: | ---: |
| Transformer, [LRA Table 1](https://arxiv.org/abs/2011.04006) | 36.37 | 64.27 | 57.46 | 71.40 |
| Performer, [LRA Table 1](https://arxiv.org/abs/2011.04006) | 18.01 | 65.40 | 53.82 | 77.05 |
| S4, [original paper](https://arxiv.org/abs/2111.00396) | 58.35 | 76.02 | 87.09 | 86.05 |
| S4-LegS, [updated training study](https://arxiv.org/abs/2206.12037) | 59.60 +/- 0.07 | 86.82 +/- 0.13 | 90.90 +/- 0.15 | 94.20 +/- 0.25 |
| LSSO, this repository (3 seeds) | 37.60 +/- 0.65 | 65.19 +/- 0.46 | 82.88 +/- 0.08 | 78.79 +/- 0.34 |

This is deliberately not presented as an apples-to-apples ranking. Official
LRA fixes its Transformer shell and permits a new model with no more than 10%
additional parameters. S4 and its updated training study use their own model,
position, readout, optimization, and reporting protocols. LSSO uses the frozen
task-specific shells above. The table therefore has no pooled average, no
cross-protocol best-value emphasis, and no state-of-the-art claim.

The formal LRA panel excludes all exploratory artifacts. In particular:

- The `87.82` Pathfinder result under
  run set `lra-formal-20260808` is a single-seed Grid-PEG run and is not
  reported. The included Pathfinder result is the later three-seed factorized
  row/column learned-PE run with PEG off.
- Path-X is not reported because there is no frozen, completed three-seed
  formal panel for it.
- Pilot, capped, validation-only, dirty-worktree, and incomplete runs are not
  included, even when an exploratory validation score is higher.

### Provenance

- ListOps, Text, and Retrieval source commit:
  `3c69cbe1e48c7408efb3a083623ddc44d2cc2fb5`.
- Their internal run-set identifier is `lra-formal-20260808`.
- Pathfinder source commit:
  `3d40f2c9c068ba6e984c54b12691dec343184d63`.
- Its internal run-set identifier is
  `lra-pathfinder-factorized-formal-20260812`. Compact per-seed metrics and
  sanitized provenance are published in [`results/lra`](../results/lra/);
  checkpoints, caches, and verbose logs are omitted.
- Every included run uses seeds `0, 1, 2` and records `formal=true`,
  `git_dirty=false`, and `validation_only=false`.
- Recorded runtime: Python 3.12.3, PyTorch 2.11.0+cu128, CUDA runtime 12.8,
  and LSSO CUDA contract 6 on an NVIDIA GeForce RTX 5070 Ti.

## Reproducing the Panels

Place the prepared datasets at explicit roots, load the published or locally
built LSSO CUDA runtime, and run from a clean committed checkout. The formal
runner records the actual source fingerprints, so changing the shell variables
below does not change dataset identity when the contents are the same.

~~~bash
DNA_DATA_ROOT=/path/to/genomic_benchmarks
LRA_DATA_ROOT=/path/to/lra
SEQUENCE_CACHE=/path/to/sequence_cache
RUN_ROOT=/path/to/sequence_runs

genomic_tasks=(
  dummy_mouse_enhancers_ensembl
  demo_coding_vs_intergenomic_seqs
  demo_human_or_worm
  human_enhancers_cohn
  human_enhancers_ensembl
  human_ensembl_regulatory
  human_nontata_promoters
  human_ocr_ensembl
)
for task in "${genomic_tasks[@]}"; do
  batch_args=(--batch-size 128 --grad-accum 1)
  if [[ "$task" == dummy_mouse_enhancers_ensembl ]]; then
    batch_args=(--batch-size 64 --grad-accum 2)
  fi
  for mixer in mha lsso; do
    for seed in 0 1 2; do
      python -m experiments.train_transformers \
        --config experiments/configs/genomic.toml \
        --task "$task" --mixer "$mixer" --seed "$seed" --formal \
        "${batch_args[@]}" \
        --data-root "$DNA_DATA_ROOT" --cache-root "$SEQUENCE_CACHE" \
        --output "$RUN_ROOT/genomic/$task/$mixer/s$seed"
    done
  done
done

for task in listops text retrieval pathfinder; do
  for seed in 0 1 2; do
    python -m experiments.train_transformers \
      --config experiments/configs/lra.toml \
      --task "$task" --mixer lsso --seed "$seed" --formal \
      --data-root "$LRA_DATA_ROOT" --cache-root "$SEQUENCE_CACHE" \
      --output "$RUN_ROOT/lra/$task/lsso/s$seed"
  done
done
~~~

To run the three active additional matched-shell DNA baselines, reuse the same loop
and replace the mixer loop with:

~~~bash
for mixer in linear_transformer performer cosformer; do
  for seed in 0 1 2; do
    python -m experiments.train_transformers \
      --config experiments/configs/genomic.toml \
      --task "$task" --mixer "$mixer" --seed "$seed" --formal \
      "${batch_args[@]}" \
      --data-root "$DNA_DATA_ROOT" --cache-root "$SEQUENCE_CACHE" \
      --output "$RUN_ROOT/genomic/$task/$mixer/s$seed"
  done
done
~~~

These baseline names are deliberately rejected for LRA: their registration is
part of the matched GenomicBenchmarks experiment, not a new repository-wide
mixer API.

The selected task resolves its frozen defaults after the TOML template is
loaded, while explicit command-line values such as `--task`, `--mixer`, and
the paths above take precedence. Do not add `--pilot-epochs`, sample caps, or
batch caps to a formal reproduction; the runner rejects these combinations.

## Reproducing the Certificate Stress Test

`experiments/certificate_diagnostics.py` measures the exact realized gain,
monotonicity margin, solved-state ratio, and deterministic adjoint-probe ratio
from trained LRA Text checkpoints. The default lengths are 1K, 2K, 4K, and 8K.
The 8K result is an operator-level stress test, not a task-accuracy evaluation;
the learned 4K absolute-position table is repeated rather than extrapolated.

~~~bash
python -m experiments.certificate_diagnostics \
  --checkpoint /path/to/text/lsso/s0/best.pt \
  --checkpoint /path/to/text/lsso/s1/best.pt \
  --checkpoint /path/to/text/lsso/s2/best.pt \
  --output /path/outside/the/repository/certificate-length-depth
~~~

The command requires CUDA because it advances the trained encoder through its
native mixer path. Diagnostics themselves use FP64 compact linear algebra and
never construct an `N x N` token operator. The output directory contains raw
observations, percentile summaries, complete protocol metadata, and the plot.

### MHA + RoPE + learned absolute embeddings control

`--mixer mha_rope` preserves the MHA baseline's QKV/output parameters and
initialization, and retains the encoder's learned absolute position embeddings.
It applies standard RoFormer adjacent-pair rotation to Q and K only, across the
full head dimension, with base 10000 and token indices starting at zero. V is
unchanged. Source: https://github.com/ZhuiyiTechnology/roformer . This is a
standard MHA RoPE control; LSSO does not use it.
Explicit rotation-matrix forward/input-gradient oracles cover padding and empty
examples, and the position-zero case agrees with ordinary MHA.
