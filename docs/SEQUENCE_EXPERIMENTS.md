# Sequence experiments

`python -m experiments.train_transformers` provides the shared sequence encoder,
data preparation, checkpoint selection, and held-out evaluation. LRA recipes
are retained in `experiments/configs/lra.toml`. The loader also supports local
GenomicBenchmarks data; old genomic training recipes and result tables are not
part of the current paper.

## Encoder and data

MemSolve uses shared relation/content projections, a compact equilibrium,
and learned absolute positions in the surrounding encoder. `--core-mode`
selects `equilibrium` or `complement`. The runner also has matched MHA and
other comparison mixers; use `--help` for supported task/mixer combinations.

- ListOps uses the supplied train/validation/test files.
- Text and Retrieval use UTF-8 bytes plus EOS. Text derives validation from
  train; Retrieval retains its supplied validation file.
- `pathfinder` accepts `--pathfinder-resolution` (default 32); `pathx` denotes
  Pathfinder-128. Splits reproduce the TFDS 4.0.1 hard-record MD5 order before
  the 80/10/10 partition. Background pixels are masked; foreground pixels retain
  their original raster positions.
- GenomicBenchmarks uses official train/test partitions and a deterministic
  stratified validation subset from train. `--max-length 0` retains the longest
  source sequence rather than truncating it.

Protocol sources: [LRA](https://github.com/google-research/long-range-arena) and
[GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks).
Prepared splits record content hashes and partition fingerprints. Hub downloads
require a full commit SHA in `--data-revision`. Token caches can be shared by
independent seeds.

## Launch

```bash
python -m experiments.train_transformers \
  --config experiments/configs/lra.toml --task listops \
  --mixer memsolve --seed 0 --formal \
  --data-root /datasets/lra --cache-root /datasets/sequence-cache \
  --output /runs/lra/listops/memsolve-seed0
```

Select `text`, `retrieval`, or `pathfinder` for the other LRA tasks. Task defaults
are resolved after the TOML template; explicit CLI settings take precedence.
`--formal` requires a clean committed source tree and content-addressed data.
Sample/batch caps and pilot budgets are diagnostic settings, not formal runs.

The output directory records resolved settings, data provenance, parameter
count, source revision, and runtime. `best.pt` is selected by validation
accuracy, with validation loss breaking ties. The test set is evaluated after
selection; `last.pt` records the final result. Use explicit `--resume` with the
same configuration. Run artifacts and historical measurements remain outside
Git's current tree.

## Position control

`--mixer mha_rope` rotates MHA Q/K in adjacent pairs with base 10000, while
retaining the encoder's learned absolute positions and leaving V unchanged.
This is a separate comparison mixer. Optional MemSolve RoVE is defined in the
[core contract](CORE_CONTRACT.md), not by this MHA option.
