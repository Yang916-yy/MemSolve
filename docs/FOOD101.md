> Structure update: MemSolve now uses the restored no-RankRotary accretive
> operator. Vision uses CPE/GELU and token-LN mean pooling without LayerScale,
> following ordinary ViT3; BERT retains its own SwiGLU shell. Historical results and recipe
> discussions below are not measurements of a new run of this restoration.
> See [the current architecture](ARCHITECTURE.md) and [core contract](CORE_CONTRACT.md).

# Food-101 runner

`python -m experiments.food101` trains a MemSolve-ViT-S/16 classifier from scratch.
The restored model uses width384, depth12, six heads, rank32, a GELU MLP of
width1584, per-block CPE and token-LN mean pooling without LayerScale. Linear DropPath reaches0.1.
Initialization is deterministic for a fixed seed. Historical measurements keep
their recorded source contracts; no new Food-101 experiment has been launched.
The runner defaults to `--implementation cuda`; `reference` is also available.

The default schedule is 30 epochs, batch size 256, AdamW with peak learning rate
0.0005 and weight decay 0.05, three warmup epochs and cosine decay to 0.000001.
It uses BF16 AMP, label smoothing 0.1 and gradient clipping at 1. These short
runs are a separate short schedule, not the full ImageNet ViT³-derived recipe.

Use the official train/test partitions. The test set is evaluated each epoch
for a curve; comparison uses the fixed final epoch, with no test-based early
stopping or checkpoint selection.

```bash
python -m experiments.food101 \
  --data /path/to/food101 --output /path/outside/repository/food101-qkv \
  --implementation cuda --epochs 30 --seed 0
```

The data directory contains `food-101/images` and `food-101/meta`. The official
archive MD5 is `85eeb15f3717b99a5da872d97d918f87`, as recorded by torchvision.
Dataset source: <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>.

`run.json` records configuration and metadata checksums. `status.json` reports
progress, `metrics.jsonl` stores epoch curves, and `last.pt` stores model,
optimizer and random states for `--resume`. `final_metrics.json` records the
fixed final epoch. Store all generated artifacts outside the repository.
