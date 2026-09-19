# Food-101 runner

`python -m experiments.food101` trains an LSSO-S/16 classifier from scratch.
Choose `--variant dynamic`, `static`, or `zero`. All variants use width 384,
depth 12, six heads, rank 32, 224-pixel inputs and learned full-grid 2D position
embeddings. Shared parameters have identical initialization for a fixed seed.
The runner defaults to `--implementation cuda`; `reference` is also available.

The default schedule is 30 epochs, batch size 256, AdamW with peak learning rate
0.0005 and weight decay 0.05, three warmup epochs and cosine decay to 0.000001.
It uses BF16 AMP, label smoothing 0.1 and gradient clipping at 1. These short
runs are not the ImageNet DeiT III training protocol.

Use the official train/test partitions. The test set is evaluated each epoch
for a curve; comparison uses the fixed final epoch, with no test-based early
stopping or checkpoint selection.

```bash
python -m experiments.food101 \
  --data /path/to/food101 --output /path/outside/repository/food101-dynamic \
  --variant dynamic --implementation cuda --epochs 30 --seed 0
```

The data directory contains `food-101/images` and `food-101/meta`. The official
archive MD5 is `85eeb15f3717b99a5da872d97d918f87`, as recorded by torchvision.
Dataset source: <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>.

`run.json` records configuration and metadata checksums. `status.json` reports
progress, `metrics.jsonl` stores epoch curves, and `last.pt` stores model,
optimizer and random states for `--resume`. `final_metrics.json` records the
fixed final epoch. Store all generated artifacts outside the repository.
