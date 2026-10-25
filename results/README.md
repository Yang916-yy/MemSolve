# Published experiment logs

These records support the experiments in [the paper](../paper/main.pdf).
They contain complete recorded histories for the included runs, their resolved
configurations, and summaries calculated from those records.

| Directory | Evidence | Reported result |
| --- | --- | --- |
| [imagenet/](imagenet/) | Tiny, Small, Base: 300 epochs each; ordinary and EMA validation | Best EMA Top-1: 74.734 / 80.990 / 82.706% |
| [mlm/](mlm/) | Absolute-position BERT with centered convolution: 70,000 steps | Masked-token NLL 1.62108; accuracy 66.3822% |
| [glue/](glue/) | 25 fine-tuning runs across eight tasks, with per-run histories and metrics | Mean of task means: 81.8533 |
| [imagenet100/](imagenet100/) | Four 100-epoch Small ablations, with subset class IDs and configurations | Equilibrium r32: 84.48% |
| [efficiency/](efficiency/) | MemSolve, FlashAttention, BiMamba; two trials per setting | Forward and forward–backward latency and peak allocated memory |

## Reading the records

**ImageNet.** `tiny.jsonl`, `small.jsonl`, and `base.jsonl` are byte-for-byte
copies of the original epoch logs. `val_*` refers to ordinary model weights;
`ema_val_*` refers to EMA weights. Accuracy is in percent. The paper's convergence
curve uses Small's `val_acc1`, while its classification table uses the maximum
`ema_val_acc1`. `metadata.json` includes the resolved model, recipe, software,
source identity, and selected epochs. The resolved `run.model.mlp_ratio` is the
actual tier-specific FFN ratio; `run.train.mlp_ratio` is the recorded shared
recipe default.

**MLM.** `history.jsonl` contains every entry of the final HF Trainer state's
`log_history`, including intermediate evaluations. Training masks 30% of tokens;
the fixed validation set masks 15%. `metadata.json` records the architecture,
recipe, dataset revision, tokenizer identity, mask settings, final selection,
and evaluation over 169,544 masked targets. This is the absolute-position model
used for the paper's GLUE results. See [the BERT guide](../docs/BERT_MLM.md) for
its configuration relative to the supplied RoVE recipe.

**GLUE.** `results.jsonl` retains each run's training and final-checkpoint
validation metrics, and adds its paper score on a 0–100 scale. `history.jsonl`
contains all Trainer history entries, identified by task and seed. The aggregate
first averages seeds within each task, then takes an unweighted mean of the
eight task means. MNLI averages matched/mismatched accuracy; QQP averages
accuracy/F1; MRPC uses F1; STS-B uses Spearman; CoLA uses Matthews correlation;
other tasks use accuracy. Seed IDs and initialization checkpoint hashes are
recorded in `metadata.json`. RTE, MRPC, and STS-B share the seed-19 MNLI parent.
For tasks using the Composer decay convention, `training.weight_decay`
records the equivalent PyTorch AdamW coefficient: recipe decay divided by
peak learning rate. The two fields use different conventions.

**ImageNet-100.** Each variant has its unchanged 100-row epoch log, including
per-layer eta values. `metadata.json` records the common subset and each
variant's configuration. The table selects the highest Top-1 and its paired CE.
The early/late comparisons use mean error over epochs 11–30 and 81–100:
`100 * (comparison_error - equilibrium_error) / comparison_error`.

**Efficiency.** `measurements.jsonl` retains each command argument list and
its original metric string. `mode=train` includes forward and backward, without
an optimizer step. The paper averages `median_gpu_ms` (CUDA-event time) over
the two independent trials; each trial takes the median of seven groups of
50 graph replays. Memory averages `peak_allocated_mib` over the same trials.
`median_ms` is the separately recorded host timer. All three operators use
CUDA Graphs. Tiny uses the six-head measurements in `batch_id=tiny-head6`;
`main` contains Small, Base, and the fixed-width length sweep. The upstream
revisions and compatibility validation are in `metadata.json`.

## Provenance and file integrity

[manifest.json](manifest.json) records SHA-256 hashes of the original source
files and the published data files. Source paths are relative run identifiers,
not paths required on a reproducer's machine. Metadata replaces local paths
with `run:`, `repository:`, or `local:` references. NLP histories are extracted
from Trainer states; GLUE histories and results are grouped into one file each.
Dataset file-stat inventories, launch queues, and checkpoint payloads are
omitted. Numeric measurements are unchanged. Code revisions and source hashes
are retained where recorded; some runs used modified source snapshots rather
than an unmodified Git checkout.

Check published data integrity from the repository root:

```python
import hashlib
import json
from pathlib import Path

root = Path("results")
manifest = json.loads((root / "manifest.json").read_text())
for name, record in manifest["exports"].items():
    payload = (root / name).read_bytes()
    assert len(payload) == record["bytes"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"], name
```

Training and benchmark entrypoints remain in `experiments/` and `benchmarks/`.
These logs can be inspected without loading datasets, weights, or CUDA kernels.
