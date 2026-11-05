# Published Experiment Logs & Summaries

This directory contains the complete training logs, evaluation metrics, and resolved configurations for every experiment reported in [the paper](../paper/main.pdf).

| Directory | Contents | Key Reported Result |
| --- | --- | --- |
| [`imagenet/`](imagenet/) | 300-epoch logs & configs for MemSolve-ViT-T / S / B | Best EMA Top-1: **74.73% / 80.99% / 82.71%** |
| [`coco/`](coco/) | Mask R-CNN 1× logs & configs for Small and Base with ViT-Adapter | Box / Mask AP: **44.7 / 40.6** (S), **46.6 / 41.8** (B) |
| [`ade20k/`](ade20k/) | UPerNet 160k logs & configs for Small and Base with ViT-Adapter | Single-scale mIoU: **47.20** (S), **48.33** (B) |
| [`mlm/`](mlm/) | 70,000-step C4 pretraining history for MemSolve-BERT (learned positions) | Validation masked NLL: **1.6211** (acc **66.38%**) |
| [`glue/`](glue/) | 25 fine-tuning runs across 8 GLUE tasks for MemSolve-BERT (learned positions) | Average GLUE score: **81.85** (**81.9**) |
| [`lra/`](lra/) | 50 controlled runs across 10 mixers × 5 LRA tasks with RoVE | Mean accuracy across 5 tasks: **67.35%** (MemSolve) |
| [`imagenet100/`](imagenet100/) | Four 100-epoch Small ablations (Equilibrium, Complement, no-skew, $r=16$) | Equilibrium $r=32$ Top-1: **84.48%** |
| [`efficiency/`](efficiency/) | Latency & peak memory across sequence lengths for MemSolve, FlashAttention-2, and BiMamba | **7.8×** faster than FlashAttention-2 at $N=8192$, **28.9×** at $N=32768$ |

## Guide to Each Subdirectory

- **`imagenet/`**: `tiny.jsonl`, `small.jsonl`, and `base.jsonl` log every epoch's training loss, raw validation accuracy (`val_acc1`), and EMA validation accuracy (`ema_val_acc1`). `metadata.json` records the resolved configurations and best epochs.
- **`coco/` & `ade20k/`**: Each scale (`small/`, `base/`) includes the full MMEngine training log, structured JSONL metrics, and resolved OpenMMLab config.
- **`mlm/` & `glue/`**: Store the step-by-step pretraining and 25-run GLUE fine-tuning logs for the primary learned-position `MemSolve-BERT` model (`mlm/history.jsonl`, `glue/results.jsonl`, `glue/history.jsonl`; parameter-free RoVE summary metrics are reported in Section 5.2 and Appendix D.2 of the paper).
- **`lra/`**: Organized as `lra/<model>/<task>/`, with 50 complete run logs and configs plus aggregated tables in `lra/summary.json`. Any run can be reproduced via `python -m experiments.reproduce_lra <model>/<task> ...`.
- **`imagenet100/`**: Contains the 100-epoch logs for `equilibrium_r32`, `complement_r32`, `no_omega_r32`, and `equilibrium_r16`, along with the fixed 100-class subset list in `metadata.json`.
- **`efficiency/`**: `measurements.jsonl` records every benchmark command and timing/memory output on an NVIDIA A800-SXM4-80GB GPU under BF16 autocast and CUDA Graphs (2 independent trials per setting, each taking the median of $7 \times 50$ graph replays).
