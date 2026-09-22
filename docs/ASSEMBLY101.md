# Assembly101 data and evaluation protocol

> Current source uses model contract 13 and native ABI 11. External position
> embeddings belong to the surrounding model. Historical measurements retain
> their recorded source versions and are not new-source results.


This workflow prepares the official TSM features and evaluates saved predictions
with the original LTContext data alignment and metrics. The same entrypoint
trains a hybrid that replaces only the nine stage-1 long-range branches with
full-sequence LSSO; the upstream local modules and stages 2–4 are retained.

## Sources and environment

Use the official [LTContext checkout](https://github.com/LTContext/LTContext)
at `ac74722b00b52b7eb9eb3d6fa8600c762e6f369b`. The audit/evaluation commands
reject another revision or tracked changes to its model, dataset, or metric code.
Upstream code is imported from that external checkout, not vendored into LSSO.
LTContext and Assembly101 have their own CC BY-NC 4.0 licenses.

Install data dependencies with `python3 -m pip install -e '.[assembly101]'`.
The native LSSO environment is documented in [CUDA_CONTRACT.md](CUDA_CONTRACT.md).
Do not install LTContext's old torch 2.0 environment over it. Training additionally
uses upstream loss, optimizer, scheduler, dataset, collation and metric code.
The runner owns checkpoint bookkeeping and does not require upstream plotting.

The [official dataset mirror](https://huggingface.co/datasets/cvml-nus/assembly101)
is gated: the account must be approved and authenticated. Use standard Hugging
Face credentials; never put tokens into configs, command-line arguments, or Git.
The TSM archives total about 403 GB (decimal), before extraction. Raw videos,
DINOv2 features, hand poses, and fine-grained labels are not needed here.

## Preparation

Run from the LSSO checkout. Example persistent data root: `/path/to/assembly101`.

```bash
python3 -m experiments.assembly101 download --data-root /path/to/assembly101
python3 -m experiments.assembly101 extract --data-root /path/to/assembly101 \
  --ltcontext /root/LTContext --workers 4
python3 -m experiments.assembly101 audit --data-root /path/to/assembly101 \
  --ltcontext /root/LTContext --load-type lmdb \
  --workers 8 --output /path/to/assembly101/audit.json
```

Download pins a Hugging Face revision before transfer; retries reuse it. Only
TSM view archives, their reader, and coarse annotations are downloaded. Extraction
streams each archive's `data.mdb`, checks ZIP CRC, and creates
`db_TSM_features/<view>/data.mdb`. It links downloaded coarse annotations into
the external LTContext checkout. Archives are retained. No existing annotation
directory pointing elsewhere is overwritten.

Audit requires every official split record, validates feature shape `[2048,T]`,
finite values, target length and class IDs, and records SHA-256 digests of the
consumed feature slices, aligned labels, split files, and annotation files.
This scans the full dataset and is not a cheap filename-only check. Missing
features or records rejected by the upstream loader cause failure rather than
silently shrinking the dataset. For preconverted NumPy features, use
`--load-type numpy`; expected files are
`TSM_features/<video_id>/<view>/features.npy` with shape `[2048,T]`.
Audit workers load, validate and hash their own samples, returning only small
records in the original split order. Parallel and serial audit hashes agree.

The pinned splits contain 4684 training and 1424 validation entries, with no
recording-ID overlap. These are view/action-phase entries, not unique recordings.
The upstream reader preserves frame sampling rate 1 and nearest-neighbor target
alignment where feature and annotation lengths differ. Annotations index the
30-fps feature timeline, not raw 60-fps video.

## Evaluation

Export final-stage argmax predictions for every validation entry, in the
upstream `TEST.SAVE_PREDICTIONS` layout:

```
<predictions>/<assembly-or-disassembly>/<video_id>/<view>/pred.npy
```

Each file must be a one-dimensional integer array of exactly the aligned target
length with class IDs in `[0,201]`. Ground truth is loaded from official data;
a colocated `gt.npy` is never trusted as the evaluation source.

```bash
python3 -m experiments.assembly101 evaluate --data-root /path/to/assembly101 \
  --ltcontext /root/LTContext --load-type lmdb \
  --predictions /path/outside/repository/assembly101/predictions \
  --output /path/outside/repository/assembly101/validation_metrics.json
```

All five metrics use LTContext's original code: MoF, Edit, F1@10, F1@25,
F1@50. Report the unweighted mean across validation entries, in percent.
Do not pool all frames or segment counts globally. All 202 classes, including
class 0, participate; only padding ID -100 is ignored. Missing or extra prediction
files are rejected. The output includes per-entry results and prediction hashes.
The checkpoint selection score is the sum of the five percentage metrics / 100,
matching upstream's normalized `metrics_sum`. Select one checkpoint with this
rule and report all its metrics; do not select a separate epoch per metric.

The upstream Assembly101 loader supports `train` and `val`, not a separate
labeled test split. Its function named `test()` evaluates `TRAIN.EVAL_SPLIT`.
Label these results **validation**, not test. Repeated model/parameter selection
on this split does not provide an independent held-out test estimate.

For training with the upstream runner, explicitly set `TEST.ENABLE False`;
otherwise its default attempts a post-training evaluation with an empty checkpoint
path. For standalone evaluation, explicitly set `TRAIN.ENABLE False`,
`TEST.DATASET assembly101`, `TRAIN.EVAL_SPLIT val`, and a checkpoint path.

## Stage-1 experiments

Keep official settings: 4 stages, 9 blocks/stage, widths 64 then 32, one attention
head, local window 64, long-term grouping 64, dilation `2**i`, Adam at 2.5e-4,
weight decay 1e-3, batch 1, 120 epochs, CE + 0.17 clipped temporal MSE.
The first 15 epochs use a constant LR, then cosine decay to 5e-5; despite the
config field name this is not a linear warmup. `USE_INSTANCE_NORM=True` creates
Identity in the pinned code; retain that actual behavior in both arms.

The experimental arms replace the nine `stage1.layers[i].ltc_attn`
modules with full-sequence LSSO, dim=64, heads=1, rank=16 or 32, bias=True,
DYNAMIC. Later stages and local operations remain upstream-owned.
This changes both the long-term mixer and its connectivity: original LTContext
uses 64 interleaved subsequences; full-sequence LSSO mixes all timestamps.
Report parameter counts and acknowledge differing internal projections,
activation/dropout structure, and precision boundaries. The adapter uses LSSO's
own projections and readout without adding an attention-weight dropout. The
block's surrounding residual and dropout remain upstream-owned. Native LSSO
activations are BF16; upstream layers, loss and all parameters remain FP32.
No whole-model AMP, gradient clipping, cropping or downsampling is introduced.
Use identical seeds and data/selection rules for both ranks.

The accelerated runner defaults to `--attention-backend sdpa`: the remaining
LTContext attentions evaluate the same masked softmax attention using PyTorch
SDPA, with the original attention dropout probability (zero in evaluation).
Window/group geometry, projections, residuals and parameters are unchanged.
The upstream caller discards attention weights, so the adapter need not return
the dense attention matrix. CPU and CUDA tests compare output and gradients,
including entirely masked groups. Dropout RNG consumption and floating-point
summation can differ from the original kernel. `--attention-backend upstream`
retains the original execution for measurement. This follows the official
[PyTorch SDPA API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).
Gradient finiteness is checked with the public foreach infinity-norm reduction;
the check does not scale, clip or otherwise modify gradients.

### Bucketed CUDA Graph execution

`--execution-backend graph-compile` captures model forward/backward per length
bucket and applies PyTorch Inductor fusion to the dilated-convolution/GELU/mask
regions and the original loss. The default
buckets are 2048, 4096, 6144, 8192, 12288, 16384, 24576 and 49152 frames;
`--graph-buckets` sets an explicit alternative. Samples retain their original
shuffle order and batch size 1. A sequence exceeding the largest bucket is an
error, not a cropped input. `graph` uses the same buckets without compilation;
`eager` retains the previous execution path.

Padding uses the original attention masks and grouping stride. Input-projection
bias is masked before each stage's first temporal convolution, so nonexistent
frames behave like the convolution's original zero padding. Subsequent blocks
already mask their outputs. This requires the Assembly101 configuration's actual
Identity normalization. The runner slices all stage logits to the original
length before calling the unchanged upstream loss, preserving both CE and the
temporal MSE's boundary and denominator. Dropout distributions are retained;
padding changes RNG consumption and floating-point results relative to eager
unpadded training. Validation runs the original lengths without graph capture
or compilation.

Each bucket owns an independent graph pool: shuffled buckets are not permitted
to share a pool that assumes a fixed replay order. They share the original model
parameters. Graph warmup does not step the optimizer, and warmup/capture restore
the CPU/CUDA RNG states. Graphs are reconstructed after resume. Deterministic
PyTorch algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are used for this path;
the selected settings are recorded in checkpoint identity. Adam and finite-value
checks remain outside the graph. Native LSSO and attention remain outside
Inductor and are captured by CUDA Graph without changing their implementation.

Convolution regions use separate compilation caches and automatic dynamic-shape
specialization; the loss supports real sequence lengths dynamically. Graph
replay itself always uses a fixed bucket shape. Compiling complete attention
blocks was substantially slower to prepare and did not consistently improve
replay throughput in the local measurements, so that scope is not enabled.
Inductor's own CUDA Graph mode is disabled to avoid nested capture. This follows
the official [CUDA Graph memory and capture constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs),
[torch.compile API](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)
and [static/dynamic shape guidance](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html).
FP64 reference tests compare padded and original full-model logits,
losses and gradients; FP64 is used only for validation, never in the training path.

For an explicitly authorized change of execution backend, use `--resume
--upgrade-execution`. This allows only runner/backend/determinism metadata to
change; data, model configuration, core source, native library, seed and rank
must still match. Each transition records the parent checkpoint hash and old/new
identities in `execution-transitions.jsonl`. Ordinary `--resume` remains strict.

Run a data-backed pilot before each formal run. It tests full training steps on
the shortest, median, 90th-percentile and longest training sequences, followed
by 200 shuffled training steps and 20 validation entries. Pilot weights are
discarded; formal training starts from the configured seed. Pilot numbers are
not formal validation results.

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.assembly101 pilot \
  --data-root /path/to/assembly101 --audit /path/to/assembly101/audit.json \
  --rank 16 --seed 0 --output /path/outside/repository/assembly101/pilot-r16
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.assembly101 train \
  --data-root /path/to/assembly101 --audit /path/to/assembly101/audit.json \
  --rank 16 --seed 0 --output /path/outside/repository/assembly101/r16-seed0
```

The default training budget is 120 epochs. Each output directory records `metadata.json`, the
resolved config, per-epoch `metrics.jsonl`, live `status.json`, `last.pt`,
`best.pt`, `best_metrics.json`, and best-checkpoint validation predictions.
`--resume` resumes `last.pt` and verifies source, native library, audit, rank,
seed, torch and configuration identity. The sampler has a separate saved RNG
so rank-dependent initialization and DataLoader worker seeding do not alter
the intended sample order.
cuDNN convolution algorithms are deterministic and benchmarking is disabled in
training. This does not promise bitwise reproducibility
across hardware or software versions. The setting follows PyTorch's
[reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

Unlike the pinned upstream training loop, this runner evaluates **before**
saving a checkpoint. Upstream saves before validation and passes the previous
epoch's metrics to its checkpoint helper. Here checkpoint selection always
uses the metrics of the weights being saved. The LR schedule remains unchanged.
Nonfinite losses or gradients stop a run before the optimizer update.

After training, evaluate the exported best predictions with the strict
`evaluate` command above. This independently checks coverage and reproduces
the selected checkpoint's metrics from the official ground truth.

## Published baseline provenance

Use Table 2 of Bahrami et al., *How Much Temporal Long-Term Context is Needed
for Action Segmentation?*, ICCV 2023
([paper, arXiv v2](https://arxiv.org/pdf/2308.11358v2)). It uses the provided
2048-dimensional TSM features and reports Assembly101 **validation** results:

| Method | F1@10 | F1@25 | F1@50 | Edit | MoF / Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| LTContext, paper Table 2 | 33.9 | 30.0 | 22.6 | 30.4 | 41.2 |

The pinned repository's `MODEL_CARD.md` instead lists F1@50 23.2 and Acc 41.6
for its released checkpoint. Do not mix those with the paper's other metrics.
The paper states the validation split and features; the pinned implementation
defines this runner's exact macro aggregation. Label the baseline as published,
not locally reproduced or a controlled ablation. Report our protocol and
precision differences. The paper's inference time is not a same-hardware
efficiency baseline for these A800 experiments.
