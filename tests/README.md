# Test Map

The test tree is organized by the public boundary it protects. Run the smallest
matching scope during normal work; each file is a focused, Git-tracked contract
rather than a local experiment or benchmark.

| Scope | Command | Covers |
| --- | --- | --- |
| `core` | `python -m pytest tests/core -m "not cuda"` | Operator mathematics, model/configuration behavior, and public API. |
| `cuda` | `python -m pytest tests/cuda` | CUDA device contract, dispatch, and CUDA oracle comparisons. |
| `integrations` | `python -m pytest tests/integrations -m "not cuda"` | timm, OpenMMLab and Hugging Face BERT adapters. |
| `experiments` | `python -m pytest tests/experiments -m "not cuda"` | Data protocols and training entrypoints. |
| `repository` | `python -m pytest tests/repository` | Repository shape and ownership contract. |

`tests/experiments/test_nlp.py` uses the optional `nlp` environment. It checks
fixed MLM masks, token-weighted loss, C4 partition/window integrity, preflight,
best-validation export, independent MLM evaluation, continued pretraining,
MLM/retrieval Trainer resume, disjoint rank/worker local-stream sharding,
streaming resume and a local MTEB retrieval task without downloading datasets. A local stock BERT checkpoint also exercises the evaluation baseline path.

ImageNet data checks use WebDataset and WIDS from the `vision` or `test` extra.
They cover global virtual-group RA, independent group mixing with one physical
forward, persistent-worker epoch changes and restart replay using local tar fixtures.

Use `python -m pytest -m cuda` to select all CUDA-marked checks across scopes.
Run `python -m pytest -q` only for a release, a broad refactor, or a change
that crosses multiple scopes.

## Mixed-precision and optional checks

CUDA math changes also require the relevant `tests/core` checks: projection
fusion, FP32 bias cancellation, strided tails and gradient comparisons live
there. GPU core checks carry the `cuda` marker, including bias cancellation and
strided tails. Set `CUDA_VISIBLE_DEVICES=''` for an explicitly CPU-only audit
on a machine with ongoing training.

```bash
python -m pytest -q tests/core tests/cuda
python tools/check_repository.py
git diff --check
```

Read skip reasons with `-rs`. Missing timm/OpenMMLab packages must not be reported
as successful full-model validation. Test other SM targets on their own hardware.
Benchmarks and exploratory probes belong
outside the tracked test tree.

Before the 2026-09-12 main publication, the full suite passed 342 tests with
6 skips; only SM120 was executed. This is a dated verification record, not a
promise that all optional stacks or architectures were exercised.

Model contract 26 / native ABI 17 covers the accretive-equilibrium
oracle checks: equilibrium/complement, masks, compact solves, complement tails,
forward/parameter/input gradients and CUDA Graph replay. The native extension
must be built before GPU tests; skipped native tests are not successful oracle
validation. See the [CUDA contract](../docs/CUDA_CONTRACT.md).

The inverse factor, common compact adjoint, and product epilogues have independent
FP64 oracle checks. Eta-adjoint reductions cover small/large scales and both
row-major and transposed dense inputs; they must respect incoming strides.

On 2026-10-12, SM80/A800 validation passed 363 checks across `tests/core`,
`tests/cuda` and `tests/integrations/test_timm.py`, plus 15 checks in
`tests/integrations/test_transformers.py` using the NLP environment. These checks
cover operator and adapter correctness, not a new training run or validation
of other GPU architectures. Performance probes and their logs remain outside Git.

On 2026-10-16, the full suite in the NLP environment passed **561 checks**,
with 9 optional checks skipped and 3 upstream MTEB deprecation warnings.
This includes local C4 streaming, rank/worker partitioning, Trainer resume,
SM80 inverse-factor accuracy and exact sequence-parallel checks. The manuscript
also compiled successfully. These checks do not establish NLP task accuracy.

The MLM compile-graph path has focused CUDA tests for masked-target-weighted
gradient accumulation, parameter updates, HF save/load, zero-target batches,
train/eval switching and advancing dropout RNG. On 2026-10-16 these 3 CUDA
checks and all 12 CPU NLP workflow checks passed. A separate two-A800 real-C4
probe exercised validation, checkpoint reload and continued graph replay;
profiling confirmed Inductor kernels and forward/backward graph launches.

## RoVE, downstream adapters and memory lifetimes

The current CUDA oracle suite covers optional RoVE in both core modes, batched
position tables, nonmultiple token/channel tiles, sparse and fully masked
samples, long-chunk statistics, and sequence-sharded gradients. FP64 comparisons
retain the existing error thresholds. Scratch-alias checks cover multiple output
CTAs; repeated-backward checks verify saved inputs and parameters are unchanged.

GLUE checks cover the task metrics, MLM/classifier checkpoint transfer, and
Composer-to-PyTorch AdamW decay conversion. The dense integration checks cover
ViT-Adapter, rectangular CPE, padding, EMA checkpoint selection, layer decay,
MMDetection/MMSegmentation heads, and the BF16/FP32 native-operator boundaries.
Use the dedicated OpenMMLab environment for these optional checks.

The supported compile path keeps `MemSolve.forward` outside Dynamo tracing,
compiles the surrounding model blocks, and captures both in an outer CUDA
Graph. This is the configuration used by the ImageNet and MLM entrypoints;
it does not assert full-graph Dynamo support for the native mixer itself.

On 2026-10-18, the full suite in the NLP environment passed **695 tests**, with
14 skips and 4 warnings on SM80/A800. Three warnings concern upstream MTEB APIs;
one reports an AccumulateGrad stream mismatch during a passing compile/graph
test. In the dedicated OpenMMLab environment, 29 CPU checks and 3 CUDA checks
passed. These are separate environment runs, not additional unique test cases.
The repository consistency check passed, and a wheel build verified that the
project notices and redistributed third-party license texts are packaged.
