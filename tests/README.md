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
fixed MLM masks, token-weighted loss, C4 partition integrity, MLM/retrieval
Trainer resume and a local MTEB retrieval task without downloading datasets.

ImageNet data checks use WebDataset and WIDS from the `vision` or `test` extra.
They cover global virtual-group RA, independent group mixing with one physical
forward, persistent-worker epoch changes and restart replay using local tar fixtures.

Use `python -m pytest -m cuda` to select all CUDA-marked checks across scopes.
Run `python -m pytest -q` only for a release, a broad refactor, or a change
that crosses multiple scopes.

## Mixed-precision and optional checks

CUDA math changes also require the relevant `tests/core` checks: projection
fusion, FP32 bias cancellation, strided tails and gradient comparisons live
there. Some CUDA-dependent core tests use availability guards rather than the
`cuda` marker, so `-m cuda` alone is not the entire GPU numerical test domain.

```bash
python -m pytest -q tests/core tests/cuda
python tools/check_repository.py
git diff --check
```

Read skip reasons with `-rs`. Missing timm/OpenMMLab packages  must not be reported as successful full-model validation. Test other
SM targets on their own hardware. Benchmarks and exploratory probes belong
outside the tracked test tree.

Before the 2026-09-12 main publication, the full suite passed 342 tests with
6 skips; only SM120 was executed. This is a dated verification record, not a
promise that all optional stacks or architectures were exercised.

Model contract 24 / CUDA contract 19 retains the original single-valid-token
FP64 oracle tolerance. Its former near-zero core-gradient failure is fixed by
residual-compensated readout and coefficient-adjoint products. The suite also
checks fused sigmoid gates, native centered convolution with fused packing,
RoPE/masks, all parameter gradients and CUDA Graph replay. The FP16 readout/Wo
forward path is compared with an FP64 oracle; upstream scales down to 1e-9
check that backward does not inherit a half-precision gradient edge. See the
[CUDA contract](../docs/CUDA_CONTRACT.md) for precision boundaries.
