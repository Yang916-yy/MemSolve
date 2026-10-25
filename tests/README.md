# Tests

Run the smallest scope covering the change. Optional integrations need their
matching dependencies; use `-rs` to inspect skipped checks.

| Scope | Command | Coverage |
| --- | --- | --- |
| Core | `python -m pytest tests/core -m "not cuda"` | Reference math, parameters, configuration, public API |
| CUDA | `python -m pytest tests/cuda` | Native math, masks, ranks, gradients, Graphs, sequence parallelism |
| Integrations | `python -m pytest tests/integrations -m "not cuda"` | timm, BERT, OpenMMLab adapters |
| Experiments | `python -m pytest tests/experiments -m "not cuda"` | Data, losses, training, evaluation, resume |
| Repository | `python -m pytest tests/repository` | Package layout and operator ownership |

For CPU-only checks while training is active:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q tests/core tests/repository -m 'not cuda'
python tools/check_repository.py
git diff --check
```

The native extension must be built before CUDA checks. `-m cuda` selects GPU
checks across all scopes, including mixed-precision projections and framework
adapters. Missing dependencies or GPU support are skips, not validated paths.

ImageNet tests use local WebDataset/WIDS fixtures for RA sampling, Mixup/CutMix,
worker reseeding, batching, EMA, checkpoint transfer, and optimizer/scheduler
resume. The NLP tests require `.[nlp]` and exercise fixed evaluation masks,
token-weighted loss, local C4 streaming/sharding, resume, GLUE transfer, and
local embedding evaluation without dataset downloads.

Use the OpenMMLab environment for dense adapter tests. They cover rectangular
CPE, valid-pixel masks, pretrained/EMA loading, task heads, layer LR decay,
initialization synchronization, and BF16/FP32 operator boundaries.

CUDA tests compare against the FP64 reference at supported ranks, including
RoVE, masked/empty samples, tails, and nonzero learned cores. Saved-tensor tests
check repeated backward and scratch-buffer ownership. Exact sequence-parallel
tests compare outputs and gradients with unsharded execution. See the
[CUDA contract](../docs/CUDA_CONTRACT.md) for error budgets and device scope.

Run the full suite for broad changes or releases. Benchmark measurements and
local audit logs belong outside the tracked test tree.
