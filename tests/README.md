# Test Suite

The test suite covers the mathematical reference, native CUDA/Triton kernels, framework integrations (`timm`, Hugging Face Transformers, OpenMMLab), and training pipelines.

## Running Tests

Run `pytest` directly from the repository root:

```bash
# Run all CPU tests (fast, no GPU required)
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q -m 'not cuda'

# Run the full suite including GPU tests (requires built CUDA extension)
python -m pytest -q

# Run a specific test module or keyword filter
python -m pytest tests/cuda -k 'inverse_factor' -q
```

## Organization

| Directory | Command | What It Tests |
| --- | --- | --- |
| `tests/core/` | `python -m pytest tests/core -m "not cuda"` | Reference operator math, spectral/contraction properties, RoVE, module config & API |
| `tests/cuda/` | `python -m pytest tests/cuda` | CUDA/Triton forward & backward parity vs. FP64 oracle, masks, ranks 16/32/48/64, CUDA Graphs, exact sequence parallelism, register spills |
| `tests/integrations/` | `python -m pytest tests/integrations` | `timm` ViT backbones, Hugging Face MemSolve-BERT, OpenMMLab ViT-Adapter & detection/segmentation wrappers |
| `tests/experiments/` | `python -m pytest tests/experiments` | ImageNet WebDataset/WIDS loader, C4 preparation & MLM/GLUE pipelines, LRA data loaders & 10-mixer attention adapters |

See [`docs/CUDA_CONTRACT.md`](../docs/CUDA_CONTRACT.md) for details on the numerical tolerances verified against the FP64 reference implementation.
