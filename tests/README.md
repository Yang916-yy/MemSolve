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

# Run fullgraph torch.compile, Hopper SM90a PTX/cubin, precision & lifecycle checks
python -m pytest tests/cuda/test_compile_hopper_precision.py -q
```

## Organization

| Directory | Command | What It Tests |
| --- | --- | --- |
| `tests/core/` | `python -m pytest tests/core -m "not cuda"` | Reference operator math, spectral/contraction properties, RoVE, `recompute_readout` config & module API |
| `tests/cuda/` | `python -m pytest tests/cuda` | `test_fast_path.py` and `test_compile_hopper_precision.py`: CUDA/Triton forward & backward parity vs. FP64 oracle (incl. $N \le r$ and `recompute_readout`), ranks 16/32/48/64, fullgraph `torch.compile` (`SymInt` fake kernels, zero graph breaks) & `CUDAGraph`, `SM80`/`SM90a` Hopper TMA/WGMMA/PDL PTX & zero register spills, exact sequence parallelism, and integration lifecycles |
| `tests/integrations/` | `python -m pytest tests/integrations` | `timm` ViT backbones, Hugging Face MemSolve-BERT, OpenMMLab ViT-Adapter & detection/segmentation wrappers |
| `tests/experiments/` | `python -m pytest tests/experiments` | ImageNet WebDataset/WIDS loader, C4 preparation & MLM/GLUE pipelines, LRA data loaders & 10-mixer attention adapters |

See [`docs/CUDA_CONTRACT.md`](../docs/CUDA_CONTRACT.md) for details on the numerical tolerances verified against the FP64 reference implementation.
