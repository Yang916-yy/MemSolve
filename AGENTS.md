# Guide for AI-assisted reproduction and contributions

Use this guide when working with MemSolve through a coding assistant. Start with
[README.md](README.md), then read the experiment or integration guide relevant to
your task. The paper describes the method; the published logs in
[results/](results/README.md) connect reported results to recorded runs.

## Reproducing experiments

- Start from a published recipe in `experiments/configs/`. Check its data,
  model, precision, effective batch size, and evaluation settings before running.
- Record configuration overrides and software versions. Changing a recipe is a
  new experiment; give it a separate output directory.
- Match weights to their model configuration. Distinguish ordinary and EMA
  weights, checkpoint selection, and validation metrics when comparing results.
- Collect tests and run a small smoke check before launching a long experiment.
  Keep downloads, checkpoints, and working outputs outside the checkout.
- Publish useful structured logs with their configurations and provenance.
  Preserve the complete recorded curve, including unsuccessful epochs, and
  distinguish measurements from derived summaries. Remove credentials and
  machine-specific paths before sharing artifacts.

## Finding the right code

| Area | Starting point |
| --- | --- |
| Operator mathematics | `memsolve/ball/reference.py` |
| Parameters and module behavior | `memsolve/ball/model.py` |
| Public configuration | `memsolve/ball/config.py` |
| CUDA implementation and numerical checks | `docs/CUDA_CONTRACT.md` |
| Training and evaluation | `experiments/` and its linked guides |
| Verification | `tests/README.md` |

Keep operator mathematics in the core. Framework adapters should translate
interfaces, and experiment runners should assemble models, data, and training.
Reuse existing utilities rather than adding parallel implementations.

## Making a contribution

Keep patches focused and explain the problem they solve. Prefer clear code and
comments that explain a non-obvious operation or design choice. Avoid repeating
what the code already says or preserving a history of edits in comments.

For a mathematical change, explain the equation being changed. For an equivalent
optimization, compare outputs and gradients with the reference implementation.
Measure performance on the intended shapes and hardware, including memory use;
report the precision and execution mode alongside timings. Consult established
upstream implementations when useful, and retain their attribution and licenses.

Use the tests covering the affected code; see `tests/README.md` for commands and
optional dependencies. For package or layout changes, run:

```bash
python -m pytest tests/core/test_model.py -k repository_contract
```

A contribution summary should say what changed, which checks ran, and what
remains unverified. Skipped tests are not passing tests. Check AI-generated code,
measurements, and references against their sources before submitting them.

Keep reusable code, current documentation, and published experimental evidence
in Git. Keep temporary probes, launch queues, plotting scripts, and personal
working notes outside it.
