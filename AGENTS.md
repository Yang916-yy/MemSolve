# AI-Assisted Contribution Contract

This file defines the repository contract for contributors who use coding
agents, large language models, or other AI-assisted development tools. It
governs proposed changes and review evidence; it does not affect installation,
runtime behavior, or ordinary use of LSSO.

The human contributor remains responsible for the scope, correctness,
provenance, testing, and reviewability of every submitted change. AI-generated
code, documentation, experiments, and citations must satisfy the same standards
as human-authored work. Material use of external research or upstream code must
be identified in the change summary with the source and the decision it
informed.

## Design Principles

- This repository has one current operator contract. Do not preserve legacy APIs,
  checkpoints, imports, variants, or behavior unless the change request
  explicitly requires it.
- Prefer the simplest implementation that is mathematically equivalent to the
  contract and measurably better for the intended workload.
- "Stable" means finite, correct, and trainable throughout the normal training
  envelope: supported dtype, rank, sequence length, optimizer settings, and
  realistically reachable parameter scales.
- Do not add persistent FP64 paths, clipping, fallback branches, thresholds, or
  extra configuration solely for astronomically pathological inputs that normal
  training cannot reach. State the residual boundary honestly instead.

## Core Ownership

- `lsso/ball/reference.py` is the only owner of operator mathematics and the
  canonical numerical contract.
- `lsso/ball/model.py` is the only owner of parameters and `nn.Module` behavior.
- `lsso/ball/config.py` is the only owner of public variants and validation.
- Do not add compatibility aliases, legacy imports, duplicate model classes, or
  a second reference implementation.
- Do not add files under `lsso/ball/` without maintainer approval.

## Equivalent Implementations

- A replacement implementation must state the algebraic equivalence to the
  reference contract before it is adopted.
- Choose the lowest-overhead implementation that is correct in the normal
  training envelope. Engineering form may differ from paper notation.
- Validate every replacement with forward and gradient comparisons against
  `reference.py`, plus a benchmark for the intended workload.
- Numerical guards must be fixed implementation details, never hidden model
  variants. Add one only when normal-range evidence justifies its cost.

## Investigation Before Invention

- First reproduce and localize a failure. For non-local questions involving
  numerical linear algebra, optimization, CUDA, PyTorch behavior, or known
  architectures, search prior work before inventing a new mechanism.
- Prefer primary papers, official framework documentation, upstream source,
  issue trackers, and established implementations.
- Record the relevant source and the decision it changed in the change summary
  when external research materially affects the design.
- Do not use web search as a substitute for a local reproducer or tests.
- Simple local bugs do not require a research detour.

## CUDA Boundary

- CUDA supports DYNAMIC, STATIC and ZERO with default skew and complement
  settings at ranks 16, 32, 48 and 64; see `docs/CUDA_CONTRACT.md`.
- Unsupported variants, layouts, dtypes, or devices must fail explicitly.
- CUDA must not silently fall back or preserve an old ABI.
- Every CUDA forward result and gradient must be checked against `reference.py`.

## Peripheral Code

- Integrations adapt framework interfaces only; they must not contain model math.
- Experiment entrypoints stay thin and share training utilities.
- Benchmarks measure implementations; they never define model defaults.
- Temporary scripts and generated artifacts belong outside the repository.

## Change Discipline

- One change scope per commit: core, CUDA, integration, experiment, or docs.
- Reuse an existing file before proposing a new one.
- `tests/` contains only deterministic, user-facing contract tests. Keep local
  probes, performance exploration, downloaded fixtures, and generated outputs
  out of Git.
- Run the narrow test domain for the boundary changed: `tests/core`,
  `tests/cuda`, `tests/integrations`, `tests/experiments`, or
  `tests/repository`. Use `-m "not cuda"` when a CPU-only check is sufficient.
- Core math changes require the relevant `tests/core` checks and CUDA oracle
  checks when the native path is affected. CUDA changes require `tests/cuda`
  and the corresponding reference comparison. Integration and experiment
  changes require their own domain.
- Run `python tools/check_repository.py` after package or repository-layout
  changes. Run `pytest -q` only before a release, a broad refactor, or a change
  crossing multiple domains.
- Report residual limitations plainly instead of adding speculative machinery.
