# Mechanism studies and experimental scope

The current operator uses shared relation/content projections, an input-dependent
soft frame, a learned shared accretive core and a scalar content complement.
`equilibrium` learns the compact core. `complement` omits it and retains
`eta (C - P P^T C)`. Both remain input-dependent. Dynamic and RankRotary are
absent. See the [exact equations](CORE_CONTRACT.md).

This document proposes experiments; it does not report new results. Historical
LSSO/Ridgon measurements retain their source contracts and cannot substitute for
measurements of the present operator. Generated run records stay outside Git.

## What the paper still needs to demonstrate

ImageNet and MLM/retrieval cover two substantial applications. A third modality
is optional; a direct test of the memory mechanism is more informative than an
unrelated benchmark added solely for breadth. This is a research judgment based
on the claims, not a conference requirement. Hydra evaluates vision, language
and controlled matrix-mixer choices; LION additionally uses LRA to examine
stability. Neither precedent makes another large application mandatory.

| Priority | Question | Proposed evidence | Cost control |
| --- | --- | --- | --- |
| First | Does compact memory resist interference, and where does capacity fail? | Controlled full-context associative recall; accuracy versus independent associations, distractors and key similarity | Small matched encoders, synthetic data, rank 16/32/48 |
| First | Does the implementation realize linear sequence scaling? | Forward/backward latency and peak memory versus sequence length, including short sequences | Existing operator benchmark; use an exclusive GPU window |
| First | What do the shared core and content branch contribute? | Equilibrium/Complement, no skew coupling, no scalar complement | Reuse one small mechanism task and a preselected natural task |
| Optional | Does the same operator transfer beyond vision and language? | Three preselected DNA classification tasks under a matched shell | Existing loaders, three seeds; no new genome-scale pretraining |
| Later | Can the complete encoder use substantially longer context? | Longer-context training plus matched full-context retrieval evaluation | Requires a position/continuation decision; 512-truncated LongEmbed is insufficient |

## Controlled memory experiment

Use randomly generated associations and separate query locations. Replace every
answer slot at a query with a mask; retain target values only in the intended
memory records. Randomize record/query positions and values so a position-only
or memorized-label shortcut cannot solve the task. Keep the same positional
encoding, embeddings, depth, FFN and training budget across mixers; report actual
parameter counts rather than assuming they match.

Start with rank 16/32/48, distinct memory records, and a fixed evaluation generator.
Vary one factor at a time: independent association count, irrelevant records,
and similarity between distinct keys. Report accuracy against association
count/rank as well as sequence length. Held-out generated sequences, a fixed
step budget and at least three seeds for final comparisons separate optimization
variance from memory capacity. Start with one seed to validate the task before
spending the full comparison budget.

Zoology's MQAR is a **causal** benchmark. Feeding its original autoregressive
input to a bidirectional encoder can expose the answer tokens. A masked
full-context adaptation must have a separate name and matched-access baselines;
its score is not the original MQAR score. Compare full-context MHA and a
bidirectional linear-attention baseline. A causal DeltaNet baseline would need
an explicitly matched information-access protocol before supporting a mechanism
comparison. Recent work such as *Memory by Design* also separates controlled
collision/recall tests from downstream language experiments; its sequential
memory mechanism is different from MemSolve's global equilibrium.

For component studies, `MemSolveConfig` already exposes `core_mode`,
`skew_coupling` and `scalar_complement`. Use the reference backend for unsupported
ablation configurations; do not silently time a fallback against the native
default. The no-core comparison is not a parameter-matched perturbation, and
removing the content branch changes the output family. Record those differences.

## Numerical and systems evidence

Use `benchmarks/benchmark_ball.py` at lengths 128, 512, 2048, 8192 and 16384,
subject to device memory, with fixed width/heads and rank 16/32/48. Separate
forward from forward-plus-backward, and eager from CUDA Graph timing. Keep
physical batch, activation precision, gradient requirements and synchronization
consistent. Report OOMs and native/reference boundaries, never a smaller batch
as if it were a same-workload speedup. Measure peak allocated memory and state
the hardware/software stack. Any active image training job should finish or
yield an explicit timing window before collecting these numbers.

These are operator scaling measurements, not evidence that the 512-position
BERT encoder understands 16K-token documents. Computational scaling and memory
capacity are distinct claims. `experiments/certificate_diagnostics.py` can check
compact residuals and contraction diagnostics. Compare direct solves with a
specified finite-step method on the same system and at matched residual/output
error; a poorly tuned iteration is not a meaningful efficiency baseline.
The head-map norm certificate does not bound arbitrary learned surrounding
projections or the complete residual network, nor does it prove generalization.

## Optional third modality

If the vision/language results and mechanism studies are complete, use Human or
Worm, Human OCR Ensembl, and Mouse Enhancers Ensembl as a small predeclared panel.
This includes a historical difficult task as well as tasks with earlier gains.
Use the existing GenomicBenchmarks official partitions, a train-only validation
split, validation-based checkpoint selection and one final held-out test.
Hold external position encoding and the encoder shell fixed for MHA, Equilibrium
and Complement. Use three seeds for reported comparisons; do not reuse historical
RankRotary/Dynamic runs as present-model results.

This panel tests transfer to biological sequences, not million-base context or
genome-wide generalization. The official short-sequence benchmark is documented
in the [sequence guide](SEQUENCE_EXPERIMENTS.md). LRA is an alternative if the
paper prioritizes long-sequence behavior, but restarting the full suite before
MLM and mechanism tests is lower priority. Detection, segmentation, Assembly101
and time-series training are not part of this plan.

## Language evaluation scope and sources

Retain the full LION-derived C4/128/global-4096/70,000-step recipe and measure
runtime before launch. Evaluate fixed-mask MLM NLL, then use the same retrieval
fine-tuning and length policy for comparison models. Start with the English v2
MTEB retrieval subset; label it as a subset and run the full suite for a full
MTEB score. Modern encoder evaluation supports this direction: Ettin reports
MTEB v2, MLDR and CodeSearchNet alongside GLUE. It does not make LongEmbed a
substitute for MLDR, and its much larger pretraining budget is not matched by
our recipe. No GLUE run is added by default. The [NLP guide](BERT_MLM.md) lists
the implemented launchers and the remaining native-checkpoint/long-context work.

Primary sources inspected for this plan:

- [Hydra, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c7f795dc3b4eb6ae630695d90001a2f8-Paper-Conference.pdf): controlled mixer studies plus language and vision.
- [LION, version 2](https://arxiv.org/html/2502.16249v2): vision/MLM, efficiency and LRA stability studies.
- [Ettin / Seq vs Seq, version 2](https://arxiv.org/html/2507.11412v2): modern encoder evaluation and budget-aware comparison.
- [Zoology, version 1](https://arxiv.org/html/2312.04927v1): causal multi-query associative recall and its information-access assumptions.
- [Memory by Design, 2026 preprint](https://arxiv.org/html/2605.31163v1): controlled collision/recall tests alongside broader model evaluation.
- [GenomicBenchmarks official implementation](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks): dataset definitions and partitions.
