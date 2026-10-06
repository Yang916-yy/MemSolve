# Research scope after the Q/K/V refactor

The current operator is independent Q/K/V, a learned sample-independent
identity-plus-delta query map T, centered depthwise Q/K convolution, ridge memory
readout and per-head RMS normalization.
It exposes no Dynamic/Static/Zero, skew or complement switches.

The input-conditioned core generator, V convolution and output gating remain
absent. Q/K use width-3 sequence filters or 3×3 image filters with identity
initialization. Vision additionally uses fixed axial 2D RoPE after Q/K convolution,
replacing residual CPE. Short distillation probes motivate studying the local prior;
they do not establish gains from full training. V names the value tensor;
T names the shared query map, so K remains unambiguously the key tensor.

The former Static and Zero experiments describe a different reflected-readout
architecture. Current T initializes to I, giving O=Pq Z. Existing results must not be
relabeled as measurements of the current model.

Future architectural interventions require a stated formula, matched training
budget and an updated mathematical/gradient oracle. Exploratory experiment
records stay outside this repository.
