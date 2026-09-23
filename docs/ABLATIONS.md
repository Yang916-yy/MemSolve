# Research scope after the Q/K/V refactor

The current operator is independent Q/K/V, a learned sample-independent
identity-plus-delta query map T, ridge memory readout and per-head RMS normalization.
It exposes no Dynamic/Static/Zero, skew or complement switches.

The input-conditioned core generator and projection-local Q/K/V convolutions
are intentionally absent. Their mathematical role and empirical benefit can
be studied after this base structure is established. V names the value tensor;
T names the shared query map, so K remains unambiguously the key tensor.

The former Static and Zero experiments describe a different reflected-readout
architecture. Current T initializes to I, giving O=Pq Z. Existing results must not be
relabeled as measurements of the current model.

Future architectural interventions require a stated formula, matched training
budget and an updated mathematical/gradient oracle. Exploratory experiment
records stay outside this repository.
