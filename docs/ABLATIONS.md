# Ablations

The public configuration exposes compact-core, skew-coupling, and scalar-complement ablations.

| Ablation | Configuration | Compact-core behavior |
| --- | --- | --- |
| sample-conditioned coordinates | core_mode=DYNAMIC | R = R0 + Z W_drive / sqrt(n_valid) |
| sample-independent coordinates | core_mode=STATIC | R = R0 |
| zero compact core | core_mode=ZERO | M = 0, with no compact parameters |
| remove skew coupling | skew_coupling=False | K = L L^T, with Omega = 0 |
| remove scalar complement | scalar_complement=False | eta = 0, fixed during training |

ZERO changes only the compact core. It retains the relation frame P, content C,
learned complement eta, and the same token lift. The no-skew ablation retains
the lower-triangular accretive factor and removes only Omega. The no-complement
ablation fixes eta to zero and excludes its parameter from gradient updates.
Both settings are explicit in the checkpoint contract.

## Backend and attribution

Native CUDA supports DYNAMIC, STATIC and ZERO. No-skew and no-complement
require the reference implementation; there is no implicit fallback. Rotation
and its configuration switch have been removed from both backends. Keep external
position embeddings, precision, optimizer and data order fixed across core modes.
Historical results retain their original rotated source snapshots.
