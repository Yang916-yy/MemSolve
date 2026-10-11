# Research scope after restoring the accretive operator

The current operator restores shared relation/content projection, a soft frame,
a learned shared accretive generator and the scalar content complement.
`equilibrium` is the default. `complement` omits the learned core, retaining
`eta (C - P P^T C)`. Both are input-dependent; neither name implies a fixed
token operator or a zero output. Dynamic has been removed. RankRotary stays absent.
See [the exact equations](CORE_CONTRACT.md).

The visual shell follows the ViT3-style CPE/GELU/mean-pooling model without LayerScale. The intervening independent-Q/K/V, query-delta and output-gate model is
preserved in Git history. Its probes are not evidence of benefits in this model.

Next structural discussion should separate the compact generator, the shared
write/read geometry and the content complement. No new component ablation or
long training run is authorized by this restoration. Exploratory records stay
outside the repository. Historical Base had RankRotary; historical Small did
not, and their training protocols differed.
