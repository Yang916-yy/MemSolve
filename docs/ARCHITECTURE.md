# MemSolve architecture

MemSolve uses the no-RankRotary accretive-equilibrium operator
from `e769a9a`, while retaining current package and model names.
See [the exact equations](CORE_CONTRACT.md).

The operator projects one relation stream and one content stream, constructs a
soft low-rank frame, compresses content, and solves a compact
equilibrium with a learned shared core. It combines a content branch with a compact correction before Wo.
`equilibrium` is the default; `complement` omits the learned compact core.
Both remain input-dependent through the relation frame.

## Vision

MemSolve-ViT uses the restored operator in the ordinary ViT3-style shell:

1. Patch-16 embedding, no CLS token or absolute patch position table.
2. Each block adds a residual 3x3 depthwise CPE to patch tokens.
3. Pre-LayerNorm -> MemSolve -> residual/DropPath.
4. Pre-LayerNorm -> GELU MLP -> residual/DropPath.
5. Per-token final LayerNorm -> mean pooling -> classifier.

Ordinary upstream ViT3 has no LayerScale; this adapter follows that choice.
The source is `vittt/models/vittt.py`, Block.forward and forward_features, at
[upstream commit e347758](https://github.com/LeapLabTHU/ViTTT/blob/e3477587d099e6b9e83e9e7c80b1b999e0989a20/vittt/models/vittt.py).
Masked pooling averages only valid patches. timm's post-pooling fc_norm is
Identity, so normalization happens before the mean, not after it.

Default stochastic depth increases linearly by depth. S retains the historical
MLP width1584 (ratio4.125), width384, 12 blocks, 6 heads and rank32.
With a 1,000-class head it has 19,876,336 parameters. This shell differs from the
historical Small checkpoint by removing CLS and LayerScale; its accuracy is not
inherited from that checkpoint. No RankRotary, RoPE, Q/K short convolution,
readout gate or operator RMSNorm is present. The CPE implementation uses a fixed
configured grid; construct the intended resolution for resolution fine-tuning.

## Text

MemSolve-BERT keeps its existing BERT embeddings, Post-LN residual wrapper,
SwiGLU FFNs and tied MLM head. Only its mixer is restored; this is an integration,
not a claim that the historical ImageNet experiment validated this text model.
