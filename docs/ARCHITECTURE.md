# MemSolve architecture

MemSolve uses a shared-frame accretive-equilibrium operator.
See [the exact equations](CORE_CONTRACT.md).

The operator projects one relation stream and one content stream, constructs a
soft low-rank frame, compresses content, and solves a compact
equilibrium with a learned shared core. It combines a content branch with a compact correction before Wo.
`equilibrium` is the default; `complement` omits the learned compact core.
Both remain input-dependent through the relation frame.

## Vision

MemSolve-ViT uses this operator in the ordinary ViT3-style shell:

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

Default stochastic depth increases linearly by depth. S uses
MLP width 1584 (ratio 4.125), width 384, 12 blocks, 6 heads and rank 32.
With a 1,000-class head it has 19,876,336 parameters.
Classification uses its configured patch grid; dense prediction supplies an
explicit rectangular grid through `VisionTokenLayout`. ViT-Adapter adds spatial
features around the same pretrained blocks; see the
[dense prediction protocol](DOWNSTREAM_PROTOCOLS.md).

## Text

MemSolve-BERT keeps its existing BERT embeddings, Post-LN residual wrapper,
SwiGLU FFNs and tied MLM head. Each mixer receives a centered residual depthwise
convolution (kernel 3), with padding masked before and after it. The default
14-layer, width-768, rank-32 model uses full-head RoVE content transport and has
111,123,938 parameters including its tied MLM head, without a learned position
table. It uses base 10000 and frequency-only YaRN beyond its 128-token training
window, with unit amplitude. This is approximately parameter-matched to
12-layer BERT-base; it is not a depth-controlled operator comparison.
The [NLP guide](BERT_MLM.md) separates training, resume, continued pretraining,
checkpoint selection and evaluation. Completed MLM/GLUE experiments used the
explicit absolute-position configuration (111,517,154 parameters, 512 learned
positions). RoVE is a separate experiment setting.
