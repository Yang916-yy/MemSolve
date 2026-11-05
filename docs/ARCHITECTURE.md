# Model Architecture

MemSolve replaces standard pairwise token mixing with a shared-frame accretive-equilibrium operator (see [CORE_CONTRACT.md](CORE_CONTRACT.md) for the operator equations). Each layer projects the input into a low-rank relation stream and a content stream, builds a soft frame from the relations, solves a compact rank-$r$ equilibrium system with a learned per-head generator, and combines the equilibrium response with a scaled residual content path before the output projection.

Two core modes are supported:
- **`equilibrium`** (default): learns a per-head accretive generator $J_h$ and solves $(I + J_h)^{-1}(I - J_h)$ in rank space.
- **`complement`**: omits the learned generator ($J_h = I$, $F_h = 0$) while keeping the input-dependent soft frame and content compensation.

## Vision (`MemSolve-ViT`)

`MemSolve-ViT` follows the 12-layer isotropic [ViT³](https://github.com/LeapLabTHU/ViTTT/blob/e3477587d099e6b9e83e9e7c80b1b999e0989a20/vittt/models/vittt.py) backbone layout:

1. **Patch embedding**: $16 \times 16$ patch projection without a `[CLS]` token or fixed 2D position table.
2. **Local positional encoding**: a residual $3 \times 3$ depthwise convolution (CPE) at the start of every block.
3. **Token mixer**: `LayerNorm -> MemSolve -> DropPath + residual`.
4. **Channel MLP**: `LayerNorm -> GELU FFN -> DropPath + residual`.
5. **Classifier head**: per-token `LayerNorm` followed by masked mean pooling over valid patches and a linear classifier.

| Model | Width | Depth | Heads | Rank | MLP Width (Ratio) | DropPath | Parameters (1K classes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MemSolve-ViT-T` | 192 | 12 | 6 | 16 | 768 (4.0) | 0.0 | 5,053,936 |
| `MemSolve-ViT-S` | 384 | 12 | 6 | 32 | 1584 (4.125) | 0.1 | 19,876,336 |
| `MemSolve-ViT-B` | 768 | 12 | 12 | 48 | 3072 (4.0) | 0.4 | 77,980,792 |

For detection and segmentation, the pretrained backbone connects to a standard `ViT-Adapter` via `VisionTokenLayout` to handle arbitrary rectangular feature grids (see [DOWNSTREAM_PROTOCOLS.md](DOWNSTREAM_PROTOCOLS.md)).

## Language (`MemSolve-BERT`)

`MemSolve-BERT` uses a 14-layer bidirectional encoder matched in parameter count to 12-layer `BERT-base`:

- **Backbone**: width 768, 14 layers, 12 heads, rank 32, Post-LayerNorm residuals, SwiGLU feed-forward networks (branch width 2048), and a tied masked-LM output head.
- **Local context**: a centered $3 \times 1$ depthwise residual convolution before each MemSolve layer (zero-initialized so each block starts as an identity residual). Padding tokens are masked before and after the convolution.
- **Position encoding**: supports both learned absolute position embeddings (`111,517,154` parameters with 512 positions) and parameter-free Rotary Value Encoding (`RoVE`, `111,123,938` parameters); both variants are evaluated in the paper's C4 MLM and GLUE experiments. See [BERT_MLM.md](BERT_MLM.md) for training and fine-tuning details.
