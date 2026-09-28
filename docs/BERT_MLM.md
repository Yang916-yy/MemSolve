# MemSolve-BERT: MLM Pretraining & GLUE

The Hugging Face integration replaces BERT's self-attention layer with MemSolve and swaps the standard FFN for SwiGLU, while keeping BERT embeddings, Post-LayerNorm residuals, dropout, and the tied MLM prediction head. Importing `integrations.transformers` registers `memsolve_bert` with `AutoConfig`, `AutoModel`, `AutoModelForMaskedLM`, and `AutoModelForSequenceClassification`.

## Architecture & Configuration

| Setting | Value |
| --- | --- |
| Depth / width / heads / rank | 14 / 768 / 12 / 32 |
| SwiGLU intermediate size | 2048 per branch (packed gate/up width 4096) |
| Local context branch | Centered residual depthwise `Conv1d` (kernel size 3) |
| Vocabulary / token types | 30,522 / 2 |
| Dropout / LayerNorm $\epsilon$ | 0.1 / $10^{-12}$ |
| Default position encoding | RoVE (base 10,000, YaRN frequency scaling beyond 128) |
| Parameters (RoVE, tied MLM head) | 111,123,938 |
| Parameters (learned absolute positions, tied MLM head) | 111,517,154 |

- **Position Encoding**: The paper's MLM and GLUE experiments use **learned absolute position embeddings** (max 512 positions). Their logs and configs are in [`results/mlm/`](../results/mlm/) and [`results/glue/`](../results/glue/). The default template in `experiments/configs/memsolve_bert_mlm.json` uses **RoVE**; to match the paper's absolute-position model, set `"position_embedding_type": "absolute"` in the config.
- **Short Convolution**: Set `memsolve_short_conv_kernel=0` to disable the depthwise convolution. When enabled (`kernel=3`), padded tokens are zeroed before and after the convolution, and its weights are zero-initialized so the branch starts as an exact identity.
- **Initialization**: Projection weights and BERT layers use $\mathcal{N}(0, 0.02^2)$ with zero bias; the MemSolve core starts at $J = I$ ($F = 0$) and $\eta = 0.9$.

```python
from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM

model = MemSolveBertForMaskedLM(MemSolveBertConfig(
    position_embedding_type="absolute",
    memsolve_implementation="cuda",
)).cuda()
```

A standard binary `[B, N]` `attention_mask` marks valid tokens (`1`) vs. padding (`0`). Padded tokens are excluded from the compact statistics and zeroed in the output.

## Preparing C4 Data

Install `.[nlp]` first, then prepare the training and validation splits:

```bash
python -m experiments.prepare_c4 --output /datasets/c4-128 --split train
python -m experiments.prepare_c4 --output /datasets/c4-128 --split validation
```

Documents are truncated to 128 tokens and padded individually (no multi-document packing). Validation masks are fixed once during data preparation (15% corruption), while training masks (30% corruption) are sampled dynamically by the collator using the 80/10/10 rule.

If you already have raw C4 JSONL shards on disk, you can also stream directly from a local manifest (`memsolve-local-text-stream-v1`) via `--streaming-train-manifest`, which only requires the prepared tokenizer and frozen validation split in `--data`.

## MLM Pretraining

Our pretraining budget matches the [LION M2-BERT setup](https://arxiv.org/html/2502.16249v2):

| Hyperparameter | Value |
| --- | --- |
| Sequence length / global batch / steps | 128 / 4096 / 70,000 |
| Per-GPU train / eval batch size | 512 / 128 |
| Optimizer | Fused AdamW, $\beta = (0.9, 0.98)$, $\epsilon = 10^{-6}$ |
| Peak / final LR | 8e-4 / 1.6e-5 (linear decay after 4,200 warmup steps) |
| Weight decay / gradient clip | 0.0125 / 1.0 |
| Train / validation mask ratio | 0.30 / 0.15 |
| Eval & checkpoint interval | Every 1,000 steps |
| Precision & execution | FP32 weights, BF16 autocast, `compile-graph` |

*Note on weight decay*: Composer's decoupled weight decay (`1e-5`) translates to PyTorch `AdamW` weight decay as `1e-5 / 8e-4 = 0.0125` (excluding norms and biases).

```bash
# Dry-run preflight check
python -m experiments.train_mlm \
  --config experiments/configs/memsolve_bert_mlm.json \
  --data /datasets/c4-128 --check-only --world-size 2

# Launch pretraining on 2 GPUs (4 gradient accumulation steps per update)
torchrun --standalone --nproc_per_node=2 -m experiments.train_mlm \
  --config experiments/configs/memsolve_bert_mlm.json \
  --data /datasets/c4-128 --output /runs/memsolve-mlm
```

Loss and gradients are normalized by the total number of masked target tokens across all ranks and microbatches. At the end of training, the checkpoint with the lowest token-weighted validation `eval_masked_nll` is exported to `final/`. Use `--resume /runs/memsolve-mlm/checkpoint-N` to resume an interrupted run.

### Standalone Evaluation & Length Continuation

```bash
# Evaluate a saved checkpoint on the frozen validation split
python -m experiments.train_mlm \
  --config /configs/matching-mlm.json --data /datasets/c4-128 \
  --evaluate-checkpoint /runs/memsolve-mlm/final --output /runs/mlm-eval

# Continue pretraining at length 512 with a fresh optimizer/schedule
torchrun --standalone --nproc_per_node=2 -m experiments.train_mlm \
  --config /configs/continued-mlm.json --data /datasets/c4-512 \
  --init-checkpoint /runs/memsolve-mlm/final --output /runs/mlm-512
```

## GLUE Fine-Tuning

`experiments/configs/memsolve_bert_glue.json` implements the LION M2-BASE fine-tuning protocol (sequence length 128, 6% linear warmup, linear decay to zero, evaluated at the final checkpoint). Following standard practice, RTE, MRPC, and STS-B initialize from the fine-tuned MNLI encoder, while MNLI, QQP, QNLI, SST-2, and CoLA initialize directly from the pretrained MLM checkpoint.

```bash
# Tokenize and cache GLUE splits
python -m experiments.train_glue --prepare-only \
  --checkpoint /runs/memsolve-mlm/final --data /datasets/glue-128

# Fine-tune MNLI first, then transfer to RTE / MRPC / STS-B
python -m experiments.train_glue --task mnli \
  --checkpoint /runs/memsolve-mlm/final --data /datasets/glue-128 \
  --output /runs/glue/mnli

python -m experiments.train_glue --task rte \
  --checkpoint /runs/glue/mnli/final --data /datasets/glue-128 \
  --output /runs/glue/rte
```

The paper's aggregate GLUE score averages the 8 task scores (using matched/mismatched mean for MNLI, accuracy/F1 mean for QQP, F1 for MRPC, Spearman correlation for STS-B, Matthews correlation for CoLA, and accuracy for QNLI and SST-2).
