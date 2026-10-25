# MemSolve-BERT

The Hugging Face adapter replaces BERT's attention mixer with MemSolve and
uses SwiGLU FFNs. Embeddings, Post-LayerNorm residuals, dropout, and the tied
MLM prediction head follow BERT. Import `integrations.transformers` to register
`memsolve_bert` with the AutoConfig/AutoModel classes.

## Model configuration

| Setting | Value |
| --- | --- |
| Depth / width / heads / rank | 14 / 768 / 12 / 32 |
| SwiGLU width | 2048 per branch; packed gate/up width 4096 |
| Local context | Centered residual depthwise Conv1d, kernel 3 |
| Vocabulary / token types | 30522 / 2 |
| Dropout / LayerNorm epsilon | 0.1 / 1e-12 |
| Supplied position setting | RoVE, base 10000, frequency-only YaRN beyond 128 |
| Parameters, RoVE with tied MLM head | 111,123,938 |
| Parameters, absolute positions with tied MLM head | 111,517,154 |

The paper's MLM/GLUE results use **learned absolute positions**, with 512
position embeddings. The supplied `memsolve_bert_mlm.json` selects **RoVE**.
To reproduce the paper's architecture, copy this JSON outside the checkout and
set `model.position_embedding_type` to `"absolute"`. Keep config and weights
together; position mechanisms cannot be swapped when loading a checkpoint.

`memsolve_short_conv_kernel=0` disables the convolution. With kernel 3, padding
is zeroed before and after the residual convolution. Zero-initialized weights
make this local transformation an identity at initialization. It sees the
corrupted MLM input, never the original masked labels.

BERT scaffold and B/C/output projections use normal initialization with standard
deviation 0.02 and zero biases. The core initializes at J=I (F=0), and eta starts
at 0.9. The MLM decoder shares the token embedding weight. The base MLM model
has no trained CLS pooler. SwiGLU uses `silu(gate) * up`; `hidden_act` controls
the MLM transform, not the FFN activation.

```python
from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM

model = MemSolveBertForMaskedLM(MemSolveBertConfig(
    position_embedding_type="absolute",
    memsolve_implementation="cuda",
)).cuda()
```

RoVE settings, position IDs, and per-sample context lengths follow the
[core contract](CORE_CONTRACT.md#optional-rotary-value-encoding). The BERT wrapper
shares rotation tables across layers within a forward pass. Set
`position_embedding_type="none"` for no position table or rotation.

## Masks and checkpoints

A binary `[batch, length]` attention mask selects valid tokens. Pairwise masks,
causal decoding, cross attention, KV caches, and attention-probability outputs
are unsupported. Padding is excluded from the memory statistics and output;
mean pooling must also use the mask. Independent documents must not be packed
into one unsegmented attention mask.

Use `save_pretrained`/`from_pretrained`. Safetensors store weights, while
`config.json` stores the operator contract and geometry. Stock BERT and
LION/Hydra checkpoints require their own loaders or an explicit conversion;
they cannot be loaded as MemSolve weights.

## C4 preparation

Install `.[nlp]` in the intended training environment. Run from the source
checkout and keep data and outputs outside it.

```bash
python -m experiments.prepare_c4 --output /datasets/c4-128 --split train
python -m experiments.prepare_c4 --output /datasets/c4-128 --split validation
```

Use `--revision` and `--tokenizer-revision` to pin Hub commits. Training documents
are truncated individually to 128 tokens and padded, without packing. Validation
has fixed corruption; training corruption is sampled by the MLM collator.
Optional `--document-policy split` keeps nonoverlapping windows and defines a
different data protocol. Use the same policy for both splits and a new output.

Preparation can partition a split with `--num-shards` and `--shard-index`.
Use one partition scheme per output; training requires every declared partition.
`--max-docs` limits each partition, rather than sampling uniformly from all C4.
Local JSONL inputs are supported through `--files`. Manifests retain dataset
revision, tokenizer identity, lengths, target counts, and split fingerprints.

For full raw C4 already on disk, train directly from a local streaming manifest.
`--data` then needs only the prepared tokenizer and frozen validation split.
The JSON manifest has format `memsolve-local-text-stream-v1`, `num_source_files`,
`tokenizer_hash`, `max_length`, `packing: "one_document"`, and `files` entries
with absolute `path`, `size`, and `mtime_ns`. Preserve the source revision and
checksums alongside it. File identity and validation compatibility are checked
at startup; the full corpus is not rehashed on each launch.

Accelerate partitions files across ranks; Datasets partitions each rank's files
across workers. Use `dispatch_batches=false` and `split_batches=false`, with at
least one source file per rank/worker. Do not shard twice. The supplied recipe
uses a 10,000-document shuffle buffer per worker and a fixed step budget.

Network settings come from the process environment. For direct downloads, unset
proxy variables only for that command; credentials are read from Hugging Face's
normal authentication store, not written into configs.

## MLM pretraining

The budget follows [LION's M2 setup](https://arxiv.org/html/2502.16249v2).

| Setting | Value |
| --- | --- |
| Sequence length / global batch / updates | 128 / 4096 / 70,000 |
| Per-GPU train / evaluation batch | 512 / 128 |
| Optimizer | Fused AdamW, betas (0.9, 0.98), epsilon 1e-6 |
| Peak / final LR | 8e-4 / 1.6e-5 |
| Warmup / decay | 4200 steps / linear |
| Weight decay / gradient clip | 0.0125 / 1.0 |
| Train / validation masking | 0.30 / 0.15 |
| Evaluation / save interval | 1000 steps |
| Execution | BF16 autocast, FP32 parameters, compile-graph |

The final LR follows the public code's 2% endpoint. Composer's decay convention
is converted to PyTorch AdamW as `1e-5 / 8e-4 = 0.0125`; norm and bias parameters
are excluded. If peak LR changes, recompute that ratio to preserve the decay.
MLM corruption uses the standard 80/10/10 mask/random/unchanged rule.

```bash
python -m experiments.train_mlm \
  --config experiments/configs/memsolve_bert_mlm.json \
  --data /datasets/c4-128 --check-only --world-size 2

torchrun --standalone --nproc_per_node=2 -m experiments.train_mlm \
  --config experiments/configs/memsolve_bert_mlm.json \
  --data /datasets/c4-128 --output /runs/memsolve-mlm
```

Add `--streaming-train-manifest /datasets/c4/train-stream.json` for local raw-C4
streaming. Effective batch determines accumulation: two GPUs at 512 each use
four microbatches. Adjust microbatch and workers for the actual machine while
retaining the effective batch.

`compile-graph` compiles the BERT scaffold and MLM loss, keeps MemSolve outside
Dynamo, and captures tensor forward/backward before DDP wrapping. Trainer owns
communication, accumulation, clipping, optimization, scheduling, and saving.
Capture restores RNG state and buffers without updating parameters. Dropout RNG
advances during replay. Graph training requires full, fixed-shape microbatches;
a partial training batch raises an error. Use a sufficient streaming source or
an explicit training drop policy. Validation always keeps the partial final
batch and uses ordinary forward execution. `graph` omits compilation; `eager`
uses the standard Trainer path.

Loss and gradients are normalized by the masked-target count across ranks and
accumulation steps. Evaluation reports token-weighted `eval_masked_nll`,
accuracy, perplexity, and target counts without gathering vocabulary logits.
Use masked NLL for selection; Trainer's per-batch `eval_loss` can differ.

`run.json` records config, data identity, and source versions. Resume from
`checkpoint-N` with the same output, source, recipe, and world size using
`--resume`. The lowest validation masked-NLL checkpoint is exported to `final/`,
including a selection record and tokenizer. `final/` is not an optimizer snapshot.
Streaming resume replays/skips consumed batches and can be CPU-intensive;
persistent-worker training corruption is not guaranteed bitwise replay.

## Independent evaluation and continuation

```bash
python -m experiments.train_mlm \
  --config /configs/matching-mlm.json --data /datasets/c4-128 \
  --evaluate-checkpoint /runs/memsolve-mlm/final --output /runs/mlm-eval

torchrun --standalone --nproc_per_node=2 -m experiments.train_mlm \
  --config /configs/continued-mlm.json --data /datasets/c4-512 \
  --init-checkpoint /runs/memsolve-mlm/final --output /runs/mlm-512
```

Independent evaluation uses all frozen validation targets and records the
checkpoint/data identities. To compare candidates, evaluate each on this same
split and select the lowest masked NLL. HF BERT baselines are supported when
the tokenizer, vocabulary, and position capacity match the prepared data.

Continuation starts a new optimizer and schedule; it is not resume. The
absolute-position model's 512-entry table supports a 128-to-512 continuation
without resizing. A RoVE checkpoint follows its saved frequency policy. Prepare
the new length explicitly and retain the checkpoint's geometry and position mode.

## GLUE fine-tuning

`experiments/configs/memsolve_bert_glue.json` specifies the LION M2-BASE
per-task recipe. It uses length 128, 6% warmup, linear decay to zero, and final
checkpoint evaluation. RTE, MRPC, and STS-B start from the fine-tuned MNLI encoder;
other tasks start from MLM. The pooler/classifier use BERT's dense+tanh head.
MNLI transfer keeps the pooler and initializes a new task classifier.

```bash
python -m experiments.train_glue --prepare-only \
  --checkpoint /runs/memsolve-mlm/final --data /datasets/glue-128
python -m experiments.train_glue --task mnli \
  --checkpoint /runs/memsolve-mlm/final --data /datasets/glue-128 \
  --output /runs/glue/mnli
python -m experiments.train_glue --task rte \
  --checkpoint /runs/glue/mnli/final --data /datasets/glue-128 \
  --output /runs/glue/rte
```

Use one GPU/process per task. Set the seed in a task recipe copy outside the
checkout for repeated runs. Partial batches are retained. Composer task decay
is divided by peak LR; STS-B's PyTorch decay is used directly. All parameters
receive decay, following the source GLUE jobs. The default scaffold executes
eagerly with the native CUDA mixer. `--compile` is optional. Resume uses
`--resume /runs/glue/task/checkpoint-N` with the same recipe and source.

Metrics include MNLI matched/mismatched accuracy, MRPC/QQP accuracy and F1,
STS-B Pearson and Spearman, and CoLA Matthews correlation. The paper averages
MNLI accuracy and QQP accuracy/F1, uses MRPC F1 and STS-B Spearman, then averages
eight task scores. Per-task means and seed sets are in the paper appendix.

## Retrieval and embedding evaluation

Retrieval uses Sentence Transformers' mean pooling,
`CachedMultipleNegativesRankingLoss`, no-duplicate sampling, and Trainer.
The [Ettin-derived recipe](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/main/retrieval_eval/train_st.py)
uses MS MARCO hard triplets, one epoch, loss scale 20, and 5% warmup. The
configuration uses contrastive batch 1024 per GPU, cache microbatch 128, and
no cross-GPU negative sharing. Accumulation does not enlarge the negative pool.
Held-out queries are excluded from training; triplet accuracy is a diagnostic,
not a corpus retrieval score.

```bash
torchrun --standalone --nproc_per_node=2 -m experiments.train_retrieval \
  --config experiments/configs/memsolve_bert_retrieval.json \
  --checkpoint /runs/memsolve-mlm/final --output /runs/retrieval
python -m experiments.evaluate_embeddings \
  --checkpoint /runs/retrieval/final --output /runs/mteb \
  --suite mteb-eng-v2 --max-length 512 --batch-size 32
python -m experiments.evaluate_embeddings \
  --checkpoint /runs/retrieval/final --output /runs/longembed \
  --suite longembed --max-length 512 --batch-size 16
```

The evaluator records truncation/window policy, checkpoint identity, task
metadata, and package versions. A 512-truncated LongEmbed run does not measure
full-document long-context understanding. `--local-ir` supports offline
retrieval fixtures. Results are local; there is no automatic Hub upload.
[Re-DocRED](RE_DOCRED.md) provides a separate document-relation workflow.
