# Ridgon BERT integration

`integrations/transformers.py` adapts the current Ridgon operator to Hugging Face
BERT. It reuses Transformers' embeddings, encoder layers, residual output
modules and tied MLM head, with packed SwiGLU in the encoder FFNs. It does not
implement a second attention operator.

Each layer computes:

\[
h=\mathrm{LayerNorm}(x+\mathrm{Dropout}(\mathrm{Ridgon}(x))),\qquad
y=\mathrm{LayerNorm}(h+\mathrm{Dropout}(\mathrm{FFN}(h))).
\]

Ridgon includes its packed QKV projection, per-head output RMSNorm with shared
channel gain, and output projection. BERT's attention output dense layer becomes
an identity, avoiding a duplicate output projection. The attention residual
dropout and LayerNorm remain. No additional mixer LayerNorm, Q/K convolution,
RoPE, CPE or LayerScale is added.

The FFN uses `silu(gate) * up`, with gate and up packed into one linear
projection. BERT's FFN output module retains the down projection, dropout,
residual and LayerNorm. The default gated width is `8 * hidden_size / 3`, rounded
up to 16 channels: for width 768 this is 2048 per branch and a packed projection
width of 4096. This matches the matrix-weight budget of the former 3072-channel
two-projection FFN. Linear biases are retained. `intermediate_size` now denotes
the width of each gated branch. `hidden_act` still controls the unchanged MLM
prediction transform (GELU); the encoder FFN is always SwiGLU.

## Default geometry and initialization

| Setting | Value |
| --- | --- |
| Encoder depth / width / heads | 12 / 768 / 12 |
| Ridgon rank | 32 (initial NLP configuration, not a tuned choice) |
| FFN | SwiGLU, 2048 per branch, 4096 packed gate/up |
| Position embedding | BERT learned absolute, 512 positions |
| Vocabulary / token types | 30522 / 2 |
| Embedding and residual dropout | 0.1 |
| Attention probability dropout | 0; this operator does not materialize probabilities |
| BERT LayerNorm epsilon | 1e-12 |
| Ridgon head RMSNorm epsilon | 1e-6, owned by the core |
| MLM parameters, tied weights counted once | r32: 102,587,706; r48: 106,315,578 |

HF's normal initialization (std 0.02, zero linear biases) applies to the BERT
scaffold and Ridgon Q/V/output projections. After child linear initialization,
the core restores its K fan-in initialization. Delta starts at zero, so T starts
at identity; RMSNorm gain starts at one. The MLM decoder shares the word
embedding weight. There is no trained CLS pooler in the MLM model or the default
base encoder.

These dimensions preserve the BERT-base scaffold but do not parameter-match a
softmax BERT-base model. The 512-position table does not imply trained long-context
ability; any position extension and longer-sequence training require a separate
experimental decision.

## Use from the source checkout

Install the optional dependencies into a separate NLP environment if other
experiments require a different Transformers / Hugging Face Hub version:

```bash
python -m pip install -e '.[nlp]'
```

Run from the repository root (framework integrations are source-checkout
modules, as with the timm integration):

```python
import torch
from transformers import AutoModel, AutoModelForMaskedLM
from integrations.transformers import RidgonBertConfig, RidgonBertForMaskedLM

model = RidgonBertForMaskedLM(RidgonBertConfig(
    ridgon_rank=32,
    ridgon_implementation="cuda",  # choose "reference" explicitly for CPU
)).cuda()

# batch contains input_ids, binary attention_mask, and MLM labels.
# Labels are original token IDs at prediction targets and -100 elsewhere.
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(**batch).loss
loss.backward()

model.save_pretrained("/path/to/ridgon-bert")
restored = AutoModelForMaskedLM.from_pretrained("/path/to/ridgon-bert")
encoder = AutoModel.from_pretrained("/path/to/ridgon-bert")
```

Import `integrations.transformers` before using Auto classes in a fresh process;
it registers `ridgon_bert`, the encoder and the MLM model. CPU loading does not
execute the CUDA backend; choose `ridgon_implementation="reference"` explicitly
when loading for CPU forward. Do not cast CUDA model parameters to BF16/FP16:
the operator requires FP32 parameters and uses its existing mixed precision
internally. Plain CUDA inference casts mixer activations to BF16; whole-model
BF16 autocast also accelerates the surrounding BERT linears.

Save the BERT tokenizer alongside the model with
`tokenizer.save_pretrained(checkpoint_dir)`, or give it to the HF Trainer as
`processing_class=tokenizer`. This preserves the tokenizer class and vocabulary
for downstream AutoTokenizer / Sentence Transformers loading. The experiment
entrypoints below carry the tokenizer through MLM and retrieval exports.

The model supports the stock HF `Trainer`, BF16 autocast, ordinary AdamW
(including fused CUDA AdamW), gradient checkpointing and checkpoint resume.
Use `TrainingArguments(bf16=True, optim="adamw_torch_fused", ...)` for CUDA
training; leave `bf16_full_eval=False` so evaluation does not cast parameters.
No C4 download or formal pretraining is started by this integration.

## Masks and checkpoint contract

- Pass a binary `[batch, length]` attention mask. Padding is excluded from the
  key statistics, key/value memory and effective token count in every layer.
  A masked-language-model `[MASK]` token remains a valid token. MLM target
  selection is controlled by labels, not by the attention mask.
- The outer BERT residual can leave nonzero hidden states at padding positions.
  Downstream mean pooling must use the attention mask.
- Pairwise/block masks, causal decoding, cross attention, KV caching, head
  pruning and attention-probability output are unsupported and fail explicitly.
  Do not concatenate independent documents and expect masked isolation.
- HF safetensors contain tensors only; `config.json` carries `ridgon_contract`,
  validated against the actual core on construction. PyTorch load pre-hooks
  restore that validated metadata for the core's strict checkpoint check.
  Save and transfer config and weights together via `save_pretrained`.
- The config must declare `ffn_type="swiglu"`. The earlier GELU adapter checkpoint
  format is not retained or silently reinterpreted.
- Native standalone Ridgon checkpoints still require their extra-state contract.
  Missing or incompatible HF contracts and missing mixer weights are rejected.
  Loading LION/Hydra or stock BERT weights is not a checkpoint conversion path.

## Source decisions and checks

The scaffold follows the operator replacement boundary in
[LION's BERT implementation](https://github.com/LIONS-EPFL/LION/blob/32a0136431dea54a634394df756365919635fe9d/Masked_Language_Modeling/src/bert_layers.py).
The public LION YAML `hf_bert` route and its `linear_attention` / `nova` names
do not match the custom `bert` factory and `lion-lit` / `lion-d` / `lion-s`
branches in that checkout. We register the Ridgon model explicitly instead of
copying those configuration switches. LION-Lit's optional output LayerNorm is
not added on top of Ridgon's output RMSNorm. These are architecture differences,
not an exact reproduction of LION training.

The implementation imports the Apache-2.0
[Transformers BERT modules](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/bert/modeling_bert.py)
and reuses their forward methods. It does not vendor the LION BERT fork.
The packed gate/up layout follows
[Transformers Phi3MLP](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/phi3/modeling_phi3.py),
with BERT biases and its existing output/residual module retained. SwiGLU and the
two-thirds hidden-width adjustment are motivated by
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
The encoder FFN therefore also differs from LION's GELU scaffold.

```bash
python -m pytest -q tests/integrations/test_transformers.py tests/core/test_model.py -m 'not cuda'
python -m pytest -q tests/integrations/test_transformers.py -m cuda
python tools/check_repository.py
```

Tests cover initialization, padding invariance, MLM gradients, gradient
checkpointing, tied weights, sharded/unsharded safetensors round trips, encoder
extraction, contract rejection and BF16 CUDA forward/gradient agreement with
the reference. `tests/experiments/test_nlp.py` additionally runs an entirely local
text → MLM → retrieval → MTEB fixture, including Trainer checkpoint resume.

## Experiment entrypoints

Use the separate NLP environment, currently `/cywang/ridgon-nlp-venv`, from the
repository root. The tested stack is Transformers 4.57.6, Datasets 4.8.5,
Sentence Transformers 5.7.0, Accelerate 1.15.0 and MTEB 2.21.6. The `nlp` extra
declares compatible version ranges. Do not install these into the running vision
environment, or combine them with the Hub-1.x Assembly101 extra.

These scripts use existing framework training/evaluation loops. There is no new
optimizer, attention implementation or hand-written retrieval metric. Commands
below are launch examples; implementing the entrypoints does not launch full C4
preparation or training. All checkpoints, datasets and results go outside Git.

### C4 preparation

`experiments/prepare_c4.py` streams `allenai/c4` English through HF Datasets,
tokenizes each document separately, truncates/pads it to 128 tokens, and saves
memory-mappable Arrow partitions. There is no cross-document packing. Training
data remain unmasked; validation uses fixed, content-seeded HF MLM masks at
15%, independent of preparation batch order. Empty/special-only documents are
excluded. `[MASK]` remains a valid attention token.

```bash
# One job; add --max-docs 100000 for a bounded development subset.
/cywang/ridgon-nlp-venv/bin/python -m experiments.prepare_c4 \
  --output /path/to/c4-tokenized --split train
/cywang/ridgon-nlp-venv/bin/python -m experiments.prepare_c4 \
  --output /path/to/c4-tokenized --split validation --max-docs 10000
```

For CPU parallelism and restart granularity, run independent file-shard jobs:

```bash
seq 0 31 | xargs -P 8 -I '{}' /cywang/ridgon-nlp-venv/bin/python \
  -m experiments.prepare_c4 --output /path/to/c4-tokenized \
  --split train --num-shards 32 --shard-index '{}'
```

Use one partition scheme per split/output; do not mix this example with an
existing one-partition preparation. `--max-docs` limits **each partition**, and
selects a prefix of its source files, not a uniform sample of all C4. Pin
`--revision` to one Hub commit across jobs for a formally fixed dataset.
Resolved revisions, tokenizer hashes, limits, lengths, seeds and fingerprints
are recorded. Completed matching partitions are reused; an interrupted job
rebuilds only its `.incomplete` partition. Do not launch the same partition twice.
Training rejects missing or inconsistent partitions. Local JSONL `--files`
provide an offline path with the same preprocessing.

Storage uses int32 token IDs and int8 masks/types, about 896 bytes/document at
length 128 with the BERT tokenizer, excluding metadata. Building an individual
partition temporarily holds both its generation cache and saved Arrow data;
allow roughly twice that partition's space on top of completed partitions.
Streaming avoids a full raw-C4 copy. Full English C4 is still a large storage
task; select the dataset budget and partition count before launching it.

Network settings come from the process environment/HF credentials. For a direct
connection, prefix a launch with `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
-u http_proxy -u https_proxy -u all_proxy`; set `HF_ENDPOINT` in that process if
using a mirror. The scripts do not change machine-wide proxy settings or write
tokens to configuration/logs.

### MLM pretraining

```bash
/cywang/ridgon-nlp-venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m experiments.train_mlm \
  --config experiments/configs/ridgon_bert_mlm.json \
  --data /path/to/c4-tokenized --output /path/to/ridgon-mlm
# Resume: repeat the command with --resume /path/to/ridgon-mlm/checkpoint-1000
```

The checked-in r32/SwiGLU configuration is **LION-derived**, not an exact LION
reproduction: sequence length is set by prepared data (128 in the C4 command),
global batch 4096, 70,000 optimizer updates, seed 17, training mask probability
30%, peak LR 8e-4, 6% warmup and a linear decay to 2% of peak LR. The official
[LION recipe](https://github.com/LIONS-EPFL/LION/blob/32a0136431dea54a634394df756365919635fe9d/Masked_Language_Modeling/yamls/pretrain/lion-lit-base.yaml)
motivates these settings. Two GPUs with microbatch 128 use 16 accumulation
steps. HF Trainer handles DDP, BF16 AMP, fused AdamW, scheduling and resume.
Parameters stay FP32, including evaluation. The entrypoint does not enable
whole-model CUDA Graph or compile.

Composer's decay multiplies weights by `1 - (lr_t / lr_peak) * wd_composer`.
PyTorch multiplies by `1 - lr_t * wd_torch`; therefore the equivalent coefficient
is `1e-5 / 8e-4 = 0.0125`. This conversion follows
[Composer's optimizer implementation](https://docs.mosaicml.com/projects/composer/en/latest/_modules/composer/optim/decoupled_weight_decay.html).
If changing peak LR while preserving Composer's decay, recompute that ratio.
HF's standard norm/bias exclusions are retained. SwiGLU/Ridgon, fixed evaluation
masks, the selected validation subset, gradient clipping at 1.0 and HF optimizer
grouping must be disclosed when comparing against LION; this is not a claim of
identical training protocols.

Loss sums are normalized by the number of masked targets across accumulation
steps and DDP ranks. Empty-target microbatches contribute zero. Evaluation
exports `eval_masked_nll`, `eval_masked_accuracy`, `eval_masked_perplexity` and
target counts without gathering full vocabulary logits. Use `eval_masked_nll`
for comparisons: ordinary Trainer `eval_loss` averages per-batch means and can
differ with variable target counts. Masked-token perplexity is not autoregressive
language-model perplexity. Validation is never dropped at a batch boundary.
Training may have a partial last batch at a dataset boundary; global batch 4096
is the full-batch target, not a claim that every update has exactly 4096 examples.

`run.json` records the recipe, data fingerprints, world size and dependency
versions. Resume requires the same recipe/world size and a checkpoint from that
output directory. Use a new output for a changed experiment. HF saves optimizer,
scheduler and RNG state; stochastic masking with persistent loader workers is
not guaranteed to resume bit-for-bit. Frozen validation masks are reproducible.
Exported HF model/tokenizer files are in `final/`.

### Retrieval fine-tuning

```bash
/cywang/ridgon-nlp-venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m experiments.train_retrieval \
  --config experiments/configs/ridgon_bert_retrieval.json \
  --checkpoint /path/to/ridgon-mlm/final --output /path/to/ridgon-retrieval
```

This imports Sentence Transformers' `Transformer`, masked `Pooling(mean)`,
`CachedMultipleNegativesRankingLoss`, no-duplicate batch sampler, Trainer and
TripletEvaluator. The dataset and loss follow the
[Ettin/ModernBERT retrieval recipe](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder/blob/main/retrieval_eval/train_st.py):
MS MARCO mined hard triplets, one epoch, scale 20 and 5% warmup. LR 2e-5 and
weight decay 0.01 are explicit starting settings, not a tuned result. The bounded
training subset contains at most 1.25M triplets. After a seeded 1000-row held-out
split, any training rows with a held-out query are excluded to prevent query
overlap. This is stricter than the upstream row split and can reduce train size.
Counts and fingerprints are recorded.

The contrastive batch is 1024 **per GPU**, processed in cache mini-batches of 128.
Cross-GPU negative gathering is off; two GPUs thus update on up to 2048 triplets,
but each query sees candidates from its local batch. Gradient accumulation does
not enlarge the negative pool. The cached loss reduces activation memory and
adds recomputation. Ettin uses cache mini-batches of 256; our initial 32 was
the Sentence Transformers default, not a measured optimum. With the 102M
parameter Ridgon BERT and all 512 tokens valid, a local BF16 forward/backward
probe on an A800 used 25.3 GiB at 128. A 256-row probe exceeded a 48%-of-card
process memory cap set to preserve room for the concurrent ImageNet run; that
does not establish that 256 would fail on an otherwise empty A800. The default
is 128 for this shared-machine workload. Real triplet lengths vary, so measure
throughput/peak memory again before a full retrieval run and increase it if
headroom permits.
No-duplicate sampling can produce smaller batches. All triplets must have string
columns `query`, `positive`, `negative`; `--data` accepts a local HF dataset.

The held-out triplet metric is a diagnostic, **not full MS MARCO retrieval**.
`final/` is a reloadable Sentence Transformer with mean pooling and tokenizer.
The unneeded MLM prediction head is discarded. Resume uses `--resume` as above.

### MTEB and long-document evaluation

```bash
/cywang/ridgon-nlp-venv/bin/python -m experiments.evaluate_embeddings \
  --checkpoint /path/to/ridgon-retrieval/final --output /path/to/mteb-results \
  --suite mteb-eng-v2 --max-length 512 --batch-size 32
/cywang/ridgon-nlp-venv/bin/python -m experiments.evaluate_embeddings \
  --checkpoint /path/to/ridgon-retrieval/final --output /path/to/longembed-results \
  --suite longembed --max-length 512 --batch-size 16
```

Evaluation uses the current official
[MTEB API](https://docs.mteb.org/get_started/usage/running_the_evaluation/), its
English v2 benchmark definition and native task metrics/splits. `--tasks` selects
individual tasks; `--list-tasks` resolves metadata without downloading task data.
LongEmbed selects SummScreenFD, QMSum, WikimQA and NarrativeQA retrieval. It does
not include the synthetic passkey/needle tasks. Dataset revisions and actual
evaluation splits are saved, including SummScreenFD's validation split.

The current 512-position model evaluates long documents **with truncation to
512**, not with native full-document context. Larger requested lengths fail
unless the loaded checkpoint already has the position capacity. Position
extension and longer-sequence training remain separate model/experiment work;
the script neither resizes positions nor hides chunking. A model trained only
at 128 tokens has not established competence at 512 or longer simply because
the position table exists. Compare baselines with matched length policies.

The evaluation manifest records length policy, AMP, normalization, checkpoint
content hash, task metadata and package versions. These determine a distinct
result directory, preventing scores from different checkpoints/lengths from
sharing a cache. MTEB's `only-missing` policy resumes completed tasks. There is
no automatic Hub upload. `--local-ir` accepts dictionaries named `queries`,
`corpus` and `relevant_docs` (query ID → relevant document-ID list) for offline
Sentence Transformers retrieval scoring. For CPU fixtures pass
`--implementation reference --device cpu --fp32`.

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /cywang/ridgon-nlp-venv/bin/python -m pytest -q tests/experiments/test_nlp.py
```
