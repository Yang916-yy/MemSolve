# MemSolve-BERT integration

**MemSolve** names the operator, **MemSolve-ViT** the visual family, and
**MemSolve-BERT** the text model. Python follows the framework's class naming:
`MemSolveBertConfig`, `MemSolveBertModel`, `MemSolveBertForMaskedLM` and
`MemSolveBertForSequenceClassification`.
Saved HF configurations use `model_type="memsolve_bert"` and the corresponding
class name in `architectures`; the MLM and retrieval launchers use
`experiments/configs/memsolve_bert_{mlm,retrieval}.json`.

`integrations/transformers.py` adapts the current MemSolve operator to Hugging Face
BERT. It reuses Transformers' embeddings, encoder layers, residual output
modules and tied MLM head, with packed SwiGLU in the encoder FFNs. It does not
implement a second attention operator.

Each layer computes:

\[
u=x+\mathrm{DWConv1D}_{k=3}(x),\qquad
h=\mathrm{LayerNorm}(x+\mathrm{Dropout}(\mathrm{MemSolve}(u))),\qquad
y=\mathrm{LayerNorm}(h+\mathrm{Dropout}(\mathrm{FFN}(h))).
\]

MemSolve owns its packed relation/content projection, learned compact core
shared across samples, bounded scalar complement and output projection. Its
relation frame and token operator remain input-dependent. BERT's attention
output dense layer becomes an identity, avoiding a duplicate Wo; its residual,
dropout and LayerNorm remain. `memsolve_core_mode` selects equilibrium (default)
or complement and is persisted in the HF configuration. There is no Q/K short
convolution, output RMSNorm or output gate. New models use RoVE-style value
rotations; `position_embedding_type="absolute"` retains BERT learned positions.
The default `memsolve_short_conv_kernel=3` adds a centered, depthwise residual
Conv1d on the mixer input, before the
relation/content projection: `x_local = x + DWConv1d(x)`. It uses native PyTorch
[Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
with `groups=hidden_size`, stride 1 and padding 1, without an activation.
Padding is zeroed before and after the convolution; the outer BERT residual
still uses the original input. The convolution weights and bias initialize to
zero, making the local transformation an identity at initialization. It preserves
the selected position mechanism and uses bidirectional rather than causal context.
Set `memsolve_short_conv_kernel=0` explicitly for the no-convolution ablation.
The option and its weights persist in HF checkpoints.

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
| Encoder depth / width / heads | 14 / 768 / 12 |
| MemSolve rank | 32 (initial NLP configuration, not a tuned choice) |
| FFN | SwiGLU, 2048 per branch, 4096 packed gate/up |
| Position mechanism | RoVE: full value-head rotation and inverse output rotation |
| Rotary base / training length / scaling | 10000 / 128 / frequency-only YaRN |
| Local context | Centered residual depthwise Conv1d, kernel 3, every layer |
| Vocabulary / token types | 30522 / 2 |
| Embedding and residual dropout | 0.1 |
| Attention probability dropout | 0; this operator does not materialize probabilities |
| BERT LayerNorm epsilon | 1e-12 |
| Parameters, including tied MLM head | 111,123,938 |
| Encoder parameters, no MLM head or CLS pooler | 110,501,288 |

HF's normal initialization (std 0.02, zero linear biases) applies to the BERT
scaffold and MemSolve B/C/output projections. After child linear initialization,
the restored core resets its learned shared core coordinates to zero and
its per-head complement to approximately 0.9. The MLM decoder shares the word
embedding weight. There is no trained CLS pooler in the MLM model or the default
base encoder.

The depth increase approximately matches the BERT-base parameter budget. A
12-layer MHA encoder with the same SwiGLU width and tied MLM head has 109,526,586
parameters; this model has 1.46% more. Stock 12-layer BERT with GELU and a tied MLM
head has 109,514,298 parameters. These counts exclude a CLS pooler and NSP head.
This is a parameter-budget comparison, not an equal-depth operator comparison.
The 12-layer short convolution pilot used 99,043,530 / 99,080,394 parameters
without / with convolution; it does not establish the 14-layer model's result.
The completed formal MLM checkpoint uses learned absolute positions and has
111,517,154 parameters (110,894,504 without its MLM head). Loading that checkpoint
retains its saved `position_embedding_type="absolute"`; it does not become a
RoVE model. The new recipe keeps the same 14 layers, SwiGLU, centered kernel-3
convolution and training budget, removing only the learned position table.

### Rotary positions and length extension

RoVE is a configuration-time option. Use `position_embedding_type="rove"` to
enable it, `"none"` to disable it without adding a position table, or
`"absolute"` for the learned BERT position table used by completed MLM/GLUE
experiments. The `rove`/`none` pair has identical parameter counts; centered
short convolution is controlled independently. At the standalone operator
boundary the switch is `MemSolveConfig(rotary=True/False)`.
`memsolve_rotary_scaling="none"` only disables YaRN interpolation, not RoVE.
The selected mode is saved in the checkpoint; changing it is an explicit
architecture ablation, not a per-batch inference toggle.

`position_embedding_type="rove"` uses the full per-head content dimension.
It rotates values before compact aggregation and inversely rotates the complete
global correction before the output projection. The local content term is
unchanged. The shared relation frame and compact core are not rotated.
This follows the value/output pathway in [RoVE, Eq. 3](https://arxiv.org/html/2606.11275v1#S3)
and its [official implementation](https://github.com/AGarciaCast/RoVE/blob/8e3eb84efb7c920289aba776fbdb189f93b79e01/model/pos_methods/rove.py#L25);
it is an adaptation to MemSolve, not a claim that RoVE validated this operator.

The HF configuration exposes `memsolve_rotary_base=10000.0`,
`memsolve_rotary_train_length=128`, and `memsolve_rotary_scaling="yarn"`
(`"none"` disables scaling). Frequency-only YaRN uses the official evaluator's
`beta_fast=32`, `beta_slow=1`: high frequencies stay fixed and low frequencies
are interpolated, with a blend between them. MemSolve has no softmax and applies
no softmax temperature or amplitude multiplier. See the
[upstream evaluation helper](https://github.com/AGarciaCast/RoVE/blob/8e3eb84efb7c920289aba776fbdb189f93b79e01/eval/_eval_utils.py#L30).

Scaling uses each example's span from its first valid token to its last valid
token, relative to the configured training length, not the largest position ID.
With no padding mask, the span is the sequence length. Explicit `position_ids`
produce one pair of cos/sin tables per BERT forward. The tables, padding mask,
and CUDA valid counts are passed through every layer and gradient-checkpoint
replay, following [HF Llama's shared position-embedding organization](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).
They are not cached across forward calls or optimizer steps; shifting all
positions by a constant leaves relative rotations unchanged. Padding remains
excluded from the operator and does not change another example's frequency
scale. Use matched truncation and position policies in comparisons.
For a fixed scale across a chosen longer window, the standalone core API accepts
an explicit `rotary_context_length`; the BERT wrapper uses the automatic mask span.

RoVE models have no learned position table or fixed-length position/token-type
buffers. `max_position_embeddings` remains an HF configuration field but is not
a capacity limit for this mode. Experiment manifests report `position_capacity`
as `null`; requested lengths are still bounded by available resources and are
truncated explicitly by the evaluation pipeline. The 128-token MLM recipe and
frequency interpolation alone do not establish long-context task performance.

## Implementation provenance

| Component | Implementation and adaptation |
| --- | --- |
| Tokenizer and embeddings | HF BERT WordPiece and `BertEmbeddings`; token/token-type embeddings, with learned positions only in explicit absolute mode |
| Residuals, dropout and LayerNorm | HF BERT output modules and PyTorch LayerNorm, including BERT's epsilon; mixer output dense is removed because MemSolve owns its output projection |
| Centered local convolution | Native PyTorch `nn.Conv1d(groups=hidden_size)`; our residual wiring, zero initialization and padding masks |
| SwiGLU | Thin packed `nn.Linear` + `F.silu` wrapper following the Phi3 layout; HF BERT owns the down projection and residual; no dedicated fused SwiGLU kernel |
| MLM head | HF `BertOnlyMLMHead`, including GELU, LayerNorm and tied vocabulary weights |
| MLM corruption and loss | HF `DataCollatorForLanguageModeling`, PyTorch cross entropy; our frozen validation masks and masked-target-weighted normalization/metrics |
| Training and optimizer | HF Trainer/Accelerate, PyTorch fused AdamW, HF polynomial scheduler, BF16 autocast with FP32 parameters |
| Retrieval and evaluation | Sentence Transformers pooling, cached contrastive loss and Trainer; official MTEB task implementations |
| MemSolve mixer | Our model mathematics, tiled Triton kernels and analytic VJP; NVIDIA cuSOLVERDx compact factorization/solves and cuBLASLt projection-gradient primitives |

The BERT/Phi3 source links are listed below. The operator's exact library/custom
kernel boundary and precision choices are documented in [CUDA_CONTRACT.md](CUDA_CONTRACT.md).
Reusing mature primitives does not make the custom operator, masking adapter or
architecture choices an unmodified upstream model. The training entry supports explicit `eager`, `graph` and `compile-graph`
execution; the measured two-A800 recipe selects `compile-graph` (details below).

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
from integrations.transformers import MemSolveBertConfig, MemSolveBertForMaskedLM

model = MemSolveBertForMaskedLM(MemSolveBertConfig(
    memsolve_rank=32,
    memsolve_implementation="cuda",  # choose "reference" explicitly for CPU
)).cuda()

# batch contains input_ids, binary attention_mask, and MLM labels.
# Labels are original token IDs at prediction targets and -100 elsewhere.
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(**batch).loss
loss.backward()

model.save_pretrained("/path/to/memsolve-bert")
restored = AutoModelForMaskedLM.from_pretrained("/path/to/memsolve-bert")
encoder = AutoModel.from_pretrained("/path/to/memsolve-bert")
```

Import `integrations.transformers` before using Auto classes in a fresh process;
it registers `memsolve_bert`, the encoder and the MLM model. CPU loading does not
execute the CUDA backend; choose `memsolve_implementation="reference"` explicitly
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
- HF safetensors contain tensors only; `config.json` carries `memsolve_contract`,
  validated against the actual core on construction. PyTorch load pre-hooks
  restore that validated metadata for the core's strict checkpoint check.
  Save and transfer config and weights together via `save_pretrained`.
- The config must declare `ffn_type="swiglu"`. The earlier GELU adapter checkpoint
  format is not retained or silently reinterpreted.
- Native standalone MemSolve checkpoints still require their extra-state contract.
  Missing or incompatible HF contracts and missing mixer weights are rejected.
  Loading LION/Hydra or stock BERT weights is not a checkpoint conversion path.

## Source decisions and checks

The scaffold follows the operator replacement boundary in
[LION's BERT implementation](https://github.com/LIONS-EPFL/LION/blob/32a0136431dea54a634394df756365919635fe9d/Masked_Language_Modeling/src/bert_layers.py).
The public LION YAML `hf_bert` route and its `linear_attention` / `nova` names
do not match the custom `bert` factory and `lion-lit` / `lion-d` / `lion-s`
branches in that checkout. We register the MemSolve model explicitly instead of
copying those configuration switches. No extra mixer output normalization is added. These are architecture differences,
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
Resolved dataset revisions, tokenizer hashes, limits, lengths, seeds and fingerprints
are recorded. `--tokenizer-revision` also accepts a pinned tokenizer commit.
Manifests count valid attention tokens, content tokens excluding specials, and
fixed validation targets. These distinguish useful tokens from padded tensor slots.
Completed matching partitions are reused; an interrupted job
rebuilds only its `.incomplete` partition. An OS file lock rejects a duplicate
job for the same partition before it can disturb an active preparation.
Training rejects missing or inconsistent partitions. Local JSONL `--files`
provide an offline path with the same preprocessing.

`--document-policy split` optionally retains a long document as nonoverlapping
windows. It never combines different documents. This changes the sampled data
distribution and must be reported; the default remains one truncated prefix per
document. With this option, `--max-docs` counts prepared windows. Use a new output
and the same policy for both splits when changing it.

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

### Local streaming training

When all raw C4 files are available locally, `--streaming-train-manifest` reads
those files through HF Datasets and tokenizes in persistent loader workers.
`--data` then supplies only the prepared tokenizer and frozen validation split;
incomplete training Arrow partitions are not used. Document filtering,
truncation and padding reuse the preparation implementation.

The JSON manifest declares `format: "memsolve-local-text-stream-v1"`,
`num_source_files`, `tokenizer_hash`, `max_length`, `packing: "one_document"`,
and `files`: objects containing absolute `path`, `size` and `mtime_ns`.
Record the dataset revision and download SHA256 in that manifest as provenance.
The loader checks file identity metadata and validation/tokenizer compatibility;
it does not rehash the full corpus on every launch. Pin and preserve the files.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m experiments.train_mlm \
  --config experiments/configs/memsolve_bert_mlm.json \
  --data /path/to/prepared-validation \
  --streaming-train-manifest /path/to/train-stream.json \
  --output /path/to/memsolve-mlm
```

Use `dispatch_batches=false` and `split_batches=false`. Accelerate shards the
HF iterable by source file across ranks; Datasets then partitions each rank's
files across workers. Do not manually shard the input again. There must be at
least one source file per rank/worker. Shuffle uses seed 17 and a 10,000-document
buffer per worker, not a full-corpus random permutation. This differs from
Arrow random-access sampling and must be reported with the experiment.

The explicit 70,000-step budget controls the iterable run. Standard Trainer
resume restores model/optimizer/scheduler state and replays/skips consumed
batches; it can require substantial CPU reading after a late interruption.
Persistent-worker corruption masks are not guaranteed bitwise identical after
resume. Do not set `ignore_data_skip=true`, which would restart data consumption.
The frozen validation set is unchanged by this choice.

### MLM pretraining

```bash
# Inspect actual prepared data and the planned two-GPU batch without allocating CUDA tensors.
/cywang/ridgon-nlp-venv/bin/python -m experiments.train_mlm \
  --config experiments/configs/memsolve_bert_mlm.json \
  --data /path/to/c4-tokenized --check-only --world-size 2

/cywang/ridgon-nlp-venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m experiments.train_mlm \
  --config experiments/configs/memsolve_bert_mlm.json \
  --data /path/to/c4-tokenized --output /path/to/memsolve-mlm
# Resume: repeat the command with --resume /path/to/memsolve-mlm/checkpoint-1000
```

The checked-in r32/SwiGLU configuration is **LION-derived**, not an exact LION
reproduction: sequence length is set by prepared data (128 in the C4 command),
global batch 4096, 70,000 optimizer updates, seed 17, training mask probability
30%, peak LR 8e-4, 6% warmup and a linear decay to 2% of peak LR. The official
[LION recipe](https://github.com/LIONS-EPFL/LION/blob/32a0136431dea54a634394df756365919635fe9d/Masked_Language_Modeling/yamls/pretrain/lion-lit-base.yaml)
motivates these settings. The completed absolute-position run's measured
two-A800 configuration uses microbatch 512 per GPU and four
accumulation steps, with eight persistent loader workers per rank and prefetch
factor two. Change microbatch and accumulation together to preserve global 4096. HF Trainer handles DDP, BF16 AMP, fused AdamW, scheduling and resume.
Parameters stay FP32, including evaluation. The entrypoint uses `compile-graph` for model/loss forward and backward,
with DDP communication and optimizer steps outside capture.

The full schedule targets 286,720,000 sequence presentations, or 36.70 billion
tensor token slots at length 128. These are not unique documents or non-padding
tokens. The complete 70,000-step recipe is retained; measure real-data step time,
validation cost, peak memory and I/O before starting it. A CPU preflight is not a
CUDA throughput measurement. The completed formal run used absolute positions;
no formal C4 pretraining result is available for the new RoVE recipe yet.

### Compiled CUDA Graph execution

The recipe's top-level `execution` selects `eager`, `graph` or `compile-graph`.
`compile-graph` uses Inductor to fuse the BERT scaffold and MLM loss while
retaining the existing MemSolve CUDA boundary. It disables Inductor's internal
CUDA Graphs and captures the tensor-only forward/loss and backward with
[PyTorch `make_graphed_callables`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html).
Capture happens before DDP construction. DDP reduction, accumulation, clipping,
fused AdamW, the scheduler and checkpointing remain in HF Trainer.

Parameters stay FP32; forward uses BF16 autocast with its weight cache disabled.
AOTAutograd specializes for backward outside autocast. Inductor preserves eager
precision casts (`emulate_precision_casts=true`). Capture warmup does not update
weights and restores CPU/CUDA RNG state and buffers before training. Graph RNG
advances on replay; compiler dropout is not required to be bitwise identical to
eager dropout. Validation uses the ordinary forward, includes every validation
sample, and never replays the fixed training graph.

Loss normalization still uses the masked-target count over the complete global
accumulation group. Independent parameter-gradient accumulators prevent the
replay-buffer aliasing reported in
[PyTorch issue #181723](https://github.com/pytorch/pytorch/issues/181723).
No optimizer update occurs during graph capture. The graph is recreated on
resume and checkpoint weights are loaded in place; graph objects are not saved
in HF checkpoints.

Training IDs, masks, token types and labels must retain the captured CUDA int64
shape. Explicit `position_ids` can also be captured as a `[batch, length]`
int64 tensor; their values and the per-example masks may change between replays.
The usual C4 pipeline omits them and generates consecutive positions in the graph.
A partial training microbatch is rejected rather than silently padded,
dropped or switched to eager. The full local C4 source exceeds this run's sample
budget. Do not combine this execution mode with Trainer's separate
`torch_compile`, gradient checkpointing, full-parameter BF16 casts or changes to
parameter storage after capture. Startup compilation is excluded from steady
throughput. `execution=eager` retains the ordinary HF path.

Composer's decay multiplies weights by `1 - (lr_t / lr_peak) * wd_composer`.
PyTorch multiplies by `1 - lr_t * wd_torch`; therefore the equivalent coefficient
is `1e-5 / 8e-4 = 0.0125`. This conversion follows
[Composer's optimizer implementation](https://docs.mosaicml.com/projects/composer/en/latest/_modules/composer/optim/decoupled_weight_decay.html).
If changing peak LR while preserving Composer's decay, recompute that ratio.
HF's standard norm/bias exclusions are retained. SwiGLU/MemSolve, fixed evaluation
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

`run.json` records the recipe, data fingerprints, world size, dependency
versions and a content hash of the operator/native sources and NLP pipeline.
Resume requires unchanged source, the same recipe/world size and a checkpoint from that
output directory. Use a new output for a changed experiment. HF saves optimizer,
scheduler and RNG state; stochastic masking with persistent loader workers is
not guaranteed to resume bit-for-bit. Frozen validation masks are reproducible.
All DDP ranks receive output-directory validation failures before training.
The default evaluates and checkpoints every 1,000 steps, selects the lowest
`eval_masked_nll`, and exports that model/tokenizer to `final/`.
`final/selection.json` records the selection metric and source checkpoint.
Resume from `checkpoint-N`, not `final/`: exported best weights are not an
optimizer/scheduler snapshot.

### Independent MLM evaluation and continued pretraining

```bash
/cywang/ridgon-nlp-venv/bin/python -m experiments.train_mlm \
  --config experiments/configs/memsolve_bert_mlm.json \
  --data /path/to/c4-tokenized \
  --evaluate-checkpoint /path/to/memsolve-mlm/final --output /path/to/mlm-eval

# A new training stage; use its own recipe, prepared data and output.
/cywang/ridgon-nlp-venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m experiments.train_mlm \
  --config /path/to/continued-mlm.json --data /path/to/c4-512 \
  --init-checkpoint /path/to/memsolve-mlm/final --output /path/to/mlm-512
```

Independent evaluation needs only the prepared validation split. It uses the
same frozen target positions and saves token-weighted metrics under a directory
keyed by checkpoint, data and evaluation settings. Local HF BERT MLM baselines
are also supported if their tokenizer/vocabulary and position capacity match.
Different tokenizers do not produce comparable token-level NLL on these labels.

`--init-checkpoint` loads MLM weights into a new stage with a fresh optimizer and
schedule. It differs from exact-run `--resume`; changing the stage also requires
its original initialization path when later resuming that stage. Model geometry
must match, including the position mechanism. For the completed absolute-position
checkpoint, its 512-position table permits 128-to-512 continuation without
resizing. A RoVE checkpoint instead follows its saved frequency-scaling policy.
Prepare the new sequence length explicitly and set that stage's own budget and
microbatch; neither path inserts a continuation stage automatically. Loading an
absolute checkpoint with the new RoVE recipe is rejected as a geometry mismatch.

LION/Hydra checkpoints using their original model classes or Composer containers
are not directly HF-loadable. Their original loaders or a separately verified
conversion are still needed for baseline evaluation. A successful MemSolve HF
round trip does not validate those checkpoints.

### Retrieval fine-tuning

```bash
/cywang/ridgon-nlp-venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m experiments.train_retrieval \
  --config experiments/configs/memsolve_bert_retrieval.json \
  --checkpoint /path/to/memsolve-mlm/final --output /path/to/memsolve-retrieval
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
adds recomputation. Ettin uses cache mini-batches of 256. The current 128 is a
starting setting, not a measured optimum for this operator contract. Earlier
102M-parameter memory probes used a different mixer and do not establish current
capacity. Measure throughput and peak memory with actual triplets before a full
retrieval run, then adjust the cache mini-batch while retaining the contrastive
batch and negative-sharing policy.
No-duplicate sampling can produce smaller batches. The selected train/validation
triplets must have nonempty string columns `query`, `positive`, `negative`;
`--data` accepts a local HF dataset.

The held-out triplet metric is a diagnostic, **not full MS MARCO retrieval**.
`final/` is a reloadable Sentence Transformer with mean pooling and tokenizer.
The unneeded MLM prediction head is discarded. Resume uses `--resume` as above.

### MTEB and long-document evaluation

```bash
/cywang/ridgon-nlp-venv/bin/python -m experiments.evaluate_embeddings \
  --checkpoint /path/to/memsolve-retrieval/final --output /path/to/mteb-results \
  --suite mteb-eng-v2 --max-length 512 --batch-size 32
/cywang/ridgon-nlp-venv/bin/python -m experiments.evaluate_embeddings \
  --checkpoint /path/to/memsolve-retrieval/final --output /path/to/longembed-results \
  --suite longembed --max-length 512 --batch-size 16
```

Evaluation uses the current official
[MTEB API](https://docs.mteb.org/get_started/usage/running_the_evaluation/), its
English v2 benchmark definition and native task metrics/splits. `--tasks` selects
individual tasks; `--list-tasks` resolves metadata without downloading task data.
`--suite mteb-eng-v2-retrieval` selects its ten retrieval tasks as a cheaper first
evaluation; a subset mean must not be reported as the full MTEB English v2 score.
LongEmbed selects SummScreenFD, QMSum, WikimQA and NarrativeQA retrieval. It does
not include the synthetic passkey/needle tasks. Dataset revisions and actual
evaluation splits are saved, including SummScreenFD's validation split.

The completed absolute-position checkpoint evaluates long documents **with
truncation to 512**; larger requested lengths fail against its learned table.
RoVE checkpoints can request longer input lengths and apply their saved scaling
policy, without resizing a table. Both modes truncate to the requested
`--max-length` and do not add chunking. A model trained only at 128 tokens has
not established competence at 512 or longer simply because the forward is
defined. Compare baselines with matched length and padding policies.
Ettin's long-context benchmark is **MLDR**, not LongEmbed; the latter is our
separate evaluation choice. The two scores are not interchangeable.

The evaluation manifest records length policy, AMP, normalization, checkpoint
content hash, source hash, task metadata and package versions. These determine a distinct
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

## GLUE: LION M2-BASE fine-tuning

`MemSolveBertForSequenceClassification` reuses the Hugging Face BERT CLS pooler
(dense + tanh), dropout and classifier. It uses cross-entropy for classification
and MSE for STS-B. All encoder weights transfer from MLM; the absent pooler and
label head are initialized with BERT's initializer. For MNLI transfer, the trained
pooler is retained and the task classifier is reset.

The recipe in `experiments/configs/memsolve_bert_glue.json` follows
[LION Appendix C.5, Table 8, M2-BASE](https://arxiv.org/html/2502.16249v2#S0.T8).
Batch sizes, evaluation intervals, sequential sampling, no dropped final batch,
optimizer betas `(0.9, 0.98)` and epsilon `1e-6` follow the
[public GLUE jobs](https://github.com/LIONS-EPFL/LION/blob/32a0136431dea54a634394df756365919635fe9d/Masked_Language_Modeling/src/glue/finetuning_jobs.py).
The current upstream YAML selects a different recipe; this config deliberately
implements the paper's M2-BASE column. All tasks use length 128, 6% step warmup,
linear decay to zero, BF16 autocast with FP32 parameters, and no gradient clipping.

| Task | Initial weights | Epochs | Batch | Peak LR | Published WD | WD convention |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MNLI | MLM | 3 | 48 | 5e-5 | 5e-6 | Composer |
| QNLI | MLM | 10 | 16 | 5e-5 | 1e-5 | Composer |
| QQP | MLM | 5 | 16 | 3e-5 | 3e-6 | Composer |
| RTE | MNLI final | 3 | 16 | 1e-5 | 1e-6 | Composer |
| SST-2 | MLM | 3 | 16 | 3e-5 | 3e-6 | Composer |
| MRPC | MNLI final | 10 | 32 | 8e-5 | 8e-5 | Composer |
| CoLA | MLM | 10 | 32 | 8e-5 | 5e-6 | Composer |
| STS-B | MNLI final | 10 | 32 | 8e-5 | 3e-6 | PyTorch |

[Composer DecoupledAdamW](https://github.com/mosaicml/composer/blob/v0.17.2/composer/optim/decoupled_weight_decay.py)
multiplies parameters by `1 - wd * lr_t / peak_lr`. PyTorch AdamW uses
`1 - wd * lr_t`; therefore the runner divides Composer WD by peak LR and uses
fused PyTorch AdamW. STS-B's plain AdamW WD is unchanged. All parameters receive
weight decay, matching upstream `create_param_groups(None, model)`. The optimizer
port has equivalent decay algebra; it does not promise bitwise Composer results.
Trainer owns scheduling, evaluation, saving and resuming.

Prepare the pinned `nyu-mll/glue` Parquet train/dev splits using the checkpoint's
own tokenizer. Test labels are hidden and are not downloaded or used:

```bash
python -m experiments.train_glue --prepare-only \
  --checkpoint /path/to/mlm/final --data /path/to/glue-128
CUDA_VISIBLE_DEVICES=0 python -m experiments.train_glue --task mnli \
  --checkpoint /path/to/mlm/final --data /path/to/glue-128 \
  --output /path/to/glue/mnli
CUDA_VISIBLE_DEVICES=1 python -m experiments.train_glue --task sst2 \
  --checkpoint /path/to/mlm/final --data /path/to/glue-128 \
  --output /path/to/glue/sst2
# After MNLI completes:
CUDA_VISIBLE_DEVICES=0 python -m experiments.train_glue --task rte \
  --checkpoint /path/to/glue/mnli/final --data /path/to/glue-128 \
  --output /path/to/glue/rte
```

Use one process/GPU/task. The default runs the native CUDA mixer with an eager
BERT scaffold. Optional `--compile` uses Inductor's reduce-overhead mode on the
BERT scaffold and retains the native MemSolve mixer. Partial final batches are
kept and may require an additional compiled shape. This is not the fixed-shape,
whole-forward/backward graph used by the MLM launcher. On a fixed pretrained
MNLI batch with dropout disabled, compilation changed representative gradients
by 1.36–4.28% relative L2 (loss difference 0.00049). This is not a demonstrated
accuracy regression, but the initial GLUE suite leaves compilation disabled
to avoid adding that numerical variable. `--max-steps` is only a
smoke-test override and is recorded in `results.json`; formal runs omit it.
Resume with `--resume /path/to/glue/task/checkpoint-N` using the same recipe,
source and output directory. Persistent loaders default to two workers.

The runner exports the final checkpoint and final validation scores, as upstream
GLUE jobs do, rather than selecting a different checkpoint for each metric.
MNLI matched/mismatched accuracies are separate; MRPC/QQP include accuracy and
binary F1; STS-B includes Pearson and Spearman; CoLA uses Matthews correlation.
Scores are fractions. Metrics use scikit-learn/SciPy, the same definitions used
by the HF GLUE metric. LION's source reports MRPC F1 and STS-B Spearman, so do not
replace them with accuracy/Pearson when comparing its table. The first run is
seed 19 only: compare per-task validation results provisionally and report seed
count explicitly; it is not a multi-seed mean or an official GLUE test submission.
