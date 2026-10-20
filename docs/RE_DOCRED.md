# Re-DocRED fine-tuning

This experiment uses the published **ATLOP without localized context pooling
(LOP)** design. BERT and MemSolve-BERT share the same relation head and data
pipeline. The head needs hidden states, not encoder attention probabilities.
It is not the full ATLOP or DREEAM model.

## Model

Insert `*` before and after entity mentions and take the starting marker's
last-layer hidden state. Aggregate an entity's mentions with channel-wise
logsumexp. Separate linear/tanh projections produce subject and object
representations. A grouped bilinear classifier predicts 96 relations and one
adaptive threshold class, TH. Training uses the original adaptive-threshold
loss; decoding selects relations above TH, with the ATLOP recipe's top-four
cap. `max_relations: 0` disables that cap and constitutes a decoding change.

For the existing 768-wide encoder, each projection is 768 → 768 and the
classifier has 12 groups of 64 channels. The head adds **5,949,025 parameters**.
All encoder and head weights are fine-tuned. The existing centered convolution
and learned absolute position embeddings remain in the encoder.

Pair classification is chunked to limit the size of each group-outer-product
temporary. All directed entity pairs are included, including negative pairs.
Its cost scales quadratically with the number of entities; linear sequence
scaling of the MemSolve operator is not an end-to-end complexity claim for this
task head.

## Long documents

The checkpoint has 512 learned position slots. Documents use overlapping
512-token BERT windows (510 content tokens), with a content stride of 256.
The last window is anchored at the document end. No tokens or entity mentions
are truncated. Overlapping hidden states are averaged in FP32, then mentions
are pooled in FP32. Positions restart in each window.

This is windowed document encoding, not single-pass full-document attention:
tokens in separate windows do not directly interact inside the encoder.
Both backbones use the same windowing policy. The manifest reports the number
of documents longer than 510 tokens. Head GEMMs use the training autocast
dtype; the AT loss uses FP32 log-softmax. Parameters and optimizer state stay
FP32 under BF16 AMP.

## Recipe

The starting settings follow the ATLOP BERT recipe:

| Setting | Value |
|---|---|
| Epochs | 30 |
| Global batch | 4 documents |
| Encoder / head learning rate | 5e-5 / 1e-4 |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-6 |
| Weight decay | 0 |
| Schedule | Linear decay, 6% warmup |
| Gradient clipping | 1.0 |
| Selection | Best development F1 |
| Default seed | 0 |

The current MemSolve checkpoint uses an uncased tokenizer, whereas ATLOP's
published BERT experiment used cased BERT. A direct backbone comparison should
use the same tokenizer and head, and report any pretraining differences.
Full ATLOP with BERT can be an additional head reference, but is not implemented
by this entrypoint.

## Prepare data

Use the NLP environment. All data and run directories belong outside Git.

```bash
python -m experiments.train_redocred \
  --checkpoint /path/to/local/mlm/final \
  --data /path/to/redocred \
  --prepare-only --download
```

Downloads use a pinned revision of the official Re-DocRED repository.
For offline preparation, put `train_revised.json`, `dev_revised.json`, and
`test_revised.json` in `/path/to/redocred/raw/` and omit `--download`.
The relation mapping is derived from training relations only. It preserves
the dataset's relation names, with NA/TH at index zero. Source and processed
file hashes, tokenizer hash, document counts, and lengths are recorded.

## Train and evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.train_redocred \
  --checkpoint /path/to/local/mlm/final \
  --data /path/to/redocred --output /path/to/run \
  --micro-batch 1 --workers 2
```

For two GPUs, use `python -m torch.distributed.run --standalone
--nproc_per_node=2 -m experiments.train_redocred` with the same arguments. The microbatch and GPU
count determine accumulation while keeping the recipe's global batch.
Accumulated/DDP loss is normalized by the number of valid entity pairs,
including a partial final batch. Workers persist, and `drop_last=False`.
Validation loss is recomputed over the gathered valid pairs, so it does not
depend on evaluation batch size or distributed sampler padding.
The initial entrypoint uses eager BF16 AMP and fused AdamW on CUDA; it does not
enable compile or CUDA Graph for these variable-sized documents.

An ordinary local BERT encoder/MLM checkpoint uses the identical command and
head. A separate prepared directory is needed if its tokenizer differs.

Each epoch evaluates development data and saves a Trainer checkpoint.
`best/` exports the complete selected model, tokenizer, and recipe.
`selection.json` records the selected checkpoint and development F1.
`--resume /path/to/run/checkpoint-N` restores optimizer, scheduler, RNG,
and Trainer state; keep the same source checkpoint and arguments.

Training never evaluates the test set. Evaluate the selected export explicitly:

```bash
python -m experiments.train_redocred \
  --checkpoint /path/to/run/best \
  --config /path/to/run/best/recipe.json \
  --data /path/to/redocred --output /path/to/evaluation \
  --evaluate-only --micro-batch 1
```

Reports contain F1, **annotated-training-fact Ign F1**, precision/recall,
intra-sentence F1, inter-sentence F1, and F1 for minimum entity sentence distance
greater than five. Empty subsets include a zero support count. Scores are
fractions, not percentages. Predictions are exported in the official JSON
format. Ign F1 keeps the official recall and modifies precision; it is not F1
after simply deleting seen facts. There is no distant-supervision training,
evidence prediction, or distant-training-fact Ign F1 in this experiment.

## Sources

- [ATLOP paper](https://arxiv.org/abs/2010.11304): no-LOP ablation, entity
  markers, mention aggregation, grouped bilinear head, adaptive thresholding,
  and starting BERT recipe.
- [DREEAM](https://github.com/YoumiMa/dreeam/tree/476a7004521d5f9b7bb3be66fc6678495b3cf291):
  MIT implementation of the AT loss and classifier; no evidence supervision
  or attention pooling is used.
- [Re-DocRED](https://github.com/tonytan48/Re-DocRED/tree/ccfb54f5ddf5836027c87badda10f6dfc56efaac):
  revised annotations and official relation-metric conventions.

Source adaptations and license copies are listed in `THIRD_PARTY_NOTICES.md`.
