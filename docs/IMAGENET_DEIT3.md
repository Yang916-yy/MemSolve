# ImageNet DeiT III

> Current source uses model contract 13 and native ABI 9. External position
> embeddings belong to the surrounding model. Historical measurements retain
> their recorded source versions and are not new-source results.


`experiments/train_imagenet.py` launches the shared `experiments/imagenet.py`
workflow, which implements the ImageNet-1K recipes from the
[official DeiT III repository at commit 7e160fe43f0252d17191b71cbb5826254114ea5b](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/README_revenge.md).
The only supported data contract is the authenticated ModelScope
[`timm/imagenet-1k-wds`](https://modelscope.cn/datasets/timm/imagenet-1k-wds)
release. `--data-root` must contain its `_info.json`, 1,024 training tar
shards, and 64 validation tar shards; no ImageFolder extraction is used. The
runner pins the currently verified `_info.json` SHA-256, rejects a partial or
different layout, and has rank zero scan tar headers for exactly one
`.jpg`/`.cls`/`.json` record triple per manifest sample before training begins.
This structural preflight is not a payload checksum: JPEG decoding and label
range checks remain in the streaming reader. ModelScope access requires
accepting ImageNet's research/education terms.
For a single-node notebook workflow that validates the environment, installs
the pinned native runtime, launches a run, and exposes a refreshable monitor,
see [`notebooks/imagenet_launcher.ipynb`](../notebooks/imagenet_launcher.ipynb).

The pretraining configurations preserve the public commands: 800 epochs,
cosine decay with five warmup epochs, three-view repeated augmentation,
3-Augment, Mixup `0.8`, CutMix `1.0`, BCE targets, EMA `0.99996`, and Apex
FusedLAMB. The S model uses 224px input; B and L use 192px before the official
20-epoch 224px AdamW fine-tuning phase. That phase retains batch-mode Mixup
`0.8` and CutMix `1.0`, while switching to RandAugment, disabling repeated
augmentation and BCE targets, and using label smoothing `0.1`. Per-GPU batches
and learning rates are not linearly rescaled, matching `--unscale-lr` in the
source commands.

The runner keeps the formal optimizer update independent of the available GPU
count. `batch_size` is a physical per-GPU batch; it must be a multiple of the
configured virtual Mixup group. The launcher derives accumulation so that
`world_size * batch_size * grad_accum` equals the fixed global effective batch.
Each physical batch is split into independent batch-mode Mixup/CutMix groups.
Repeated augmentation buffers three physical batches of source groups, then
interleaves their three views across nine output physical batches, so neither a
physical batch nor a Mixup/CutMix group contains two views of the same image.
Epochs are truncated to whole effective-batch updates.

Training shards are globally permuted from the fixed run seed at each epoch,
then each rank assigns a finite, unique source-record quota to every worker.
Workers use a bounded 8,192-sample WebDataset shuffle buffer, initially filled
with 2,048 samples. They never cycle a shard within an epoch; their quotas
align with virtual Mixup-group boundaries.
Repeated augmentation buffers a bounded window of source groups before image
decoding, then emits one independently transformed view from every group before
cycling to their next views. The fixed update schedule can truncate only the
final rank-local group at a physical-batch boundary. Views are deliberately
local to the streaming rank; this preserves batch-mode Mixup boundaries and the
fixed update schedule, but is not an index-identical replay of the retired
ImageFolder sampler's cross-rank view placement. This behavior is recorded in
each checkpoint's data contract.

Validation retains the public DeiT behavior: every rank reads the complete
50,000-example validation split and reductions preserve the metric. Only its
workers are strided over the 64 validation shards. `train_workers` and
`val_workers` are independent per-rank settings (defaults: 10 and 4), and
workers are recreated at each epoch boundary so transform RNG can replay after
an exact resume. The streaming readers use one prefetched physical batch per
worker to bound host memory.

| Phase | Effective global batch | Virtual Mixup group | Updates per ImageNet-1K epoch |
| --- | ---: | ---: | ---: |
| S/B pretraining | 2048 | 256 | 625 |
| L pretraining | 2048 | 64 | 625 |
| B/L 224px fine-tuning | 512 | 64 | 2502 |

`--grad-accum` is optional and only accepted when it exactly matches this
contract. It is useful for asserting a launch plan, not for changing the
effective batch. A larger single-GPU physical batch remains `deit3-derived`
when the resolved plan is exact. For example, one H100 can use 512 images for
S/B pretraining or 128 for L pretraining:

```bash
torchrun --standalone --nproc_per_node=1 experiments/train_imagenet.py \
  --tier small --phase pretrain --data-root /datasets/imagenet-1k-wds \
  --output runs/imagenet/deit3_small_h100 --batch-size 512

torchrun --standalone --nproc_per_node=1 experiments/train_imagenet.py \
  --tier large --phase pretrain --data-root /datasets/imagenet-1k-wds \
  --output runs/imagenet/deit3_large_h100 --batch-size 128
```

The model geometry is:

| Tier | Width | Depth | Heads | LSSO rank | Pretraining input |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 384 | 12 | 6 | 32 | 224 |
| B | 768 | 12 | 12 | 48 | 192 |
| L | 1024 | 24 | 16 | 64 | 192 |

Install `timm`, `torchvision`, `webdataset`, and Apex with FusedLAMB in the experiment
environment. The launcher fails rather than silently changing the published
optimizer. `--allow-lamb-fallback` is available only for explicitly marked
non-fused diagnostic runs.

```bash
python -m pip install -e '.[vision]'
python -m pip install modelscope
modelscope login
modelscope download timm/imagenet-1k-wds --repo-type dataset \
  --local-dir /datasets/imagenet-1k-wds
```

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_imagenet.py \
  --tier small --phase pretrain --data-root /datasets/imagenet-1k-wds \
  --output runs/imagenet/deit3_small
```

```bash
torchrun --standalone --nproc_per_node=8 experiments/train_imagenet.py \
  --tier base --phase pretrain --data-root /datasets/imagenet-1k-wds \
  --output runs/imagenet/deit3_base_192

torchrun --standalone --nproc_per_node=8 experiments/train_imagenet.py \
  --tier base --phase finetune_224 --data-root /datasets/imagenet-1k-wds \
  --init-checkpoint runs/imagenet/deit3_base_192/checkpoint_best.pt \
  --output runs/imagenet/deit3_base_224
```

The large 192px pretraining command uses 32 GPUs in the official recipe. On
each node, set the usual `torchrun` rendezvous arguments explicitly:

```bash
torchrun --nnodes=4 --node_rank="$NODE_RANK" --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  experiments/train_imagenet.py --tier large --phase pretrain \
  --data-root /datasets/imagenet-1k-wds --output runs/imagenet/deit3_large_192
```

Every output directory contains `metadata.json`, append-only `metrics.jsonl`,
`checkpoint_last.pt`, and `checkpoint_best.pt`. Checkpoints are written through
a same-directory temporary file, `fsync`, and atomic replace, so a preemption
cannot replace a complete checkpoint with a partial one. Resume only with the
same tier, phase, model contract, recipe, WebDataset manifest and streaming
contract, resolved batching plan, device type, and world size; the runner
rejects mismatches, including a changed accumulation factor. At each saved epoch
boundary it records every rank's Python, NumPy,
Torch CPU/CUDA, and train/validation worker-generator states. Workers are
recreated at epoch boundaries, so this replays the next epoch's augmentation
and data-loader random streams in the same environment. It does not promise
bitwise identity across changed GPU, CUDA, PyTorch, or kernel environments.
Each current checkpoint carries a canonical SHA-256 digest of its contract.
This streaming data contract uses ImageNet checkpoint format 5, so older ImageNet
runner checkpoints are intentionally not resumable. Fine-tuning and downstream
loading validate the current checkpoint envelope, then compare only the
compatible model and operator contract before loading model tensors.
`metadata.json` also records the source commit
and whether the worktree was dirty before the output directory was created; a
dirty marker does not identify the uncommitted patch.

The shared backbone boundary is intentionally narrow:

```python
integrations.timm.create_lsso_deit3(
    image_size, patch_size, num_classes, embed_dim, depth, num_heads, rank,
    mlp_ratio, core_mode, bias, implementation, drop_path_rate,
    layer_scale_init_value, norm_eps, no_embed_class=True,
)
```

It must build the current LSSO DeiT III model with a learned 2D patch position
table and no CLS position embedding. The runner owns only data, optimizer,
scheduler, checkpoint, and distributed-training concerns.

## Current source and pretrained-weight transfer

Three version numbers describe different boundaries:

| Boundary | Current version | Checked by |
| --- | ---: | --- |
| ImageNet checkpoint envelope | 5 | `validate_checkpoint_contract`, including its digest |
| Per-layer LSSO model state | 12 | `_extra_state` during model loading |
| Native CUDA runtime | 8 | `cuda.load` and native ABI checks |

Passing the ImageNet envelope check does not make an older per-layer model
contract compatible. In particular, v0.6.3 model-contract-11 weights are not
automatically accepted by current source. No checkpoint migration command is
implemented. Preserve the original checkpoint, inspect its geometry and
contracts, and validate an explicit migration before fine-tuning or downstream
loading; changing only the version field is insufficient evidence.

The current operator supports both FP16 and BF16 inputs, but this ImageNet
runner's canonical recipe still requires `train.amp_dtype = 'float16'`.
Operator capability is not a change to the published training recipe.
The notebook's pinned released runtime must be used with matching released
source; it must not be combined with current ABI-9 source. For current source,
follow [the source build instructions](../README.md).

Dense-task initialization uses `--backbone-checkpoint`, not an ImageNet
optimizer resume. It drops the classifier, initializes new pyramid/task-head
parameters, and interpolates the patch position table when needed. See
[downstream setup and data](DOWNSTREAM_PROTOCOLS.md).
