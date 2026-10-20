# Third-party notices

## Re-DocRED relation extraction

`experiments/redocred.py` adapts the adaptive-threshold loss and grouped
bilinear classifier from [DREEAM](https://github.com/YoumiMa/dreeam), commit
`476a7004521d5f9b7bb3be66fc6678495b3cf291`, Copyright (c) 2023 YoumiMa,
under MIT (`licenses/DREEAM-MIT.txt`). Changes: the published ATLOP no-LOP
variant, FP32 loss/pooling, non-mutating labels, pair padding/accumulation,
and chunked pair classification.

Relation F1 and annotated-train Ign F1 follow the evaluator in
[Re-DocRED](https://github.com/tonytan48/Re-DocRED), commit
`ccfb54f5ddf5836027c87badda10f6dfc56efaac`, Copyright (c) 2023 tonytan48,
under MIT (`licenses/Re-DocRED-MIT.txt`). The implementation retains the
official ignored-precision denominator and unchanged recall, deduplicates
predictions, and handles empty distance subsets. It does not report evidence
F1 or the separate distant-training-fact metric. Data are downloaded outside
the repository; their upstream revision and local file hashes are recorded.

Entity markers and logsumexp mention aggregation follow
[ATLOP](https://arxiv.org/abs/2010.11304). Long documents use full-coverage
overlapping BERT windows with averaged hidden states, rather than the
upstream first/last-window attention reconstruction.

## Rotary value encoding

The optional content transport is an independent adaptation of
[RoVE](https://arxiv.org/abs/2606.11275), following its
[standard value/output rotation](https://github.com/AGarciaCast/RoVE/blob/8e3eb84efb7c920289aba776fbdb189f93b79e01/model/pos_methods/rove.py)
and evaluation-time YaRN frequency ramp. No RoVE source files are vendored.
The Triton rotation uses the adjacent-pair reshape/split/join organization in
[FlashAttention rotary](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py)
(BSD-3-Clause), with independently written indexing and fusion into MemSolve's
existing statistics/readout/gradient kernels. These references describe
algorithmic provenance; they do not change the repository's license.

## ViT-Adapter dense vision integration

`integrations/vit_adapter.py` adapts `adapter_modules.py` from
[Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV), commit
`0e0e38a6fd4475ae3817e14420ab85fab08d9097`, Copyright Shanghai AI Lab,
under Apache-2.0. Changes: MemSolve block/layout interface, non-reentrant
checkpointing, meshgrid indexing, and removal of the unused CLS path.

`integrations/deform_attention.py` adapts the Deformable DETR projections,
initialization and CPU oracle distributed in
[ViT-Adapter](https://github.com/czczup/ViT-Adapter), commit
`fd64a5190f9f4530e8e5961cc87b2a3ad591190b`, Copyright 2020 SenseTime,
under Apache-2.0. CUDA dispatch uses MMCV's compiled operator; sampling
retains the upstream FP32 precision island. The original source credits
Deformable-Convolution-V2-PyTorch. The layer-decay grouping follows the
BEiT convention used by ViT-Adapter/Vision-RWKV, adapted to MMEngine.

A copy of Apache-2.0 is in `licenses/Apache-2.0.txt`. These additions do
not relicense the rest of the repository. The current heads, dataset
pipelines, training loops and native deformable sampling are supplied by
OpenMMLab rather than copied into this repository.

The sliding inference loop in `integrations/openmmlab.py` is adapted from
OpenMMLab MMSegmentation 1.2.2 `EncoderDecoder`, Apache-2.0, with crop-local
mask slicing and without mutation of the caller's image metadata.

## Projection GEMM

The grouped program scheduling, contiguous-load hints, and store-index
rematerialization in `memsolve/ball/reference.py::_biased_gemm_kernel` are
adapted from PyTorch Inductor's
[`triton_mm.py.jinja`](https://github.com/pytorch/pytorch/blob/9664c41ccfc87a9661cb2b177985131948306876/torch/_inductor/kernel/templates/triton_mm.py.jinja),
under the PyTorch BSD-style license reproduced in `licenses/PyTorch-BSD.txt`.
Changes: FP32 bias and token-mask epilogue, fixed measured tile schedules,
and masked loads for matrices smaller than a tile. The accumulator remains
FP32 until the output store. FLA, Triton's persistent matmul tutorial and
FlagGems were benchmark references; their candidate implementations are not
vendored in the package.

## Long-sequence sufficient statistics

The 64-token accumulation loop inside a 512-token statistics chunk in
`memsolve/ball/cuda.py` is independently written, with scheduling informed by
[Mamba SSD chunk-state](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_state.py).
No Mamba source is copied. MemSolve retains its own Gram, cross-statistic,
RoVE compensation and gradient equations with FP32 accumulators. FLA separate
VJPs and PyTorch Inductor shared-A BMM informed benchmark-only candidates;
those implementations are not included in the package.
