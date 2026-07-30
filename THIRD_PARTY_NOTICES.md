# Third-party notices

## Nyströmformer LRA recipe

`experiments/train_lra.py` adapts the embedding initialization, two-layer
pre-LayerNorm backbone, single/paired ReLU classification heads, and fixed-step
AdamW/OneCycle recipe from [Nyströmformer](https://github.com/mlpen/Nystromformer),
commit `56893131bf3fa99b5a2d3ab452a591dca722529a`, under Apache-2.0 (the same
license as this repository). Source files: `LRA/code/model.py`,
`model_wrapper.py`, `lra_config.py`, and `run_tasks.py`. Changes: MemSolve replaces
the attention module, BF16/FP16 autocast and selective compilation use current
PyTorch, shuffled sample streams resume by optimizer step, and evaluation
includes incomplete final batches. Training data and model outputs remain
outside the repository.

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
rematerialization in `memsolve/ball/cuda.py::_biased_gemm_kernel` are
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

## LRA attention adapters

`experiments/sequence_attention.py` adapts the following operator algorithms:

- Performer positive orthogonal random features from
  [performer-pytorch](https://github.com/lucidrains/performer-pytorch),
  `performer_pytorch/performer_pytorch.py`, commit
  `fc8b78441b1e27eb5d9b01fc738a8772cee07127`, MIT, Copyright 2020 Phil Wang.
  License: `licenses/Performer-MIT.txt`.
- Nyströmformer's landmark attention and six inverse iterations from
  [Nystromformer](https://github.com/mlpen/Nystromformer),
  `LRA/code/attention_nystrom.py`, commit
  `56893131bf3fa99b5a2d3ab452a591dca722529a`, Apache-2.0
  (`licenses/Apache-2.0.txt`). The initialization uses per-head matrix norms.
- Pure Agent Attention and MiTA from
  [MiTA](https://github.com/QishuaiWen/MiTA), `mita/agent_attention.py` and
  `mita/mita_attention.py`, commit `95460f703ef5bd5d814c45a6ff52d9582207d01c`,
  MIT, Copyright 2026 Qishuai Wen (`licenses/MiTA-MIT.txt`). MiTA routing follows
  the released algorithm, including its additional agent-key scaling. Our
  grouped attention concatenates agent and routed memories before calling
  PyTorch's public `varlen_attn`, instead of copying MiTA/MoBA's custom backward.
- Polarity-aware feature maps from
  [PolaFormer](https://github.com/ZacharyMeng/PolaFormer),
  `models/pola_swin.py`, commit `167f48f70b0992f000c98aebe48198b230a72f4b`,
  MIT, Copyright 2025 ZacharyMeng (`licenses/PolaFormer-MIT.txt`).

Adapters add explicit padding masks, masked landmark pooling and optional
value-path RoVE. Sensitive feature maps, denominators and Nyström inverse
iterations use FP32. These are pure mixer controls in a common LRA encoder:
Agent's spatial bias/convolution and PoLA's convolution/gate/extra position
embedding are omitted. This follows the pure-operator comparison used by MiTA,
not the complete vision architectures. Random features stay fixed throughout
an experiment. For 2D inputs, 128 Agent/MiTA routers use an 8-by-16 pooled grid.


## Conventional LRA attention baselines

- `experiments/sequence_attention.py` implements the noncausal ELU+1 KV reduction
  from Idiap fast-transformers (`2ad36b97e64cb93862937bd21fcc9568d989561f`,
  `fast_transformers/attention/linear_attention.py`). Copyright (c) 2020 Idiap
  Research Institute; Angelos Katharopoulos and Apoorv Vyas. MIT license retained
  in `licenses/FastTransformers-MIT.txt`. Padding and value-path RoVE use the
  shared LRA adapter; feature statistics and denominators accumulate in FP32.
- Local attention calls PyTorch FlexAttention with a cached sliding-window block
  mask. The 256-key window spans offsets -128 through +127, clipped at boundaries.
- BigBird uses the installed Hugging Face Transformers `BigBirdBlockSparseAttention`
  planning methods (Apache-2.0) at setup, with 64-token blocks, three random
  blocks, three local blocks and two global blocks. Its fixed per-layer training
  plan is also used for evaluation. PyTorch FlexAttention executes the sparse
  pattern; any repeated key blocks retain their softmax multiplicity. Numerical
  tests compare outputs and gradients against the upstream sparse operation.
  No Transformers source is vendored. The experiment manifest records its version.
