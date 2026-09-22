# LSSO

LSSO turns contextual adaptation into a solved global operator. Each sample
generates a compact strictly accretive system; one direct solve returns its
equilibrium without an inner learning rate, unrolling, fixed-point iteration,
or a materialized token-to-token map.

The construction scales linearly with sequence length at fixed rank. Its
compact parameterization spans the open real matrix ball, and every realized
token mixer, after its sample-conditioned quantities are fixed, is a strict
L2 contraction around a learned scalar complement. This is a per-mixer
certificate, not an end-to-end Lipschitz claim for the surrounding network.

The operator has no internal rotation; learned position embeddings belong to
the surrounding model. DYNAMIC, STATIC and ZERO have native CUDA inference
and an analytic first-order backward. On the measured RTX 5070 Ti long-sequence
workloads, the complete mixer reaches up to 1.79x the forward speed and 2.30x
the forward-backward speed of PyTorch MHA backed by Flash SDPA. Those historical measurements use an earlier operator (native contract 6);
current ABI-11 changes have not rerun
those formal panels. See [result provenance](results/README.md).

Read the current paper: **[LSSO: Solving Contextual Adaptation with Certified
Global Mixing](paper/main.pdf)**. The LaTeX source is in
[`paper/main.tex`](paper/main.tex).

## Documentation

| Topic | Guide |
| --- | --- |
| Mathematics, dtypes and checkpoint contracts | [Core contract](docs/CORE_CONTRACT.md) |
| Code ownership and execution paths | [Architecture](docs/ARCHITECTURE.md) |
| Native build, Triton JIT and GPU validation | [CUDA contract](docs/CUDA_CONTRACT.md) |
| ImageNet pretraining and checkpoint transfer | [Plain ViT³ training workflow](docs/IMAGENET_VIT3.md) |
| COCO detection/instance segmentation and ADE20K semantic segmentation | [Dense downstream protocols](docs/DOWNSTREAM_PROTOCOLS.md) |
| Assembly101 data and validation protocol | [Assembly101 workflow](docs/ASSEMBLY101.md) |
| GenomicBenchmarks and LRA | [Sequence experiments](docs/SEQUENCE_EXPERIMENTS.md) |
| Supported ablations | [Ablation guide](docs/ABLATIONS.md) |
| Test commands and historical evidence | [Tests](tests/README.md), [results](results/README.md) |

## Quick start

~~~python
import torch
from lsso import LSSO, LSSOConfig

layer = LSSO(LSSOConfig(dim=192, num_heads=3, rank=16)).cuda()
x = torch.randn(8, 65, 192, device="cuda")
y = layer(x)
~~~

The native CUDA path covers DYNAMIC, STATIC and ZERO
operators, with rank 16, 32, 48, or 64 and any positive practical head dimension.
It accepts the current operator's optional boolean `valid_mask`  without falling back to another implementation. It
targets Ampere SM80 and newer supported NVIDIA architectures.
Build with CMake 3.25+, a C++20 compiler, and MathDx 26.06.1 (CUDA 13 package).
Build its strict per-SM artifacts with `tools/build_cuda.sh`, then load the
artifact for the device before requesting it:

~~~python
from lsso.ball import cuda

cuda.load(device=x.device)
x = x.to(torch.bfloat16)  # Native inputs must be FP16 or BF16.
y = layer(x, implementation="cuda")
~~~

The current source requires `torch==2.14.0+cu132`, CUDA `13.2`, native
contract `11`, and Linux x86_64 for its native runtime. Released v0.6.3 wheels
must not be mixed with this newer source contract; build matching native
artifacts from this checkout:

~~~bash
python -m pip install --index-url https://download.pytorch.org/whl/cu132 \
  'torch==2.14.0+cu132'
python -m pip install -e .
PYTHON=python bash tools/build_cuda.sh
~~~

A matching runtime wheel can also be built with `tools/package_cuda_runtime.py`.
It packages seven architecture-specific artifacts (SM80 through supported
Blackwell targets). The toolchain migration was validated on SM80 (A800); earlier optimization
checks used SM120. Other targets require validation with the new toolchain.
Biased FP16/BF16 projections additionally use the Triton runtime supplied by
Linux CUDA PyTorch; their first call JIT-compiles a fused GEMM. Warm up on the
capture stream before CUDA Graph capture. CPU reference imports do not require
Triton. See [the CUDA contract](docs/CUDA_CONTRACT.md) for precision and build
coverage.

The current model checkpoint contract is version 13. A v0.6.3 checkpoint with
version 11 requires an explicit, validated migration; `strict=False` does not
bypass the contract check. Keep the original checkpoint when transferring
pretrained weights.

Source checkouts still prefer `build/cuda/lib/` for development; an explicit
`LSSO_CUDA_LIBRARY` remains available for a manually built artifact.

The timm adapter keeps the backend explicit for the ImageNet and downstream
workflows.

Supported ablations include DYNAMIC, STATIC, and ZERO core ownership, skew coupling, and the scalar complement. See
[`docs/CORE_CONTRACT.md`](docs/CORE_CONTRACT.md) for the canonical mathematical
and numerical contract.

The public CUDA runtime uniformly uses a no-frame Triton/PyTorch schedule.
It shares token statistics/readout/VJPs
across Dynamic, Static and Zero and avoids storing token-sized P. No extra flag
is needed; see [CUDA scheduling and numerical policy](docs/CUDA_CONTRACT.md#default-cuda-scheduling-without-an-explicit-frame).

## CUDA wall clock

The table below measures the complete mixer boundary, including input and
output projections, at `B=64, D=256, H=8`, FP16 AMP, and rank 32 for LSSO.
Values are synchronized host wall-clock milliseconds per batch, reported as
the median of seven repetitions of 100 iterations after 50 warmup iterations.
MHA uses `nn.MultiheadAttention(..., need_weights=False)` and profiler-confirmed
Flash SDPA. Forward-backward includes the scalar loss and gradient reset.

| Length | LSSO native forward | MHA forward | LSSO native forward-backward | MHA forward-backward |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 2.539 | 2.131 | **7.512** | 8.487 |
| 2048 | **5.577** | 5.888 | **15.260** | 22.940 |
| 4096 | **10.135** | 18.173 | **29.160** | 67.195 |

These are single-device RTX 5070 Ti measurements under WSL2, PyTorch
2.11.0+cu128, and CUDA 12.8. They are not a cross-architecture or end-to-end
training-throughput claim. The paper appendix records the full protocol and
the comparison with the canonical PyTorch LSSO implementation.

## Sequence results

GenomicBenchmarks and LRA use the same sequence runner, validation-based
checkpoint selection, and one final test evaluation. Formal GenomicBenchmarks
runs compare MHA and LSSO in the same `D128/L4/H4` shell over three seeds. LSSO
is higher on 7 of 8 tasks; its unweighted macro accuracy is 78.53%, versus
75.37% for MHA (+3.16 percentage points).

| GenomicBenchmarks panel | MHA | LSSO |
| --- | ---: | ---: |
| 8-task macro test accuracy | 75.37 | **78.53** |
| Tasks with higher accuracy | 1/8 | **7/8** |

The formal LRA panel reports three-seed, validation-selected test accuracy for
the current task-specific shared shells:

| Task | LSSO accuracy (mean +/- std) |
| --- | ---: |
| ListOps | 37.60 +/- 0.65 |
| Text | 65.19 +/- 0.46 |
| Retrieval | 82.88 +/- 0.08 |
| Pathfinder-32 | 78.79 +/- 0.34 |

Published LRA values are split into two architectural groups in the paper:
general-purpose global mixers and efficient-attention approximations, and
models centered on structured recurrence or state-space dynamics. This is a
mechanism-level grouping, not a claim that every implementation in the first
group is free of auxiliary local operations. Within the first group, LSSO
reports the strongest displayed Retrieval and Pathfinder results and is within
0.38 points on ListOps and 0.71 points on Text of the strongest displayed
result. Its unweighted average over those four tasks is the highest in the
group. These are still cross-paper results with different shells and training
protocols, so they provide context rather than an apples-to-apples leaderboard.

After placing the datasets at the roots recorded in the config files, the full
three-seed panels can be reproduced on Linux with the CUDA runtime loaded:

~~~bash
DNA_DATA_ROOT=/path/to/genomic_benchmarks
LRA_DATA_ROOT=/path/to/lra
SEQUENCE_CACHE=/path/to/sequence_cache

genomic_tasks=(
  dummy_mouse_enhancers_ensembl
  demo_coding_vs_intergenomic_seqs
  demo_human_or_worm
  human_enhancers_cohn
  human_enhancers_ensembl
  human_ensembl_regulatory
  human_nontata_promoters
  human_ocr_ensembl
)
for task in "${genomic_tasks[@]}"; do
  batch_args=(--batch-size 128 --grad-accum 1)
  if [[ "$task" == dummy_mouse_enhancers_ensembl ]]; then
    batch_args=(--batch-size 64 --grad-accum 2)
  fi
  for mixer in mha lsso; do
    for seed in 0 1 2; do
      python -m experiments.train_transformers \
        --config experiments/configs/genomic.toml \
        --task "$task" --mixer "$mixer" --seed "$seed" --formal \
        "${batch_args[@]}" \
        --data-root "$DNA_DATA_ROOT" --cache-root "$SEQUENCE_CACHE" \
        --output "runs/sequence/genomic/$task/$mixer/s$seed"
    done
  done
done

for task in listops text retrieval pathfinder; do
  for seed in 0 1 2; do
    python -m experiments.train_transformers \
      --config experiments/configs/lra.toml \
      --task "$task" --mixer lsso --seed "$seed" --formal \
      --data-root "$LRA_DATA_ROOT" --cache-root "$SEQUENCE_CACHE" \
      --output "runs/sequence/lra/$task/lsso/s$seed"
  done
done
~~~

The GenomicBenchmarks runner also supports `linear_transformer`, `performer`,
`nystromformer`, `cosformer`, and `rebased` through the same `--mixer`
argument. They use the same data protocol and encoder shell as the matched
MHA/LSSO panel and run as ordinary PyTorch CUDA tensor programs. Their
implementations and upstream parity tests are included, but formal three-seed
results have not yet been reported.

`nystromformer` and `rebased` are frozen as numerical-reference implementations
and excluded from formal runs until suitable bidirectional Triton kernels are
available.

The runner pins source and split provenance, keeps the compared DNA backbones
matched, and refuses dirty or mutable formal inputs. See
[docs/SEQUENCE_EXPERIMENTS.md](docs/SEQUENCE_EXPERIMENTS.md) for per-task
results, exact recipes, data layout, and protocol boundaries. The final
per-seed metrics, dataset fingerprints, certificate summaries, and wall-clock
protocol are published as machine-readable artifacts in [`results/`](results/README.md).

ImageNet-1K uses plain ViT³-derived T/S/B training (300 epochs, AdamW, no EMA)
with LSSO ranks 16/32/48; see [docs/IMAGENET_VIT3.md](docs/IMAGENET_VIT3.md). COCO 2017
Mask R-CNN + FPN 3x and ADE20K UperNet 160k use the shared dense DeiT III
backbone and explicit padded-image masking. Their protocol provenance and
launch commands are in [docs/DOWNSTREAM_PROTOCOLS.md](docs/DOWNSTREAM_PROTOCOLS.md).

## Contributing

Contributors using AI-assisted development tools must follow the repository
rules in [`AGENTS.md`](AGENTS.md).
