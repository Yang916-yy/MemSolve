from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn as nn

from memsolve import MemSolve, MemSolveConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a complete MemSolve or MHA mixer block on one CUDA device."
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--spatial-shape", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph", action="store_true", help="Capture a complete gradient-accumulation cycle.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--autotune", action="store_true",
                        help="Tune MemSolve token launches during eager warmup, before Graph capture.")
    parser.add_argument("--launch-report", type=Path,
                        help="Write selected launch configs, resource usage and tuning times as JSON.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Microbatches per parameter-gradient reset.",
    )
    parser.add_argument("--operator", choices=("memsolve", "mha"), default="memsolve")
    parser.add_argument("--mode", choices=("forward", "train"), default="train")
    parser.add_argument(
        "--implementation",
        choices=("reference", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


class _BenchmarkMixer(nn.Module):
    """Present MemSolve and MHA through one tensor-only benchmark surface."""

    def __init__(
        self,
        mixer: MemSolve | nn.MultiheadAttention,
        *,
        operator: str,
        implementation: str,
        spatial_shape: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        self.mixer = mixer
        self.operator = operator
        self.implementation = implementation
        self.spatial_shape = spatial_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.operator == "memsolve":
            assert isinstance(self.mixer, MemSolve)
            return self.mixer(
                x,
                implementation=self.implementation,
                spatial_shape=self.spatial_shape,
            )
        assert isinstance(self.mixer, nn.MultiheadAttention)
        output, _ = self.mixer(x, x, x, need_weights=False)
        return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the benchmark requires a CUDA device")
    if args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative; steps and repeats must be positive")
    if args.grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if (args.autotune or args.launch_report) and (args.operator != "memsolve" or args.implementation != "cuda"):
        raise ValueError("launch tuning/reporting requires MemSolve implementation=cuda")
    if args.autotune and args.warmup < 1:
        raise ValueError("autotune requires at least one eager warmup step")
    if args.steps % args.grad_accum:
        raise ValueError("steps must be divisible by grad_accum")
    spatial_shape = None if args.spatial_shape is None else tuple(args.spatial_shape)
    if spatial_shape is not None and (min(spatial_shape) <= 0 or spatial_shape[0] * spatial_shape[1] != args.length):
        raise ValueError("spatial shape must be positive with H*W=length")
    if args.graph and args.operator == "memsolve" and args.implementation != "cuda":
        raise ValueError("MemSolve graph benchmarking requires implementation=cuda")

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    torch.backends.cuda.matmul.fp32_precision = "tf32" if args.tf32 else "ieee"
    torch.backends.cudnn.fp32_precision = "tf32" if args.tf32 else "ieee"
    if args.operator == "memsolve":
        mixer: MemSolve | nn.MultiheadAttention = MemSolve(
            MemSolveConfig(
                dim=args.dim,
                num_heads=args.heads,
                rank=args.rank,
                bias=args.bias,
                qk_conv_dim=1 if spatial_shape is None else 2,
            )
        ).to(device)
    else:
        mixer = nn.MultiheadAttention(
            args.dim,
            args.heads,
            dropout=0.0,
            bias=args.bias,
            batch_first=True,
        ).to(device)
    layer = _BenchmarkMixer(
        mixer,
        operator=args.operator,
        implementation=args.implementation,
        spatial_shape=spatial_shape,
    )
    if args.mode == "forward":
        layer.eval()
    else:
        layer.train()
    x = torch.randn(
        args.batch,
        args.length,
        args.dim,
        device=device,
        dtype=dtype,
    )
    if args.mode == "train":
        x.requires_grad_(True)
    if args.operator == "memsolve" and args.implementation == "cuda":
        from memsolve.ball import cuda

        cuda.load(device=device)

    def step() -> None:
        if args.mode == "forward":
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=dtype is not torch.float32,
                ):
                    layer(x)
            return
        with torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=dtype is not torch.float32,
        ):
            (layer(x).float().square().mean() / args.grad_accum).backward()

    def clear_parameter_gradients() -> None:
        layer.zero_grad(set_to_none=True)

    def clear_input_gradient() -> None:
        if x.grad is not None:
            x.grad = None

    clear_parameter_gradients()
    with cuda.autotune() if args.autotune else nullcontext():
        for step_index in range(args.warmup):
            clear_input_gradient()
            step()
            if (step_index + 1) % args.grad_accum == 0:
                clear_parameter_gradients()
    clear_parameter_gradients()
    clear_input_gradient()
    torch.cuda.synchronize()

    graph = None
    if args.graph:
        def cycle() -> None:
            clear_parameter_gradients()
            for _ in range(args.grad_accum):
                clear_input_gradient()
                step()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                cycle()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cycle()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    gpu_samples: list[float] = []
    for _ in range(args.repeats):
        if graph is None:
            clear_parameter_gradients()
            clear_input_gradient()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start = time.perf_counter()
        for step_index in range(args.steps):
            if graph is not None:
                if step_index % args.grad_accum == 0:
                    graph.replay()
                continue
            clear_input_gradient()
            step()
            if (step_index + 1) % args.grad_accum == 0:
                clear_parameter_gradients()
        end_event.record()
        torch.cuda.synchronize()
        samples.append(1000.0 * (time.perf_counter() - start) / args.steps)
        gpu_samples.append(start_event.elapsed_time(end_event) / args.steps)

    properties = torch.cuda.get_device_properties(device)
    if args.launch_report:
        args.launch_report.parent.mkdir(parents=True, exist_ok=True)
        args.launch_report.write_text(json.dumps({
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "launches": cuda.launch_report(),
        }, indent=2) + "\n")
    implementation = args.implementation if args.operator == "memsolve" else "torch_mha"
    rank = str(args.rank) if args.operator == "memsolve" else "na"
    position = "rope_2d" if args.operator == "memsolve" and spatial_shape is not None else "external"
    print(
        f"device={properties.name} sm={properties.major}.{properties.minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"operator={args.operator} implementation={implementation} mode={args.mode} "
        f"dtype={args.dtype} tf32={args.tf32} graph={args.graph} bias={args.bias} "
        f"batch={args.batch} grad_accum={args.grad_accum} length={args.length} "
        f"dim={args.dim} heads={args.heads} rank={rank} position={position} "
        f"median_ms={statistics.median(samples):.3f} "
        f"median_gpu_ms={statistics.median(gpu_samples):.3f} "
        f"min_ms={min(samples):.3f} max_ms={max(samples):.3f} "
        f"peak_allocated_mib={torch.cuda.max_memory_allocated() / 2**20:.2f}"
    )


if __name__ == "__main__":
    main()
