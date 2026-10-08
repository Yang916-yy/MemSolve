from __future__ import annotations

from functools import lru_cache
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import importlib.util

import torch

_SUPPORTED_ARCHITECTURES = frozenset((80, 86, 87, 89, 90, 100, 120))
_CUDA_CONTRACT_VERSION = 19


@dataclass(frozen=True)
class _LaunchConfig:
    tokens: int
    warps: int
    stages: int = 1


_TUNE_LAUNCHES = ContextVar("memsolve_tune_launches", default=False)
# Only scalar metadata is retained: never keep a training tensor or graph alive.
_LAUNCH_PLANS: dict[tuple, dict] = {}


@contextmanager
def autotune():
    """Measure token-kernel configurations during eager forward/backward warmup.

    Use before CUDA Graph capture and preferably on an otherwise idle GPU.
    Results are cached per device and workload for this process. Normal calls
    only validate resources and keep the existing schedule when it fits.
    """
    token = _TUNE_LAUNCHES.set(True)
    try:
        yield
    finally:
        _TUNE_LAUNCHES.reset(token)


def launch_report() -> list[dict]:
    """Return JSON-compatible resource/timing records, without GPU synchronization."""
    from copy import deepcopy

    return deepcopy(list(_LAUNCH_PLANS.values()))


@lru_cache(maxsize=None)
def _device_resources(index: int) -> tuple:
    prop = torch.cuda.get_device_properties(index)
    return (
        index, str(prop.uuid), prop.major, prop.minor,
        prop.shared_memory_per_block_optin,
        prop.shared_memory_per_multiprocessor, prop.regs_per_multiprocessor,
    )


def _launch_candidates(default: _LaunchConfig) -> tuple[_LaunchConfig, ...]:
    # Bounded search, inspired by Mamba-2's Triton launch configurations.
    # Keep arithmetic/dtypes fixed. Smaller tiles provide lower-resource options;
    # two stages are an optional latency-hiding alternative, never mandatory.
    return tuple(dict.fromkeys((
        default,
        _LaunchConfig(default.tokens, 2 if default.warps == 4 else 4),
        _LaunchConfig(max(16, default.tokens // 2), default.warps),
        _LaunchConfig(16, 4),
        _LaunchConfig(default.tokens, default.warps, 2),
    )))


def _launch_token(kind, kernel, device, signature, default, prepare):
    # A tensor on cuda:1 must never reuse cuda:0's resource plan or compile for
    # cuda:0 merely because that device happens to be current in the caller.
    with torch.cuda.device(device):
        return _launch_token_on_device(kind, kernel, device, signature, default, prepare)


def _launch_token_on_device(kind, kernel, device, signature, default, prepare):
    """Select a schedule, then allocate exactly its partial buffers and launch.

    prepare(config) returns (arguments, grid, finish, workspace_bytes). finish
    includes partial reduction, so tuning measures the whole token operation,
    not a kernel that wins by exporting more work to a later reduction.
    Every candidate overwrites its outputs; none updates model parameters.
    """
    from triton.runtime.errors import OutOfResources

    resources = _device_resources(device.index)
    key = (resources, kind, signature)
    plan = _LAUNCH_PLANS.get(key)
    tuning = _TUNE_LAUNCHES.get()
    if plan is not None and (not tuning or plan["tuned"]):
        config = _LaunchConfig(**plan["selected"])
        args, grid, finish, _ = prepare(config)
        kernel[grid](*args, num_warps=config.warps, num_stages=config.stages)
        return finish()

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MemSolve launch configuration is cold: run eager forward/backward "
                "warmup for this shape before CUDA Graph capture"
            )
        records = []
        winner = None
        best_ms = float("inf")
        for config in _launch_candidates(default):
            options = dict(num_warps=config.warps, num_stages=config.stages)
            record = {"config": asdict(config)}
            args = finish = None
            try:
                # Smaller token tiles use less on-chip storage but may need
                # MORE global partial-buffer memory. An oversized tuning
                # candidate must not discard another viable schedule.
                args, grid, finish, workspace = prepare(config)
                record["workspace_bytes"] = workspace
                compiled = kernel.warmup(*args, grid=grid, **options)
                shared = compiled.metadata.shared
                record["shared_bytes"] = shared
                if shared > resources[4]:
                    raise OutOfResources(shared, resources[4], "shared memory")
                # Load through Triton's public launch path, which also validates
                # thread/tensor-memory limits and populates register metadata.
                kernel[grid](*args, **options)
                record.update(registers=compiled.n_regs, spills=compiled.n_spills)
                if tuning:
                    from triton.testing import do_bench_cudagraph

                    def run():
                        kernel[grid](*args, **options)
                        return finish()

                    record["milliseconds"] = do_bench_cudagraph(run, rep=10)
                    if record["milliseconds"] < best_ms:
                        winner, best_ms = config, record["milliseconds"]
                else:
                    winner = config
                records.append(record)
            except (OutOfResources, torch.cuda.OutOfMemoryError) as error:
                record["rejected"] = str(error)
                records.append(record)
            # Do not retain the last candidate's workspace during the next one.
            del args, finish
            if winner is not None and not tuning:
                break
        if winner is None:
            raise RuntimeError(
                f"No resource-compatible MemSolve {kind} configuration on {device}: {records}"
            )
        _LAUNCH_PLANS[key] = {
            "device": list(resources), "kernel": kind, "signature": list(signature),
            "selected": asdict(winner), "tuned": tuning, "candidates": records,
        }
    args, grid, finish, _ = prepare(winner)
    kernel[grid](*args, num_warps=winner.warps, num_stages=winner.stages)
    return finish()


def is_available() -> bool:
    """The fast path uses PyTorch CUDA and Triton; no native MemSolve library."""
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def require_available() -> None:
    if not is_available():
        raise RuntimeError("the MemSolve CUDA fast path requires CUDA PyTorch and Triton")


def _device_architecture(device: torch.device | int | None = None) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the MemSolve CUDA fast path requires an available CUDA device")

    if device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        resolved_device = torch.device("cuda", device)
    else:
        resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError(
            f"the MemSolve CUDA fast path requires a CUDA device, got {resolved_device}"
        )
    if resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())

    major, minor = torch.cuda.get_device_capability(resolved_device)
    architecture = major * 10 + minor
    if architecture == 121:
        architecture = 120
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            "the MemSolve CUDA fast path supports SM80, SM86, SM87, SM89, "
            f"SM90, SM100, and SM120; got SM{major}{minor}"
        )
    return architecture


def load(*, device: torch.device | int | None = None) -> None:
    """Validate the requested CUDA runtime before kernel compilation/capture."""
    require_available()
    _device_architecture(device)


@lru_cache(maxsize=1)
def _rotary_kernel():
    # Interleaved-pair loading and conjugate backward follow FLA rotary.py
    # (MIT; see NOTICE). Specialize to packed [Q,K,V], copying V in the same
    # launch, with no full-size FP32 intermediate or saved input activations.
    from triton import jit
    import triton.language as tl

    @jit
    def rotate(X, COS, SIN, Y, SIZE: tl.constexpr, N: tl.constexpr,
               C: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
               CONJUGATE: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        channel = offset % C
        token = (offset // C) % N
        valid = offset < SIZE
        is_qk = channel < 2 * H * R
        pair = (channel % R) // 2
        cosine = tl.load(COS + token * (R // 2) + pair, valid & is_qk, other=1).to(tl.float32)
        sine = tl.load(SIN + token * (R // 2) + pair, valid & is_qk, other=0).to(tl.float32)
        x = tl.load(X + offset, valid, other=0).to(tl.float32)
        mate = tl.load(X + offset + 1 - 2 * (channel % 2), valid & is_qk, other=0).to(tl.float32)
        if CONJUGATE:
            sine = -sine
        base, cross = x * cosine, mate * sine
        rotated = tl.where(channel % 2 == 0, base - cross, base + cross)
        tl.store(Y + offset, tl.where(is_qk, rotated, x), valid)

    return rotate


def _rotate_packed(projected, num_heads, cos, sin, *, conjugate):
    projected = projected.contiguous()
    output = torch.empty_like(projected)
    batch, length, width = projected.shape
    size = batch * length * width
    _rotary_kernel()[((size + 511) // 512,)](
        projected, cos, sin, output, size, length, width, num_heads, 2 * cos.shape[1],
        conjugate, 512, num_warps=4, enable_fp_fusion=False,
    )
    return output


class _RotaryQK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, num_heads, cos, sin):
        ctx.num_heads = num_heads
        ctx.save_for_backward(cos, sin)
        return _rotate_packed(projected, num_heads, cos, sin, conjugate=False)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        cos, sin = ctx.saved_tensors
        return _rotate_packed(gradient, ctx.num_heads, cos, sin, conjugate=True), None, None, None


def rotary_qk(projected, num_heads, cos, sin):
    """Packed Q/K rotation; tables obey reference.axial_rotary_tables."""
    if projected.device.type != "cuda" or projected.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("CUDA RoPE requires CUDA float16/bfloat16/float32 projections")
    if projected.ndim != 3 or min(projected.shape) <= 0:
        raise ValueError("CUDA RoPE requires nonempty [B,N,C] projections")
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
        raise ValueError("CUDA RoPE requires positive num_heads")
    if cos.ndim != 2 or cos.shape != sin.shape or cos.shape[0] != projected.shape[1] or cos.shape[1] <= 0:
        raise ValueError("CUDA RoPE tables must have shape [N,rank/2]")
    if projected.shape[-1] <= 4 * num_heads * cos.shape[1]:
        raise ValueError("CUDA RoPE requires packed Q/K and nonempty V")
    for table in (cos, sin):
        if table.device != projected.device or table.dtype != torch.float32 or not table.is_contiguous():
            raise TypeError("CUDA RoPE requires contiguous FP32 tables on the projection device")
        if table.requires_grad:
            raise ValueError("CUDA RoPE uses fixed positional frequencies")
    load(device=projected.device)
    return _RotaryQK.apply(projected, num_heads, cos, sin)


@lru_cache(maxsize=1)
def _local_pack_kernel():
    # FLA-style adjacent-pair RoPE, fused with packing / unpacking and masking.
    # The depthwise convolution itself stays in cuDNN.
    from triton import jit
    import triton.language as tl

    @jit
    def pack(QK, V, COS, SIN, MASK, PACKED,
             SIZE: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
             Q: tl.constexpr, R: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
             ROPE: tl.constexpr, MASKED: tl.constexpr, BACKWARD: tl.constexpr,
             BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        c, row = o % C, o // C
        b, n = row // N, row % N
        valid = o < SIZE
        active = valid
        if MASKED:
            active = active & tl.load(MASK + row, valid, 0)
        is_qk = c < Q
        if BACKWARD:
            x = tl.load(PACKED + o, active, 0).to(tl.float32)
        else:
            qk = tl.load(QK + row * Q + c, active & is_qk, 0).to(tl.float32)
            v = tl.load(V + b * V0 + n * V1 + c - Q, active & ~is_qk, 0).to(tl.float32)
            x = tl.where(is_qk, qk, v)
        if ROPE:
            phase = n * (R // 2) + (c % R) // 2
            co = tl.load(COS + phase, active & is_qk, 1)
            si = tl.load(SIN + phase, active & is_qk, 0)
            if BACKWARD:
                mate = tl.load(PACKED + o + 1 - 2 * (c % 2), active & is_qk, 0).to(tl.float32)
                si = -si
            else:
                mate = tl.load(QK + row * Q + (c ^ 1), active & is_qk, 0).to(tl.float32)
            base, cross = x * co, mate * si
            x = tl.where(c % 2 == 0, base - cross, base + cross)
        if BACKWARD:
            tl.store(QK + row * Q + c, x, valid & is_qk)
            tl.store(V + row * (C - Q) + c - Q, x, valid & ~is_qk)
        else:
            tl.store(PACKED + o, x, valid)
    return pack


class _PackLocalQK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qk, value, cos, sin, mask):
        batch, length, channels = qk.shape
        width = channels + value.shape[-1]
        output = torch.empty((batch, length, width), device=qk.device, dtype=qk.dtype)
        args = (batch * length * width, length, width, channels,
                0 if cos is None else 2 * cos.shape[1], value.stride(0), value.stride(1),
                cos is not None, mask is not None)
        _local_pack_kernel()[((output.numel() + 511) // 512,)](
            qk, value, cos, sin, mask, output, *args, False, 512,
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(cos, sin, mask)
        ctx.args = args
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        cos, sin, mask = ctx.saved_tensors
        batch, length, width = gradient.shape
        channels = ctx.args[3]
        qk = torch.empty((batch, length, channels), device=gradient.device, dtype=gradient.dtype)
        value = torch.empty((batch, length, width - channels), device=gradient.device, dtype=gradient.dtype)
        _local_pack_kernel()[((gradient.numel() + 511) // 512,)](
            qk, value, cos, sin, mask, gradient.contiguous(), *ctx.args, True, 512,
            num_warps=4, enable_fp_fusion=False,
        )
        return qk, value, None, None, None


def local_qk(projected, weight, *, cos=None, sin=None, valid_mask=None, spatial_shape=None):
    """Native channels-last convolution plus fused QKV packing/RoPE/masking.

    A sequence is a 1 x N grid and its width-k filter is a 1 x k kernel. This
    equals centered Conv1d exactly, while letting cuDNN retain channels-last
    activations in both modalities. Invalid input rows are zeroed by the model.
    """
    import torch.nn.functional as functional

    batch, length, width = projected.shape
    channels = weight.shape[0]
    shape = (1, length) if spatial_shape is None else spatial_shape
    qk, value = projected.split((channels, width - channels), -1)
    with torch.autocast('cuda', enabled=False):
        x = qk.reshape(batch, *shape, channels).permute(0, 3, 1, 2)
        x = x.contiguous(memory_format=torch.channels_last)
        w = weight.to(torch.bfloat16)
        if spatial_shape is None:
            w = w.unsqueeze(2)
        qk = functional.conv2d(x, w, padding=(w.shape[-2] // 2, w.shape[-1] // 2), groups=channels)
        qk = qk.flatten(2).transpose(1, 2).contiguous()
        return _PackLocalQK.apply(qk, value, cos, sin, valid_mask)


@lru_cache(maxsize=1)
def _token_kernels():
    """Parallel token tiles, compact reduction, then a separate readout.

    The decomposition follows FLA's selective-fusion pattern. BF16x3 is
    Triton's upstream FP32-input Tensor Core decomposition, also implemented
    by OpenXLA following Henry et al. (2019). FP32 storage and accumulation
    are retained; no direct FP16 cast of unbounded coefficients or adjoints.
    BF16 Q/K/V statistics use ordinary BF16 dots. The normalized readout and
    its coefficient VJP use three residual products for the FP32 operand,
    retaining the low component omitted by ordinary BF16x3.
    The grid traverses (head, token tile, batch), keeping heads/tokens local
    before advancing to another sample, following FLA chunk-kernel scheduling.
    """
    from triton import jit
    import triton.language as tl

    @jit
    def wide_dot(a, b):
        return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="bf16x3")

    @jit
    def mixed_dot(a, b):
        # A is exactly BF16. Three residual components of the FP32 operand
        # retain its mantissa; ordinary BF16x3 omits the third component.
        # Accumulate small products first so the large term rounds only once.
        hi = b.to(tl.bfloat16)
        residual = b - hi.to(tl.float32)
        mid = residual.to(tl.bfloat16)
        lo = (residual - mid.to(tl.float32)).to(tl.bfloat16)
        result = tl.dot(a, lo)
        result = tl.dot(a, mid, result)
        return tl.dot(a, hi, result)

    @jit
    def statistics(
        X,
        E,
        COUNT,
        G,
        S,
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        T: tl.constexpr,
        GRAD: tl.constexpr,
        HAS_COUNT: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles = tl.cdiv(N, BT)
        h, tile, b = pid % H, (pid // H) % tiles, pid // (H * tiles)
        s = b * H + h
        count = tl.load(COUNT + b) if HAS_COUNT else float(N)
        n = tile * BT + tl.arange(0, BT)
        r = tl.arange(0, BR)
        # Forward uses keys; the readout adjoint uses independent queries.
        offset = 0 if GRAD else H * R
        a = tl.load(
            X + (b * N + n[:, None]) * (H * (2 * R + D)) + offset + h * R + r[None, :],
            (n[:, None] < N) & (r[None, :] < R),
            0,
        )
        if not GRAD:
            g = tl.dot(tl.trans(a), a) / count
            tl.store(
                G + ((s * T + tile) * R + r[:, None]) * R + r[None, :],
                g,
                (r[:, None] < R) & (r[None, :] < R),
            )
        for c in range(tl.cdiv(D, BD)):
            d = c * BD + tl.arange(0, BD)
            if GRAD:
                v = tl.load(
                    E + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0,
                )
                result = wide_dot(tl.trans(a), v) / tl.sqrt(count)
            else:
                v = tl.load(
                    X
                    + (b * N + n[:, None]) * (H * (2 * R + D))
                    + 2 * H * R
                    + h * D
                    + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0,
                )
                result = tl.dot(tl.trans(a), v) / tl.sqrt(count)
            tl.store(
                S + ((s * T + tile) * R + r[:, None]) * D + d[None, :],
                result,
                (r[:, None] < R) & (d[None, :] < D),
            )

    @jit
    def readout(
        X,
        COEF,
        COUNT,
        Y,
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        HAS_COUNT: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles = tl.cdiv(N, BT)
        h, tile, b = pid % H, (pid // H) % tiles, pid // (H * tiles)
        s = b * H + h
        count = tl.load(COUNT + b) if HAS_COUNT else float(N)
        n, r = tile * BT + tl.arange(0, BT), tl.arange(0, BR)
        q = tl.load(
            X + (b * N + n[:, None]) * (H * (2 * R + D)) + h * R + r[None, :],
            (n[:, None] < N) & (r[None, :] < R),
            0,
        )
        for c in range(tl.cdiv(D, BD)):
            d = c * BD + tl.arange(0, BD)
            coef = tl.load(
                COEF + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0,
            )
            y = wide_dot(q, coef) / tl.sqrt(count)
            tl.store(
                Y + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                y,
                (n[:, None] < N) & (d[None, :] < D),
            )

    @jit
    def backward(
        X,
        E,
        COEF,
        DG,
        DS,
        COUNT,
        DX,
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        HAS_COUNT: tl.constexpr,
        QUERY_GRAD: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles = tl.cdiv(N, BT)
        h, tile, b = pid % H, (pid // H) % tiles, pid // (H * tiles)
        s = b * H + h
        count = tl.load(COUNT + b) if HAS_COUNT else float(N)
        inv = 1.0 / tl.sqrt(count)
        n, r = tile * BT + tl.arange(0, BT), tl.arange(0, BR)
        k = tl.load(
            X + (b * N + n[:, None]) * (H * (2 * R + D)) + H * R + h * R + r[None, :],
            (n[:, None] < N) & (r[None, :] < R),
            0,
        )
        dg = tl.load(
            DG + (s * R + r[:, None]) * R + r[None, :],
            (r[:, None] < R) & (r[None, :] < R),
            0,
        )
        dk = wide_dot(k, dg + tl.trans(dg)) * (inv * inv)
        dq = tl.full((BT, BR), 0.0, tl.float32)
        for c in range(tl.cdiv(D, BD)):
            d = c * BD + tl.arange(0, BD)
            v = tl.load(
                X
                + (b * N + n[:, None]) * (H * (2 * R + D))
                + 2 * H * R
                + h * D
                + d[None, :],
                (n[:, None] < N) & (d[None, :] < D),
                0,
            )
            if QUERY_GRAD:
                e = tl.load(
                    E + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0,
                )
                coef = tl.load(
                    COEF + (s * R + r[:, None]) * D + d[None, :],
                    (r[:, None] < R) & (d[None, :] < D),
                    0,
                )
            ds = tl.load(
                DS + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0,
            )
            if QUERY_GRAD:
                dq += wide_dot(e, tl.trans(coef)) * inv
            dk += wide_dot(v, tl.trans(ds)) * inv
            dv = wide_dot(k, ds) * inv
            tl.store(
                DX
                + (b * N + n[:, None]) * (H * (2 * R + D))
                + 2 * H * R
                + h * D
                + d[None, :],
                dv,
                (n[:, None] < N) & (d[None, :] < D),
            )
        if QUERY_GRAD:
            tl.store(
                DX + (b * N + n[:, None]) * (H * (2 * R + D)) + h * R + r[None, :],
                dq,
                (n[:, None] < N) & (r[None, :] < R),
            )
        tl.store(
            DX + (b * N + n[:, None]) * (H * (2 * R + D)) + H * R + h * R + r[None, :],
            dk,
            (n[:, None] < N) & (r[None, :] < R),
        )

    @jit
    def normalized_readout(
        X, COEF, COUNT, W, E, Y, DW, DM, GATE, DGATE,
        N: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        HAS_COUNT: tl.constexpr, BACKWARD: tl.constexpr, HAS_GATE: tl.constexpr,
        BR: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr,
    ):
        # Recompute the compact query readout in backward instead of saving a
        # token-sized FP32 activation, following FlashAttention's rematerialization
        # principle (arxiv:2205.14135). The per-head RMS equations match FLA.
        pid = tl.program_id(0)
        tiles = tl.cdiv(N, BT)
        h, tile, b = pid % H, (pid // H) % tiles, pid // (H * tiles)
        s = b * H + h
        count = tl.load(COUNT + b) if HAS_COUNT else float(N)
        n = tile * BT + tl.arange(0, BT)
        r, d = tl.arange(0, BR), tl.arange(0, BD)
        q = tl.load(
            X + (b * N + n[:, None]) * (H * (2 * R + D)) + h * R + r[None, :],
            (n[:, None] < N) & (r[None, :] < R), 0,
        )
        coef = tl.load(
            COEF + (s * R + r[:, None]) * D + d[None, :],
            (r[:, None] < R) & (d[None, :] < D), 0,
        )
        value = mixed_dot(q, coef) / tl.sqrt(count)
        rstd = tl.rsqrt(tl.sum(value * value, 1) / D + 1e-6)
        xhat = value * rstd[:, None]
        weight = tl.load(W + d, d < D, 0)
        offset = (b * N + n[:, None]) * (H * D) + h * D + d[None, :]
        if HAS_GATE:
            logits = tl.load(GATE + offset, (n[:, None] < N) & (d[None, :] < D), 0).to(tl.float32)
            gate = tl.sigmoid(logits)
            # RMS, gain and gating share FP32 registers; round only on store.
            normalized = xhat * weight[None, :]
        if BACKWARD:
            e = tl.load(
                E + offset, (n[:, None] < N) & (d[None, :] < D), 0,
            ).to(tl.float32)
            if HAS_GATE:
                dg = ((e * normalized) * gate) * (1.0 - gate)
                tl.store(DGATE + offset, dg, (n[:, None] < N) & (d[None, :] < D))
                e = e * gate
            we = e * weight[None, :]
            result = (we - xhat * (tl.sum(we * xhat, 1) / D)[:, None]) * rstd[:, None]
            dw = tl.sum(e * xhat, 0)
            tl.store(DW + (s * tl.cdiv(N, BT) + tile) * D + d, dw, d < D)
            inv = 1.0 / tl.sqrt(count)
            # Both readout VJPs consume the same RMS adjoint. Compute them here
            # so that no token-sized FP32 adjoint crosses the compact-solve barrier.
            dm = mixed_dot(tl.trans(q), result) * inv
            tl.store(
                DM + ((s * tl.cdiv(N, BT) + tile) * R + r[:, None]) * D + d[None, :],
                dm, (r[:, None] < R) & (d[None, :] < D),
            )
            # Reload the small coefficient after dM so it need not remain
            # live across normalization and another Tensor Core product.
            coef = tl.load(
                COEF + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D), 0, volatile=True,
            )
            dq = wide_dot(result, tl.trans(coef)) * inv
            tl.store(
                Y + (b * N + n[:, None]) * (H * (2 * R + D)) + h * R + r[None, :],
                dq, (n[:, None] < N) & (r[None, :] < R),
            )
        else:
            result = normalized * gate if HAS_GATE else xhat * weight[None, :]
            tl.store(Y + offset, result, (n[:, None] < N) & (d[None, :] < D))

    return statistics, readout, backward, normalized_readout


def _block_size(size):
    return max(16, 1 << (size - 1).bit_length())


def _statistics(projected, heads, rank, counts, gradient=None):
    batch, length, width = projected.shape
    dim = (width - 2 * heads * rank) // heads
    # BF16 forward statistics fit 256 tokens per tile; short sequences then
    # need no partial-sum kernels. FP32 adjoints retain smaller live tiles.
    bt = (32 if rank > 32 else 128) if gradient is not None else 256
    def prepare(config):
        tiles = (length + config.tokens - 1) // config.tokens
        options = dict(device=projected.device, dtype=torch.float32)
        cross = torch.empty((batch * heads, tiles, rank, dim), **options)
        gram = torch.empty((batch * heads, tiles, rank, rank), **options) if gradient is None else cross
        args = (projected, gradient, counts, gram, cross, length, heads, rank, dim,
                tiles, gradient is not None, counts is not None,
                _block_size(rank), min(128, _block_size(dim)), config.tokens)

        def finish():
            reduced = (cross[:, 0] if tiles == 1 else cross.sum(1)).view(batch, heads, rank, dim)
            if gradient is not None:
                return reduced
            return (gram[:, 0] if tiles == 1 else gram.sum(1)).view(batch, heads, rank, rank), reduced

        workspace = (cross.numel() + (gram.numel() if gradient is None else 0)) * 4
        return args, (batch * heads * tiles,), finish, workspace

    return _launch_token(
        "statistics_grad" if gradient is not None else "statistics", _token_kernels()[0],
        projected.device, (batch, length, heads, rank, dim, counts is not None,
                           str(gradient.dtype) if gradient is not None else "none"),
        _LaunchConfig(bt, 4), prepare,
    )


@lru_cache(maxsize=1)
def _ridge_kernel():
    from triton import jit
    import triton.language as tl

    @jit
    def prepare(
        X, Y, R: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
        XS2: tl.constexpr, XS3: tl.constexpr, H: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        s = tl.program_id(0)
        a = tl.arange(0, BLOCK)
        i, j = a // R, a % R
        base = (s // H) * XS0 + (s % H) * XS1
        x = tl.load(X + base + i * XS2 + j * XS3, a < R * R, 0)
        result = x + (i == j).to(tl.float32)
        # Produce the column-major layout consumed by the vendor factorization.
        tl.store(Y + s * R * R + j * R + i, result, a < R * R)
    return prepare


def _ridge_system(value):
    batch, heads, rank, _ = value.shape
    result = torch.empty_like(value, memory_format=torch.contiguous_format).mT
    _ridge_kernel()[(batch * heads,)](
        value, result, rank, *value.stride(), heads,
        _block_size(rank * rank), num_warps=4,
    )
    return result


@lru_cache(maxsize=1)
def _compact_product_kernel():
    from triton import jit
    import triton.language as tl
    @jit
    def kernel(
        T, DT, DZ, Z, OUT, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        Z0: tl.constexpr, Z1: tl.constexpr, Z2: tl.constexpr, Z3: tl.constexpr,
        BR: tl.constexpr,
    ):
        s = tl.program_id(2)
        i = tl.program_id(0) * BR + tl.arange(0, BR)
        j = tl.program_id(1) * BR + tl.arange(0, BR)
        # Only lower blocks are needed; mirror them into the upper triangle.
        # A 32x32 output tile avoids the register spills of full-rank IEEE dots.
        if tl.program_id(0) >= tl.program_id(1):
            inner = tl.arange(0, 32)
            product = tl.full((BR, BR), 0., tl.float32)
            for start in range(tl.cdiv(R, 32)):
                k = start * 32 + inner
                t = tl.load(T + (s % H) * R * R + i[:, None] * R + k[None, :],
                            (i[:, None] < R) & (k[None, :] < R), 0)
                dt = tl.load(DT + s * R * R + j[:, None] * R + k[None, :],
                             (j[:, None] < R) & (k[None, :] < R), 0)
                product -= tl.dot(t, tl.trans(dt), input_precision="ieee")
            for start in range(tl.cdiv(D, 32)):
                d = start * 32 + inner
                dz = tl.load(DZ + (s * R + i[:, None]) * D + d[None, :],
                             (i[:, None] < R) & (d[None, :] < D), 0)
                z = tl.load(Z + (s // H) * Z0 + (s % H) * Z1
                            + j[:, None] * Z2 + d[None, :] * Z3,
                            (j[:, None] < R) & (d[None, :] < D), 0)
                product -= tl.dot(dz, tl.trans(z), input_precision="ieee")
            if tl.program_id(0) == tl.program_id(1):
                product = tl.where(i[:, None] >= j[None, :], product, tl.trans(product))
            value = product * 0.5
            mask = (i[:, None] < R) & (j[None, :] < R)
            tl.store(OUT + s * R * R + i[:, None] + j[None, :] * R, value, mask)
            if tl.program_id(0) != tl.program_id(1):
                tl.store(OUT + s * R * R + j[None, :] + i[:, None] * R, value, mask)
    return kernel


def _compact_product(mapping, core_samples, state_gradient, state):
    batch, heads, rank, dim = state.shape
    result = torch.empty_like(core_samples, memory_format=torch.contiguous_format).mT
    tile = min(32, _block_size(rank))
    _compact_product_kernel()[((rank + tile - 1) // tile, (rank + tile - 1) // tile, batch * heads)](
        mapping, core_samples, state_gradient, state, result,
        heads, rank, dim, *state.stride(), tile, num_warps=4,
    )
    return result


@lru_cache(maxsize=1)
def _shared_core_kernel():
    # Standard tiled GEMM (Triton tutorial), addressing the shared head map
    # directly rather than expanding it across the batch before a cuBLAS BMM.
    from triton import jit
    import triton.language as tl

    @jit
    def multiply(T, X, Y, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                 TRANSPOSE: tl.constexpr):
        s = tl.program_id(2)
        i = tl.program_id(0) * 32 + tl.arange(0, 32)
        j = tl.program_id(1) * 32 + tl.arange(0, 32)
        inner = tl.arange(0, 32)
        value = tl.full((32, 32), 0., tl.float32)
        for start in range(tl.cdiv(R, 32)):
            k = start * 32 + inner
            index = k[None, :] * R + i[:, None] if TRANSPOSE else i[:, None] * R + k[None, :]
            t = tl.load(T + (s % H) * R * R + index,
                        (i[:, None] < R) & (k[None, :] < R), 0)
            x = tl.load(X + (s * R + k[:, None]) * D + j[None, :],
                        (k[:, None] < R) & (j[None, :] < D), 0)
            value = tl.dot(t, x, value, input_precision="ieee")
        tl.store(Y + (s * R + i[:, None]) * D + j[None, :], value,
                 (i[:, None] < R) & (j[None, :] < D))
    return multiply


def _shared_core_product(mapping, state, *, transpose=False):
    batch, heads, rank, dim = state.shape
    result = torch.empty_like(state)
    _shared_core_kernel()[((rank + 31) // 32, (dim + 31) // 32, batch * heads)](
        mapping, state, result, heads, rank, dim, transpose, num_warps=4,
    )
    return result


def _compact_forward(gram, cross, core_map):
    factor, _ = torch.linalg.cholesky_ex(_ridge_system(gram), check_errors=False)
    # Reuse the compact triangular inverse across forward and backward.
    # FP32 is retained; no token-sized inverse or P is formed.
    inverse = torch.linalg.solve_triangular(
        factor, torch.eye(factor.shape[-1], device=factor.device, dtype=factor.dtype).expand_as(factor),
        upper=False,
    )
    state = inverse @ cross
    mapping = core_map
    coefficient = inverse.mT @ _shared_core_product(mapping, state)
    return coefficient, (inverse, state, mapping)


def _compact_backward(tape, gradient):
    """Analytic VJP; sum sample contributions to the shared direct T."""
    inverse, state, mapping = tape
    memory_gradient = inverse @ gradient
    core_samples = memory_gradient @ state.mT
    core_gradient = core_samples.sum(0)
    state_gradient = _shared_core_product(mapping, memory_gradient, transpose=True)
    cross_gradient = inverse.mT @ state_gradient
    # L^T dL = -(T Z) U^T - (T^T U) Z^T
    #         = -T (U Z^T)^T - state_gradient Z^T.
    # Reuse per-sample dT=U Z^T; neither dL nor L^T dL needs materializing via dL.
    symmetric = _compact_product(mapping, core_samples, state_gradient, state)
    gram_gradient = inverse.mT @ symmetric @ inverse
    return gram_gradient, cross_gradient, core_gradient


def _forward(projected, core_map, counts, norm_weight=None, gate_logits=None, *, storage_dtype=torch.bfloat16):
    from .reference import _ieee_fp32_matmul

    heads, rank = core_map.shape[:2]
    batch, length, width = projected.shape
    dim = (width - 2 * heads * rank) // heads
    gram, cross = _statistics(projected, heads, rank, counts)
    with (
        torch.no_grad(),
        torch.autocast("cuda", enabled=False),
        _ieee_fp32_matmul(projected.device),
    ):
        coefficient, tape = _compact_forward(gram, cross, core_map)
    if norm_weight is not None:
        output = torch.empty(
            (batch, length, heads * dim), device=projected.device, dtype=storage_dtype,
        )
        def prepare(config):
            args = (projected, coefficient, counts, norm_weight, None, output, None, None, gate_logits, None,
                    length, heads, rank, dim, counts is not None, False, gate_logits is not None,
                    _block_size(rank), _block_size(dim), config.tokens)
            return args, (batch * heads * ((length + config.tokens - 1) // config.tokens),), lambda: output, 0

        _launch_token(
            "normalized_readout", _token_kernels()[3], projected.device,
            (batch, length, heads, rank, dim, counts is not None, gate_logits is not None, str(storage_dtype)),
            _LaunchConfig(32, 4), prepare,
        )
        return output, (coefficient, *tape)
    output = torch.empty(
        (batch, length, heads * dim), device=projected.device, dtype=torch.float32
    )
    def prepare(config):
        args = (projected, coefficient, counts, output, length, heads, rank, dim,
                counts is not None, _block_size(rank), min(128, _block_size(dim)), config.tokens)
        return args, (batch * heads * ((length + config.tokens - 1) // config.tokens),), lambda: output, 0

    _launch_token(
        "readout", _token_kernels()[1], projected.device,
        (batch, length, heads, rank, dim, counts is not None), _LaunchConfig(128, 4), prepare,
    )
    return output, (coefficient, *tape)


class _QKVMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, core_map, counts, norm_weight, gate_logits,
                output_weight, output_bias, output_dtype):
        # Autograd may execute backward on a worker thread; Python ContextVars
        # do not propagate there. Carry the warmup request on this graph only.
        ctx.tune_launches = _TUNE_LAUNCHES.get()
        output, saved = _forward(projected, core_map, counts, norm_weight, gate_logits,
            storage_dtype=torch.float16 if output_weight is not None else torch.bfloat16)
        value16 = weight16 = None
        if output_weight is not None:
            from .reference import _readout_linear_forward
            output, value16, weight16 = _readout_linear_forward(output, output_weight, output_bias, output_dtype)
        ctx.save_for_backward(projected, counts, norm_weight, gate_logits, value16, weight16, *saved)
        ctx.has_output_bias = output_bias is not None
        ctx.heads, ctx.rank = core_map.shape[:2]
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        with autotune() if ctx.tune_launches else nullcontext():
            return _QKVMix._backward(ctx, gradient)

    @staticmethod
    def _backward(ctx, gradient):
        from .reference import _ieee_fp32_matmul

        projected, counts, norm_weight, gate_logits, value16, weight16, coefficient, *tape = ctx.saved_tensors
        output_weight_gradient = output_bias_gradient = None
        if value16 is not None:
            from .reference import _readout_linear_backward
            gradient, output_weight_gradient, output_bias_gradient = _readout_linear_backward(
                gradient, value16, weight16, ctx.has_output_bias)
        heads, rank = ctx.heads, ctx.rank
        batch, length, width = projected.shape
        dim = (width - 2 * heads * rank) // heads
        gradient = gradient.contiguous()
        norm_gradient = None
        gate_gradient = None if gate_logits is None else torch.empty_like(gate_logits)
        projected_gradient = torch.empty_like(projected)
        if norm_weight is not None:
            # FLA-style small tiles. Store the gain VJP early and reload the
            # compact coefficient inside the kernel to keep live ranges short.
            def prepare(config):
                tiles = (length + config.tokens - 1) // config.tokens
                partial = torch.empty(
                    (batch, heads, tiles, dim), device=projected.device, dtype=torch.float32,
                )
                coefficient_partial = torch.empty(
                    (batch, heads, tiles, rank, dim), device=projected.device, dtype=torch.float32,
                )
                args = (projected, coefficient, counts, norm_weight, gradient,
                        projected_gradient, partial, coefficient_partial, gate_logits, gate_gradient,
                        length, heads, rank, dim, counts is not None, True, gate_logits is not None,
                        _block_size(rank), _block_size(dim), config.tokens)

                def finish():
                    return partial.sum((0, 1, 2)), coefficient_partial.sum(2)

                return args, (batch * heads * tiles,), finish, (partial.numel() + coefficient_partial.numel()) * 4

            norm_gradient, coefficient_gradient = _launch_token(
                "normalized_readout_grad", _token_kernels()[3], projected.device,
                (batch, length, heads, rank, dim, counts is not None, gate_logits is not None, str(gradient.dtype)),
                _LaunchConfig(32, 2), prepare,
            )
            gradient = None
        else:
            coefficient_gradient = _statistics(projected, heads, rank, counts, gradient)
        with (
            torch.no_grad(),
            torch.autocast("cuda", enabled=False),
            _ieee_fp32_matmul(projected.device),
        ):
            gram_gradient, cross_gradient, core_gradient = _compact_backward(
                tape, coefficient_gradient
            )
        gram_gradient, cross_gradient = gram_gradient.contiguous(), cross_gradient.contiguous()

        def prepare(config):
            args = (projected, gradient, coefficient, gram_gradient, cross_gradient, counts,
                    projected_gradient, length, heads, rank, dim, counts is not None,
                    norm_weight is None, _block_size(rank), min(128, _block_size(dim)), config.tokens)
            return args, (batch * heads * ((length + config.tokens - 1) // config.tokens),), lambda: None, 0

        _launch_token(
            "token_grad", _token_kernels()[2], projected.device,
            (batch, length, heads, rank, dim, counts is not None, norm_weight is None,
             str(gradient.dtype) if gradient is not None else "none"),
            _LaunchConfig(32, 2 if rank <= 32 else 4), prepare,
        )
        return (projected_gradient, core_gradient, None, norm_gradient, gate_gradient,
                output_weight_gradient, output_bias_gradient, None)


def fast_mix(projected, core_map, valid_counts=None, *, norm_weight=None, gate_logits=None,
             output_weight=None, output_bias=None, output_dtype=None):
    """Direct Q/K/V readout without materializing P, for every N and rank.

    Packed input rows excluded by a mask must already be zero. Counts are
    clamped to one for empty samples by the model; they are not differentiable.
    A norm weight fuses the canonical head RMSNorm and returns BF16. Optional
    gate logits fuse post-norm sigmoid selection, without intermediate rounding.
    An output weight includes Wo in the autograd boundary: the internal readout
    and forward Wo operands use FP16, while its incoming readout VJP uses BF16 storage and FP32 arithmetic.
    Backward
    recomputes the FP32 readout from the saved compact coefficient. Without
    it, this primitive returns the unnormalized FP32 readout for operator use.
    """
    load(device=projected.device)
    if not projected.is_cuda or projected.ndim != 3 or not projected.is_contiguous():
        raise ValueError("projected must be contiguous CUDA [B,N,2*H*R+H*D]")
    if projected.dtype != torch.bfloat16:
        raise TypeError("projected must use bfloat16")
    if core_map.ndim != 3 or core_map.shape[-1] != core_map.shape[-2]:
        raise ValueError("core_map must have shape [H,R,R]")
    heads, rank, _ = core_map.shape
    if rank not in (16, 32, 48, 64):
        raise ValueError("CUDA rank must be in {16,32,48,64}")
    if (
        core_map.device != projected.device
        or core_map.dtype != torch.float32
        or not core_map.is_contiguous()
    ):
        raise TypeError("core_map must be contiguous FP32 on the projected device")
    batch, length, width = projected.shape
    if (
        min(batch, length, heads) <= 0
        or width <= 2 * heads * rank
        or (width - 2 * heads * rank) % heads
    ):
        raise ValueError("invalid packed QKV shape")
    if valid_counts is not None:
        if valid_counts.requires_grad:
            raise ValueError("valid_counts are not differentiable")
        if (
            valid_counts.shape != (batch,)
            or valid_counts.device != projected.device
            or valid_counts.dtype != torch.float32
            or not valid_counts.is_contiguous()
        ):
            raise ValueError(
                "valid_counts must be contiguous FP32 [B] on the projected device"
            )
    if norm_weight is not None and (
        norm_weight.shape != ((width - 2 * heads * rank) // heads,)
        or norm_weight.dtype != torch.float32
        or norm_weight.device != projected.device
        or not norm_weight.is_contiguous()
    ):
        raise ValueError("norm_weight must be contiguous FP32 [D], shared across heads, on the projected device")
    if gate_logits is not None and (
        norm_weight is None
        or gate_logits.shape != (batch, length, width - 2 * heads * rank)
        or gate_logits.dtype != torch.bfloat16
        or gate_logits.device != projected.device
        or not gate_logits.is_contiguous()
    ):
        raise ValueError("gate_logits require norm_weight and contiguous BF16 [B,N,H*D] on the projected device")
    if output_weight is not None:
        dim = width - 2 * heads * rank
        if (norm_weight is None or output_weight.shape != (dim, dim)
                or output_weight.device != projected.device or output_weight.dtype != torch.float32
                or not output_weight.is_contiguous()):
            raise ValueError("output_weight requires norm_weight and contiguous FP32 [H*D,H*D] on the projected device")
        if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("output_dtype must be float16, bfloat16 or float32")
        if output_bias is not None and (output_bias.shape != (dim,)
                or output_bias.device != projected.device or output_bias.dtype != torch.float32
                or not output_bias.is_contiguous()):
            raise ValueError("output_bias must be contiguous FP32 [H*D] on the projected device")
    elif output_bias is not None or output_dtype is not None:
        raise ValueError("output_bias/output_dtype require output_weight")
    if torch.is_grad_enabled() and (
        projected.requires_grad or core_map.requires_grad
        or (norm_weight is not None and norm_weight.requires_grad)
        or (gate_logits is not None and gate_logits.requires_grad)
        or (output_weight is not None and output_weight.requires_grad)
        or (output_bias is not None and output_bias.requires_grad)
    ):
        return _QKVMix.apply(projected, core_map, valid_counts, norm_weight, gate_logits,
                             output_weight, output_bias, output_dtype)
    output = _forward(projected, core_map, valid_counts, norm_weight, gate_logits,
        storage_dtype=torch.float16 if output_weight is not None else torch.bfloat16)[0]
    if output_weight is not None:
        from .reference import _readout_linear_forward
        output = _readout_linear_forward(output, output_weight, output_bias, output_dtype)[0]
    return output


@lru_cache(maxsize=1)
def _rms_kernels():
    # Adapted from FLA grouped RMSNorm, commit
    # 954438d1fcb5e1bb05c22f9908de9c5c2df74ae5, fla/modules/layernorm.py.
    # Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li.
    # Copyright (c) 2023 Tri Dao. MIT license reproduced in NOTICE.
    # Specialization: no bias/residual, FP32 input/VJP, BF16 output, per-head
    # statistics with shared affine weights. Parallel partial weight gradients
    # sum over heads and token tiles without atomic updates.
    from triton import jit
    import triton.language as tl

    @jit
    def forward(
        X,
        W,
        Y,
        RSTD,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
    ):
        rows = tl.program_id(0) * BT + tl.arange(0, BT)
        d = tl.arange(0, BD)
        x = tl.load(
            X + rows[:, None] * D + d[None, :],
            (rows[:, None] < T * H) & (d[None, :] < D),
            0,
        ).to(tl.float32)
        w = tl.load(W + d[None, :], d[None, :] < D, 0).to(
            tl.float32
        )
        rstd = tl.rsqrt(tl.sum(x * x, 1) / D + 1e-6)
        y = x * rstd[:, None] * w
        tl.store(
            Y + rows[:, None] * D + d[None, :],
            y,
            (rows[:, None] < T * H) & (d[None, :] < D),
        )
        tl.store(RSTD + rows, rstd, rows < T * H)

    @jit
    def backward(
        X,
        W,
        E,
        RSTD,
        DX,
        DW,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        SPLITS: tl.constexpr,
        SPAN: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
    ):
        h, s = tl.program_id(0), tl.program_id(1)
        d = tl.arange(0, BD)
        w = tl.load(W + d, d < D, 0).to(tl.float32)
        dw = tl.full((BT, BD), 0.0, tl.float32)
        for start in range(s * SPAN, (s + 1) * SPAN, BT):
            t = start + tl.arange(0, BT)
            active = (t < T) & (t < (s + 1) * SPAN)
            offset = (t[:, None] * H + h) * D + d[None, :]
            x = tl.load(X + offset, active[:, None] & (d[None, :] < D), 0).to(
                tl.float32
            )
            e = tl.load(E + offset, active[:, None] & (d[None, :] < D), 0).to(
                tl.float32
            )
            rstd = tl.load(RSTD + t * H + h, active, 0)
            xhat = x * rstd[:, None]
            we = e * w[None, :]
            dx = (we - xhat * (tl.sum(we * xhat, 1) / D)[:, None]) * rstd[:, None]
            tl.store(DX + offset, dx, active[:, None] & (d[None, :] < D))
            dw += e * xhat
        tl.store(DW + (h * SPLITS + s) * D + d, tl.sum(dw, 0), d < D)

    return forward, backward


class _HeadRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight):
        batch, length, heads, dim = value.shape
        value = value.contiguous()
        rows = batch * length
        output = torch.empty_like(value, dtype=torch.bfloat16)
        rstd = torch.empty((rows * heads,), device=value.device, dtype=torch.float32)
        _rms_kernels()[0][((rows * heads + 31) // 32,)](
            value,
            weight,
            output,
            rstd,
            rows,
            heads,
            dim,
            _block_size(dim),
            32,
            num_warps=4,
        )
        ctx.save_for_backward(value, weight, rstd)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        value, weight, rstd = ctx.saved_tensors
        batch, length, heads, dim = value.shape
        rows = batch * length
        splits = min(
            (rows + 31) // 32,
            max(
                1,
                torch.cuda.get_device_properties(value.device).multi_processor_count
                * 4
                // heads,
            ),
        )
        span = ((rows + splits - 1) // splits + 31) // 32 * 32
        partial = torch.empty(
            (heads, splits, dim), device=value.device, dtype=torch.float32
        )
        dx = torch.empty_like(value)
        _rms_kernels()[1][(heads, splits)](
            value,
            weight,
            gradient.contiguous(),
            rstd,
            dx,
            partial,
            rows,
            heads,
            dim,
            splits,
            span,
            _block_size(dim),
            32,
            num_warps=4,
        )
        return dx, partial.sum((0, 1))


def head_rms_norm(value, weight):
    """FLA-derived per-head RMSNorm with a shared [D] gain and epsilon 1e-6."""
    if weight.shape != (value.shape[-1],):
        raise ValueError("RMSNorm weight must have shape [D], shared across heads")
    if weight.dtype != torch.float32 or weight.device != value.device or not weight.is_contiguous():
        raise ValueError("RMSNorm weight must be contiguous FP32 on the value device")
    return _HeadRMSNorm.apply(value, weight)
