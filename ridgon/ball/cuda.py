from __future__ import annotations

from functools import lru_cache
import importlib.util

import torch

_SUPPORTED_ARCHITECTURES = frozenset((80, 86, 87, 89, 90, 100, 120))
_CUDA_CONTRACT_VERSION = 17


def is_available() -> bool:
    """The fast path uses PyTorch CUDA and Triton; no native Ridgon library."""
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


def require_available() -> None:
    if not is_available():
        raise RuntimeError("the Ridgon CUDA fast path requires CUDA PyTorch and Triton")


def _device_architecture(device: torch.device | int | None = None) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the Ridgon CUDA fast path requires an available CUDA device")

    if device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        resolved_device = torch.device("cuda", device)
    else:
        resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError(
            f"the Ridgon CUDA fast path requires a CUDA device, got {resolved_device}"
        )
    if resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())

    major, minor = torch.cuda.get_device_capability(resolved_device)
    architecture = major * 10 + minor
    if architecture == 121:
        architecture = 120
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            "the Ridgon CUDA fast path supports SM80, SM86, SM87, SM89, "
            f"SM90, SM100, and SM120; got SM{major}{minor}"
        )
    return architecture


def load(*, device: torch.device | int | None = None) -> None:
    """Validate the requested CUDA runtime before kernel compilation/capture."""
    require_available()
    _device_architecture(device)


@lru_cache(maxsize=1)
def _token_kernels():
    """Parallel token tiles, compact reduction, then a separate readout.

    The decomposition follows FLA's selective-fusion pattern. BF16x3 is
    Triton's upstream FP32-input Tensor Core decomposition, also implemented
    by OpenXLA following Henry et al. (2019). FP32 storage and accumulation
    are retained; no direct FP16 cast of unbounded coefficients or adjoints.
    BF16 Q/K/V statistics use ordinary BF16 dots.
    The grid traverses (head, token tile, batch), keeping heads/tokens local
    before advancing to another sample, following FLA chunk-kernel scheduling.
    """
    from triton import jit
    import triton.language as tl

    @jit
    def wide_dot(a, b):
        return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="bf16x3")

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
        X, COEF, COUNT, W, E, Y, DW, DM,
        N: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        HAS_COUNT: tl.constexpr, BACKWARD: tl.constexpr,
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
        value = wide_dot(q, coef) / tl.sqrt(count)
        rstd = tl.rsqrt(tl.sum(value * value, 1) / D + 1e-6)
        xhat = value * rstd[:, None]
        weight = tl.load(W + d, d < D, 0)
        offset = (b * N + n[:, None]) * (H * D) + h * D + d[None, :]
        if BACKWARD:
            e = tl.load(
                E + offset, (n[:, None] < N) & (d[None, :] < D), 0,
            ).to(tl.float32)
            we = e * weight[None, :]
            result = (we - xhat * (tl.sum(we * xhat, 1) / D)[:, None]) * rstd[:, None]
            inv = 1.0 / tl.sqrt(count)
            # Both readout VJPs consume the same RMS adjoint. Compute them here
            # so that no token-sized FP32 adjoint crosses the compact-solve barrier.
            dm = wide_dot(tl.trans(q), result) * inv
            tl.store(
                DM + ((s * tl.cdiv(N, BT) + tile) * R + r[:, None]) * D + d[None, :],
                dm, (r[:, None] < R) & (d[None, :] < D),
            )
            dq = wide_dot(result, tl.trans(coef)) * inv
            tl.store(
                Y + (b * N + n[:, None]) * (H * (2 * R + D)) + h * R + r[None, :],
                dq, (n[:, None] < N) & (r[None, :] < R),
            )
            dw = tl.sum(e * xhat, 0)
            tl.store(DW + (s * tl.cdiv(N, BT) + tile) * D + d, dw, d < D)
        else:
            result = xhat * weight[None, :]
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
    tiles = (length + bt - 1) // bt
    options = dict(device=projected.device, dtype=torch.float32)
    cross = torch.empty((batch * heads, tiles, rank, dim), **options)
    gram = (
        torch.empty((batch * heads, tiles, rank, rank), **options)
        if gradient is None
        else cross
    )
    _token_kernels()[0][(batch * heads * tiles,)](
        projected,
        gradient,
        counts,
        gram,
        cross,
        length,
        heads,
        rank,
        dim,
        tiles,
        gradient is not None,
        counts is not None,
        _block_size(rank),
        min(128, _block_size(dim)),
        bt,
        num_warps=4,
        num_stages=1,
    )
    cross = (cross[:, 0] if tiles == 1 else cross.sum(1)).view(batch, heads, rank, dim)
    return (
        ((gram[:, 0] if tiles == 1 else gram.sum(1)).view(batch, heads, rank, rank), cross)
        if gradient is None
        else cross
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
        s = tl.program_id(0)
        r = tl.arange(0, BR)
        square_mask = (r[:, None] < R) & (r[None, :] < R)
        t = tl.load(T + (s % H) * R * R + r[:, None] * R + r[None, :], square_mask, 0)
        dt = tl.load(DT + s * R * R + r[:, None] * R + r[None, :], square_mask, 0)
        product = -tl.dot(t, tl.trans(dt), input_precision="ieee")
        for start in range(tl.cdiv(D, 32)):
            d = start * 32 + tl.arange(0, 32)
            mask = (r[:, None] < R) & (d[None, :] < D)
            dz = tl.load(DZ + s * R * D + r[:, None] * D + d[None, :], mask, 0)
            z = tl.load(
                Z + (s // H) * Z0 + (s % H) * Z1
                + r[:, None] * Z2 + d[None, :] * Z3, mask, 0,
            )
            product -= tl.dot(dz, tl.trans(z), input_precision="ieee")
        # sym(Phi(product)): copy the lower triangle, halve every stored entry.
        symmetric = tl.where(r[:, None] >= r[None, :], product, tl.trans(product)) * 0.5
        tl.store(OUT + s * R * R + r[:, None] + r[None, :] * R, symmetric, square_mask)
    return kernel


def _compact_product(mapping, core_samples, state_gradient, state):
    batch, heads, rank, dim = state.shape
    result = torch.empty_like(core_samples, memory_format=torch.contiguous_format).mT
    _compact_product_kernel()[(batch * heads,)](
        mapping, core_samples, state_gradient, state, result,
        heads, rank, dim, *state.stride(), _block_size(rank), num_warps=4,
    )
    return result


def _compact_forward(gram, cross, core_map):
    factor, _ = torch.linalg.cholesky_ex(_ridge_system(gram), check_errors=False)
    state = torch.linalg.solve_triangular(factor, cross, upper=False)
    mapping = core_map
    coefficient = torch.linalg.solve_triangular(
        factor.mT, mapping @ state, upper=True
    ).contiguous()
    return coefficient, (factor, state, mapping)


def _compact_backward(tape, gradient):
    """Analytic VJP; sum sample contributions to the shared direct T."""
    factor, state, mapping = tape
    memory_gradient = torch.linalg.solve_triangular(factor, gradient, upper=False)
    core_samples = memory_gradient @ state.mT
    core_gradient = core_samples.sum(0)
    state_gradient = mapping.mT @ memory_gradient
    cross_gradient = torch.linalg.solve_triangular(
        factor.mT, state_gradient, upper=True
    )
    # L^T dL = -(T Z) U^T - (T^T U) Z^T
    #         = -T (U Z^T)^T - state_gradient Z^T.
    # Reuse per-sample dT=U Z^T; neither dL nor L^T dL needs materializing via dL.
    symmetric = _compact_product(mapping, core_samples, state_gradient, state)
    intermediate = torch.linalg.solve_triangular(factor.mT, symmetric, upper=True)
    gram_gradient = torch.linalg.solve_triangular(
        factor.mT, intermediate.mT, upper=True
    ).mT
    return gram_gradient, cross_gradient, core_gradient


def _forward(projected, core_map, counts, norm_weight=None):
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
            (batch, length, heads * dim), device=projected.device, dtype=torch.bfloat16,
        )
        _token_kernels()[3][(batch * heads * ((length + 31) // 32),)](
            projected, coefficient, counts, norm_weight, None, output, None, None,
            length, heads, rank, dim, counts is not None, False,
            _block_size(rank), _block_size(dim), 32, num_warps=4, num_stages=1,
        )
        return output, (coefficient, *tape)
    output = torch.empty(
        (batch, length, heads * dim), device=projected.device, dtype=torch.float32
    )
    _token_kernels()[1][(batch * heads * ((length + 127) // 128),)](
        projected,
        coefficient,
        counts,
        output,
        length,
        heads,
        rank,
        dim,
        counts is not None,
        _block_size(rank),
        min(128, _block_size(dim)),
        128,
        num_warps=4,
        num_stages=1,
    )
    return output, (coefficient, *tape)


class _QKVMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, core_map, counts, norm_weight):
        output, saved = _forward(projected, core_map, counts, norm_weight)
        ctx.save_for_backward(projected, counts, norm_weight, *saved)
        ctx.heads, ctx.rank = core_map.shape[:2]
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        from .reference import _ieee_fp32_matmul

        projected, counts, norm_weight, coefficient, *tape = ctx.saved_tensors
        heads, rank = ctx.heads, ctx.rank
        batch, length, width = projected.shape
        dim = (width - 2 * heads * rank) // heads
        gradient = gradient.contiguous()
        norm_gradient = None
        projected_gradient = torch.empty_like(projected)
        if norm_weight is not None:
            # FLA-style small tiles with fewer warps reduce register pressure
            # in the fused readout VJP without changing its BF16x3 arithmetic.
            bt = 32
            tiles = (length + bt - 1) // bt
            partial = torch.empty(
                (batch, heads, tiles, dim), device=projected.device, dtype=torch.float32,
            )
            coefficient_partial = torch.empty(
                (batch, heads, tiles, rank, dim), device=projected.device, dtype=torch.float32,
            )
            _token_kernels()[3][(batch * heads * tiles,)](
                projected, coefficient, counts, norm_weight, gradient,
                projected_gradient, partial, coefficient_partial,
                length, heads, rank, dim, counts is not None, True,
                _block_size(rank), _block_size(dim), bt, num_warps=2, num_stages=1,
            )
            # Shared affine weights accumulate gradients from every head.
            norm_gradient = partial.sum((0, 1, 2))
            coefficient_gradient = coefficient_partial.sum(2)
            del coefficient_partial, partial
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
        _token_kernels()[2][(batch * heads * ((length + 31) // 32),)](
            projected,
            gradient,
            coefficient,
            gram_gradient.contiguous(),
            cross_gradient.contiguous(),
            counts,
            projected_gradient,
            length,
            heads,
            rank,
            dim,
            counts is not None,
            norm_weight is None,
            _block_size(rank),
            min(128, _block_size(dim)),
            32,
            num_warps=2 if rank <= 32 else 4,
            num_stages=1,
        )
        return projected_gradient, core_gradient, None, norm_gradient


def fast_mix(projected, core_map, valid_counts=None, *, norm_weight=None):
    """Direct Q/K/V readout without materializing P, for every N and rank.

    Packed input rows excluded by a mask must already be zero. Counts are
    clamped to one for empty samples by the model; they are not differentiable.
    A norm weight fuses the canonical head RMSNorm and returns BF16. Backward
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
    if torch.is_grad_enabled() and (
        projected.requires_grad or core_map.requires_grad
        or (norm_weight is not None and norm_weight.requires_grad)
    ):
        return _QKVMix.apply(projected, core_map, valid_counts, norm_weight)
    return _forward(projected, core_map, valid_counts, norm_weight)[0]


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
