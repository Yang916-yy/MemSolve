from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import math

import torch
import torch.nn.functional as functional
from torch.autograd.function import once_differentiable


_SOFTPLUS_ONE_OFFSET = math.log(math.expm1(1.0))


def _calculation_dtype(value: torch.Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


@lru_cache(maxsize=1)
def _biased_gemm_kernel():
    # CUDA PyTorch supplies Triton; keep CPU-only imports independent of it.
    from triton import jit
    import triton.language as tl

    @jit
    def kernel(
        X, W, B, Y,
        M, N, K: tl.constexpr,
        XM: tl.constexpr, XK: tl.constexpr,
        WK: tl.constexpr, WN: tl.constexpr, BS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        columns = tl.program_id(1) * BN + tl.arange(0, BN)
        inner = tl.arange(0, BK)
        accumulator = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            indices = start * BK + inner
            left = tl.load(
                X + rows[:, None] * XM + indices[None, :] * XK,
                (rows[:, None] < M) & (indices[None, :] < K), 0,
            )
            right = tl.load(
                W + columns[None, :] * WN + indices[:, None] * WK,
                (columns[None, :] < N) & (indices[:, None] < K), 0,
            )
            accumulator = tl.dot(left, right, accumulator)
        bias = tl.load(B + columns * BS, columns < N, 0)
        tl.store(
            Y + rows[:, None] * N + columns[None, :],
            accumulator + bias[None, :],
            (rows[:, None] < M) & (columns[None, :] < N),
        )
    return kernel


def _biased_gemm(
    left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # AB + bias: FP32 accumulation/addition, then one output rounding.
    # Blocked GEMM follows Triton's official matrix-multiplication tutorial:
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    rows, inner = left.shape
    columns = right.shape[1]
    output = torch.empty((rows, columns), device=left.device, dtype=output_dtype)
    if rows == 0 or columns == 0:
        return output
    _biased_gemm_kernel()[((rows + 63) // 64, (columns + 63) // 64)](
        left, right, bias, output, rows, columns, inner,
        *left.stride(), *right.stride(), bias.stride(0),
        64, 64, 32, num_warps=4,
    )
    return output


def _wide_gemm(
    left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor | None = None,
    *, output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """BF16 operands with FP32 accumulation, independent of ambient AMP policy."""
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            if bias is not None:
                if output_dtype in (torch.float16, torch.bfloat16):
                    return _biased_gemm(left, right, bias, output_dtype)
                return torch.ops.aten.addmm.dtype(bias, left, right, torch.float32)
            operation = torch.bmm if left.ndim == 3 else torch.mm
            if output_dtype == torch.bfloat16 and bias is None:
                return operation(left, right)
            return operation(left, right, out_dtype=torch.float32)
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous


def _tensor_core_batches(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Broadcast matrix operands into the CUDA BMM layout."""

    batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    rows, inner, columns = left.shape[-2], left.shape[-1], right.shape[-1]
    left_batches = left.to(dtype=torch.bfloat16).expand(
        batch_shape + (rows, inner)
    ).reshape(-1, rows, inner)
    right_batches = right.to(dtype=torch.bfloat16).expand(
        batch_shape + (inner, columns)
    ).reshape(-1, inner, columns)
    return left_batches, right_batches, batch_shape


class _TensorCoreBmm(torch.autograd.Function):
    """BF16 Tensor Core BMM with an FP32 result and first-order VJP."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        left_batches, right_batches, batch_shape = _tensor_core_batches(left, right)
        ctx.save_for_backward(left, right)
        ctx.batch_shape = batch_shape
        ctx.rows = left.shape[-2]
        ctx.inner = left.shape[-1]
        ctx.columns = right.shape[-1]
        return _wide_gemm(
            left_batches,
            right_batches,
        ).reshape(batch_shape + (ctx.rows, ctx.columns))

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        left, right = ctx.saved_tensors
        left_batches, right_batches, _batch_shape = _tensor_core_batches(left, right)
        grad_batches = grad_output.to(dtype=torch.bfloat16).reshape(
            -1,
            ctx.rows,
            ctx.columns,
        )

        grad_left = None
        if ctx.needs_input_grad[0]:
            grad_left = _wide_gemm(
                grad_batches,
                right_batches.mT,
            ).reshape(ctx.batch_shape + (ctx.rows, ctx.inner))
            grad_left = grad_left.sum_to_size(*left.shape).to(dtype=left.dtype)

        grad_right = None
        if ctx.needs_input_grad[1]:
            grad_right = _wide_gemm(
                left_batches.mT,
                grad_batches,
            ).reshape(ctx.batch_shape + (ctx.inner, ctx.columns))
            grad_right = grad_right.sum_to_size(*right.shape).to(dtype=right.dtype)

        return grad_left, grad_right


class _TensorCoreLinear(torch.autograd.Function):
    """BF16 Tensor Core linear map with FP32 accumulation and dtype-aware stores."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        value_bf16 = value.to(dtype=torch.bfloat16)
        weight_bf16 = weight.to(dtype=torch.bfloat16)
        ctx.save_for_backward(value_bf16, weight_bf16)
        ctx.value_shape = value.shape
        ctx.value_dtype = value.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.bias_device = None if bias is None else bias.device
        ctx.has_bias = bias is not None

        flat_value = value_bf16.reshape(-1, value.shape[-1])
        output = _wide_gemm(
            flat_value,
            weight_bf16.mT,
            None if bias is None else bias.to(device=value.device, dtype=torch.float32),
            output_dtype=(
                output_dtype
                if output_dtype in (torch.float16, torch.bfloat16)
                and (bias is not None or output_dtype == torch.bfloat16)
                else torch.float32
            ),
        ).reshape(value.shape[:-1] + (weight.shape[0],))
        return output if output_dtype is None else output.to(dtype=output_dtype)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        value, weight = ctx.saved_tensors
        flat_value = value.reshape(-1, value.shape[-1])
        flat_gradient = grad_output.to(dtype=torch.bfloat16).reshape(
            -1,
            weight.shape[0],
        )

        grad_value = None
        if ctx.needs_input_grad[0]:
            grad_value = _wide_gemm(
                flat_gradient,
                weight,
                output_dtype=(
                    torch.bfloat16 if ctx.value_dtype == torch.bfloat16 else torch.float32
                ),
            ).reshape(ctx.value_shape)
            grad_value = grad_value.to(dtype=ctx.value_dtype)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = _wide_gemm(
                flat_gradient.mT,
                flat_value,
            ).to(dtype=ctx.weight_dtype)

        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            # Preserve the original FP32 bias-reduction boundary without
            # materializing a full FP32 copy of a half-precision upstream VJP.
            dimensions = tuple(range(grad_output.ndim - 1))
            grad_bias = (
                grad_output.sum(dim=dimensions, dtype=torch.float32)
                if dimensions
                else grad_output.to(dtype=torch.float32)
            ).to(device=ctx.bias_device, dtype=ctx.bias_dtype)

        return grad_value, grad_weight, grad_bias, None


@contextmanager
def _ieee_fp32_matmul(device: torch.device):
    """Temporarily retain full FP32 products for compact-system arithmetic."""

    if device.type != "cuda":
        yield
        return

    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        matmul.fp32_precision = previous


class _FP32FactorGram(torch.autograd.Function):
    """Evaluate F F^T and its first-order VJP with full FP32 products."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        factor: torch.Tensor,
    ) -> torch.Tensor:
        value = factor.to(dtype=torch.float32)
        ctx.save_for_backward(value)
        ctx.factor_dtype = factor.dtype
        with torch.autocast(device_type=factor.device.type, enabled=False):
            with _ieee_fp32_matmul(factor.device):
                return torch.matmul(value, value.mT)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None]:
        if not ctx.needs_input_grad[0]:
            return (None,)

        (factor,) = ctx.saved_tensors
        with torch.autocast(device_type=factor.device.type, enabled=False):
            with _ieee_fp32_matmul(factor.device):
                gradient = torch.matmul(
                    grad_output.to(dtype=torch.float32)
                    + grad_output.mT.to(dtype=torch.float32),
                    factor,
                )
        return (gradient.to(dtype=ctx.factor_dtype),)


def tensor_core_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply matrices under the canonical BF16-multiplicand/FP32-accumulation contract.

    CUDA production inputs use BF16 multiplicands with FP32 accumulation. FP64
    inputs deliberately bypass that reduction so the same mathematical code
    remains the test oracle. CPU evaluation remains FP32 or FP64 because it has
    no Tensor Core execution target.
    """

    if left.ndim < 2 or right.ndim < 2:
        raise ValueError("tensor_core_matmul requires matrix dimensions")
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError("tensor_core_matmul requires floating-point inputs")
    if left.device != right.device:
        raise ValueError("tensor_core_matmul inputs must share a device")
    if left.shape[-1] != right.shape[-2]:
        raise ValueError(
            "tensor_core_matmul inner dimensions must agree, got "
            f"{left.shape[-1]} and {right.shape[-2]}"
        )

    if left.dtype == torch.float64 or right.dtype == torch.float64:
        return torch.matmul(left.to(dtype=torch.float64), right.to(dtype=torch.float64))
    if left.device.type != "cuda":
        return torch.matmul(left.to(dtype=torch.float32), right.to(dtype=torch.float32))

    return _TensorCoreBmm.apply(left, right)


def _fp32_factor_gram(factor: torch.Tensor) -> torch.Tensor:
    """Evaluate the accretive F F^T term without the BF16/FP32 rounding boundary.

    This is algebraically the same factor Gram as the reference definition.
    The complete CUDA operator uses FP32 FMA here because the compact factor is
    small and this avoids quantizing its learned lower-triangular coordinates.
    """

    if factor.dtype == torch.float64:
        return torch.matmul(factor, factor.mT)
    if factor.device.type != "cuda":
        value = factor.to(dtype=torch.float32)
        return torch.matmul(value, value.mT)
    return _FP32FactorGram.apply(factor)


class _FP32CompactMatmul(torch.autograd.Function):
    """Compact correction products and VJPs without BF16 operand rounding."""

    @staticmethod
    def forward(ctx, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(left, right)
        with torch.autocast(device_type=left.device.type, enabled=False):
            with _ieee_fp32_matmul(left.device):
                return left @ right

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        left, right = ctx.saved_tensors
        with torch.autocast(device_type=left.device.type, enabled=False):
            with _ieee_fp32_matmul(left.device):
                gradient = grad_output.to(dtype=left.dtype)
                grad_left = (
                    (gradient @ right.mT).sum_to_size(left.shape)
                    if ctx.needs_input_grad[0] else None
                )
                grad_right = (
                    (left.mT @ gradient).sum_to_size(right.shape)
                    if ctx.needs_input_grad[1] else None
                )
        return grad_left, grad_right


def _compact_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.device.type == "cuda" and left.dtype == torch.float32:
        return _FP32CompactMatmul.apply(left, right)
    return left @ right


class _FP32CoreCorrection(torch.autograd.Function):
    """Residual-form forward with the simplified resolvent VJP.

    For D = (I+K)^-1 (I-K) Z and V = (I+K)^-T dD,
    dZ = 2 V - dD and dK = -V (D+Z)^T. This avoids separately
    differentiating both occurrences of K in the residual-form solve.
    """

    @staticmethod
    def forward(ctx, generator: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        with torch.autocast(device_type=generator.device.type, enabled=False):
            identity = torch.eye(generator.shape[-1], device=generator.device, dtype=generator.dtype)
            system = identity + generator
            if state is None:
                state = identity
                rhs = identity - generator
            else:
                with _ieee_fp32_matmul(generator.device):
                    rhs = (identity - generator) @ state
            lu, pivots, _info = torch.linalg.lu_factor_ex(system, check_errors=True)
            correction = torch.linalg.lu_solve(lu, pivots, rhs)
            ctx.save_for_backward(lu, pivots, correction + state)
        return correction

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        lu, pivots, twice_equilibrium = ctx.saved_tensors
        with torch.autocast(device_type=lu.device.type, enabled=False):
            gradient = grad_output.to(dtype=lu.dtype)
            adjoint = torch.linalg.lu_solve(lu, pivots, gradient, adjoint=True)
            with _ieee_fp32_matmul(lu.device):
                grad_generator = (
                    -(adjoint @ twice_equilibrium.mT)
                    if ctx.needs_input_grad[0] else None
                )
            grad_state = 2.0 * adjoint - gradient if ctx.needs_input_grad[1] else None
        return grad_generator, grad_state


def tensor_core_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply a BF16/FP32 projection, with FP32 accumulation/bias and an optional final cast.

    The cast belongs to this projection's autograd boundary: its input VJP
    still uses the input dtype, avoiding an upstream half-to-float-to-half copy.
    """

    if value.ndim < 1:
        raise ValueError("tensor_core_linear requires a feature dimension")
    if weight.ndim != 2:
        raise ValueError("tensor_core_linear weight must have shape [out, in]")
    if value.shape[-1] != weight.shape[-1]:
        raise ValueError(
            "tensor_core_linear input and weight features must agree, got "
            f"{value.shape[-1]} and {weight.shape[-1]}"
        )
    if not value.is_floating_point() or not weight.is_floating_point():
        raise TypeError("tensor_core_linear requires floating-point inputs")
    if value.device != weight.device:
        raise ValueError("tensor_core_linear value and weight must share a device")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError(
            "tensor_core_linear bias must have shape "
            f"[{weight.shape[0]}], got {tuple(bias.shape)}"
        )

    if (
        value.device.type == "cuda"
        and value.dtype != torch.float64
        and weight.dtype != torch.float64
    ):
        # Flattening leading dimensions is exactly vec(X) W^T.  Unlike the
        # broadcast BMM primitive, its weight VJP is one GEMM rather than a
        # per-batch gradient tensor followed by a reduction.
        needs_backward = torch.is_grad_enabled() and (
            value.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )
        if needs_backward:
            return _TensorCoreLinear.apply(value, weight, bias, output_dtype)
        # Low-precision biased outputs fuse the FP32 bias and final cast;
        # other outputs retain the established GEMM conversion boundary.
        direct_dtype = (
            output_dtype
            if output_dtype in (torch.float16, torch.bfloat16)
            and (bias is not None or output_dtype == torch.bfloat16)
            else None
        )
        output = _TensorCoreLinear.apply(value, weight, bias, direct_dtype)
        bias = None
    else:
        output = tensor_core_matmul(value, weight.mT)
    if bias is not None:
        output = output + bias.to(device=output.device, dtype=output.dtype)
    return output if output_dtype is None else output.to(dtype=output_dtype)


def qr_soft_frame(relation: torch.Tensor) -> torch.Tensor:
    """Map relation coordinates to the QR soft frame P."""

    if relation.ndim < 2:
        raise ValueError("relation must have at least two matrix dimensions")
    if not relation.is_floating_point():
        raise TypeError("relation must be a floating-point tensor")

    calc_dtype = _calculation_dtype(relation)
    with torch.autocast(device_type=relation.device.type, enabled=False):
        value = relation.to(dtype=calc_dtype)
        length, rank = value.shape[-2:]
        if length == 0 or rank == 0:
            return value

        scale = (
            value.abs()
            .amax(dim=(-2, -1), keepdim=True)
            .clamp_min(1.0)
            .detach()
        )
        eye = torch.eye(rank, dtype=calc_dtype, device=value.device)
        augmented = torch.cat(
            (
                value / scale,
                eye.expand(value.shape[:-2] + (rank, rank)) / scale,
            ),
            dim=-2,
        )
        orthogonal, factor = torch.linalg.qr(augmented, mode="reduced")
        diagonal = torch.diagonal(factor, dim1=-2, dim2=-1)
        signs = torch.where(
            diagonal < 0,
            -torch.ones_like(diagonal),
            torch.ones_like(diagonal),
        ).detach()
        return (orthogonal * signs.unsqueeze(-2))[..., :length, :]


def bounded_complement(raw: torch.Tensor) -> torch.Tensor:
    """Map learned complement coordinates to the strict contractive interior.

    ``tanh`` rounds to exactly ``+/-1`` in FP32 for otherwise reachable raw
    coordinates.  Evaluate its logistic identity so the tail VJP remains
    representable, then reserve one dtype ULP from the boundary.
    """

    if not raw.is_floating_point():
        raise TypeError("complement coordinates must be a floating-point tensor")

    calc_dtype = _calculation_dtype(raw)
    with torch.autocast(device_type=raw.device.type, enabled=False):
        value = raw.to(dtype=calc_dtype)
        interior_scale = 1.0 - torch.finfo(calc_dtype).eps
        # Use the stable logistic identity for each sign. The clamps keep the
        # inactive torch.where branch from evaluating an overflowing exponent.
        positive_exponent = torch.exp(-2.0 * torch.clamp_min(value, 0.0))
        negative_exponent = torch.exp(2.0 * torch.clamp_max(value, 0.0))
        positive = (1.0 - positive_exponent) / (1.0 + positive_exponent)
        negative = (negative_exponent - 1.0) / (1.0 + negative_exponent)
        return interior_scale * torch.where(value >= 0.0, positive, negative)


def accretive_generator(raw: torch.Tensor, *, skew_coupling: bool = True) -> torch.Tensor:
    r"""Build K = L L^T + Omega from compact raw coordinates."""

    if raw.ndim < 2:
        raise ValueError("raw must have at least two matrix dimensions")
    if raw.shape[-2] != raw.shape[-1]:
        raise ValueError("raw must be square in its final two dimensions")
    if not raw.is_floating_point():
        raise TypeError("raw must be a floating-point tensor")

    calc_dtype = _calculation_dtype(raw)
    with torch.autocast(device_type=raw.device.type, enabled=False):
        value = raw.to(dtype=calc_dtype)
        rank = value.shape[-1]
        if rank == 0:
            return value

        diagonal = torch.diagonal(value, dim1=-2, dim2=-1)
        factor = torch.tril(value, diagonal=-1) + torch.diag_embed(
            functional.softplus(diagonal + _SOFTPLUS_ONE_OFFSET)
        )
        if not skew_coupling:
            return _fp32_factor_gram(factor)
        upper = torch.triu(value, diagonal=1)
        skew = upper - upper.mT
        return _fp32_factor_gram(factor) + skew


def accretive_equilibrium_mix(
    frame: torch.Tensor,
    generator: torch.Tensor | None,
    compact_state: torch.Tensor,
    content: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
    r"""Apply the zero or accretive-equilibrium compact token mix.

    The Zero base and learned correction are

        correction = (I + K)^-1 (I - K) Z,
        Y = eta (C - P Z) + P correction.

    This equals the reflected-resolvent readout because
    (I + K)^-1 (I - K) = 2 (I + K)^-1 - I. The two frame
    products are fused as P (correction - eta Z). Static K is solved
    once per head, with gradients retained and no cross-forward cache.
    Z = P^T C; generator=None means zero correction, equivalent to K=I.
    """

    if frame.ndim != 4:
        raise ValueError("frame must have shape [B, H, N, R]")
    if content.ndim != 4:
        raise ValueError("content must have shape [B, H, N, D]")

    batch, heads, length, rank = frame.shape
    if content.shape[:3] != (batch, heads, length):
        raise ValueError("frame and content must agree on [B, H, N]")
    if compact_state.shape != (batch, heads, rank, content.shape[-1]):
        raise ValueError(
            "compact_state must have shape "
            f"{(batch, heads, rank, content.shape[-1])}, "
            f"got {tuple(compact_state.shape)}"
        )
    if eta.shape != (heads,):
        raise ValueError(f"eta must have shape [{heads}], got {tuple(eta.shape)}")
    if (
        frame.dtype != content.dtype
        or compact_state.dtype != content.dtype
        or eta.dtype != content.dtype
        or frame.device != content.device
        or compact_state.device != content.device
        or eta.device != content.device
    ):
        raise ValueError("frame, compact_state, content, and eta must share dtype and device")

    eta_batch = eta.view(1, heads, 1, 1)
    if generator is None:
        compact = -eta_batch * compact_state
    else:
        if not generator.is_floating_point():
            raise TypeError("generator must be a floating-point tensor")
        if generator.dtype != content.dtype or generator.device != content.device:
            raise ValueError("generator must share content dtype and device")
        if generator.ndim == 3:
            if generator.shape != (heads, rank, rank):
                raise ValueError(
                    f"static generator must have shape {(heads, rank, rank)}, "
                    f"got {tuple(generator.shape)}"
                )
        elif generator.ndim == 4:
            if generator.shape != (batch, heads, rank, rank):
                raise ValueError(
                    f"dynamic generator must have shape {(batch, heads, rank, rank)}, "
                    f"got {tuple(generator.shape)}"
                )
        else:
            raise ValueError("generator must have shape [H, R, R] or [B, H, R, R]")

        # Keep the small-system products in FP32, like the solve itself.
        # Avoid adding a BF16 operand-rounding boundary to the correction.
        with torch.autocast(device_type=content.device.type, enabled=False):
            fp32_cuda = generator.device.type == "cuda" and generator.dtype == torch.float32
            if not fp32_cuda:
                identity = torch.eye(rank, dtype=content.dtype, device=content.device)
                system = identity + generator
                deviation = identity - generator
            if generator.ndim == 3:
                # [H,R,R] solve, rather than B copies of the same system.
                if fp32_cuda:
                    correction_map = _FP32CoreCorrection.apply(generator, None)
                else:
                    correction_map = torch.linalg.solve(system, deviation)
                correction = _compact_matmul(correction_map, compact_state)
            else:
                if fp32_cuda:
                    correction = _FP32CoreCorrection.apply(generator, compact_state)
                else:
                    correction = torch.linalg.solve(system, deviation @ compact_state)
            compact = correction - eta_batch * compact_state

    return eta_batch * content + tensor_core_matmul(frame, compact)


def compact_equilibrium_diagnostics(
    frame: torch.Tensor,
    generator: torch.Tensor,
    compact_state: torch.Tensor,
    eta: torch.Tensor,
    adjoint_rhs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    r"""Measure the exact realized gain and solve/adjoint certificate ratios.

    The returned tensors have shape ``[B, H]``.  Calculations use FP64 on the
    realized compact factors so diagnostics measure the mathematical operator
    without adding low-precision eigensolver or norm error.  For the supported
    training envelope ``N >= R``, the token gain is recovered from the full
    ``R x R`` certificate block and never materializes an ``N x N`` map.
    """

    if frame.ndim != 4:
        raise ValueError("frame must have shape [B, H, N, R]")
    if generator.ndim not in (3, 4):
        raise ValueError("generator must have shape [H, R, R] or [B, H, R, R]")
    if compact_state.ndim != 4 or adjoint_rhs.ndim != 4:
        raise ValueError("compact_state and adjoint_rhs must have shape [B, H, R, D]")

    batch, heads, length, rank = frame.shape
    if length < rank:
        raise ValueError(
            "compact diagnostics require sequence length N >= rank R, got "
            f"N={length}, R={rank}"
        )
    expected_state_shape = (batch, heads, rank, compact_state.shape[-1])
    if compact_state.shape != expected_state_shape:
        raise ValueError(
            f"compact_state must have shape {expected_state_shape}, "
            f"got {tuple(compact_state.shape)}"
        )
    if adjoint_rhs.shape != compact_state.shape:
        raise ValueError(
            "adjoint_rhs must match compact_state, got "
            f"{tuple(adjoint_rhs.shape)} and {tuple(compact_state.shape)}"
        )
    if eta.shape != (heads,):
        raise ValueError(f"eta must have shape [{heads}], got {tuple(eta.shape)}")
    if generator.ndim == 3:
        if generator.shape != (heads, rank, rank):
            raise ValueError(
                f"static generator must have shape {(heads, rank, rank)}, "
                f"got {tuple(generator.shape)}"
            )
        generator = generator.unsqueeze(0).expand(batch, -1, -1, -1)
    elif generator.shape != (batch, heads, rank, rank):
        raise ValueError(
            f"dynamic generator must have shape {(batch, heads, rank, rank)}, "
            f"got {tuple(generator.shape)}"
        )
    tensors = (generator, compact_state, eta, adjoint_rhs)
    if any(value.device != frame.device for value in tensors):
        raise ValueError("all diagnostic inputs must share a device")
    if any(not value.is_floating_point() for value in (frame, *tensors)):
        raise TypeError("all diagnostic inputs must be floating-point tensors")

    with torch.autocast(device_type=frame.device.type, enabled=False):
        frame64 = frame.to(dtype=torch.float64)
        generator64 = generator.to(dtype=torch.float64)
        compact64 = compact_state.to(dtype=torch.float64)
        rhs64 = adjoint_rhs.to(dtype=torch.float64)
        eta64 = eta.to(dtype=torch.float64).view(1, heads)

        identity = torch.eye(rank, dtype=torch.float64, device=frame.device)
        system = generator64 + identity
        equilibrium = torch.linalg.solve(system, compact64)
        adjoint = torch.linalg.solve(system.mT, rhs64)

        symmetric_system = 0.5 * (system + system.mT)
        mu = torch.linalg.eigvalsh(symmetric_system)[..., 0]

        frame_gram = frame64.mT @ frame64
        gram_values, gram_vectors = torch.linalg.eigh(frame_gram)
        scales = torch.sqrt(gram_values.clamp_min(0.0))
        inverse_system = torch.linalg.solve(
            system,
            identity.expand(batch, heads, rank, rank),
        )
        reflected = 2.0 * inverse_system - identity
        eta_matrix = eta64[..., None, None] * identity
        rotated = gram_vectors.mT @ (reflected - eta_matrix) @ gram_vectors
        certificate_block = eta_matrix + scales.diag_embed() @ rotated @ scales.diag_embed()
        token_gain = torch.linalg.svdvals(certificate_block)[..., 0]
        if length > rank:
            token_gain = torch.maximum(token_gain, eta64.abs())

        state_denominator = torch.linalg.vector_norm(
            compact64, dim=(-2, -1)
        ).clamp_min(torch.finfo(torch.float64).tiny)
        adjoint_denominator = torch.linalg.vector_norm(
            rhs64, dim=(-2, -1)
        ).clamp_min(torch.finfo(torch.float64).tiny)
        state_ratio = (
            torch.linalg.vector_norm(equilibrium, dim=(-2, -1))
            / state_denominator
        )
        adjoint_ratio = (
            torch.linalg.vector_norm(adjoint, dim=(-2, -1))
            / adjoint_denominator
        )

    return {
        "q": token_gain,
        "contraction_slack": 1.0 - token_gain,
        "mu": mu,
        "state_ratio": state_ratio,
        "adjoint_ratio": adjoint_ratio,
        "state_bound_usage": mu * state_ratio,
        "adjoint_bound_usage": mu * adjoint_ratio,
    }


__all__ = [
    "accretive_equilibrium_mix",
    "accretive_generator",
    "bounded_complement",
    "compact_equilibrium_diagnostics",
    "qr_soft_frame",
    "tensor_core_linear",
    "tensor_core_matmul",
]
