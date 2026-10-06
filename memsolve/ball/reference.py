from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache

import torch
import torch.nn.functional as functional
from torch.autograd.function import once_differentiable


def _calculation_dtype(value: torch.Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def axial_rotary_tables(rank, spatial_shape, *, device, dtype):
    """RoPE-ViT axial convention: x pairs, then y pairs; theta=100.

    Frequencies are 100**(-4*j/rank), j=0..rank/4-1, shared by heads.
    Coordinates follow the row-major patch grid without normalization or
    interpolation. Compute phases in FP32 (FP64 for the mathematical oracle).
    See naver-ai/rope-vit, models/vit_rope.py: compute_axial_cis.
    """
    height, width = spatial_shape
    with torch.inference_mode(False), torch.no_grad(), torch.autocast(device.type, enabled=False):
        frequencies = 100.0 ** (-torch.arange(0, rank, 4, device=device, dtype=dtype) / rank)
        index = torch.arange(height * width, device=device)
        x, y = (index % width).to(dtype), (index // width).to(dtype)
        phase = torch.cat((x[:, None] * frequencies, y[:, None] * frequencies), dim=-1)
        return phase.cos(), phase.sin()


def rotary_qk(projected, num_heads, cos, sin):
    """Rotate adjacent Q/K pairs after convolution, leaving packed V intact.

    The real-pair form equals multiplication by exp(i*phase). Production
    arithmetic uses FP32 and rounds once to the input projection dtype.
    """
    rank = 2 * cos.shape[-1]
    qk_width = 2 * num_heads * rank
    qk, value = projected.split((qk_width, projected.shape[-1] - qk_width), dim=-1)
    with torch.autocast(projected.device.type, enabled=False):
        pairs = qk.reshape(*projected.shape[:2], 2 * num_heads, rank // 2, 2)
        even, odd = pairs.to(_calculation_dtype(projected)).unbind(-1)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return torch.cat((rotated.reshape_as(qk).to(projected.dtype), value), dim=-1)


def convolve_qk(
    projected: torch.Tensor,
    weight: torch.Tensor,
    spatial_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Centered depthwise Q/K filtering of packed [Q, K, V]; V is unchanged.

    The caller zeros invalid projected rows before and after this operation.
    Coordinates are preserved: masking a token does not close a sequence gap.
    Native PyTorch/cuDNN convolutions operate in the projection dtype (BF16,
    or FP64 for the oracle), independently of ambient autocast. Master weights
    and their accumulated gradients remain FP32 in production.
    """
    channels = weight.shape[0]
    qk, value = projected.split((channels, projected.shape[-1] - channels), dim=-1)
    with torch.autocast(projected.device.type, enabled=False):
        weight = weight.to(projected.dtype)
        if spatial_shape is None:
            qk = functional.conv1d(
                qk.transpose(1, 2), weight, padding=weight.shape[-1] // 2, groups=channels,
            ).transpose(1, 2)
        else:
            # NHWC -> channels-last NCHW lets cuDNN use its depthwise kernels.
            qk = qk.reshape(projected.shape[0], *spatial_shape, channels)
            qk = functional.conv2d(
                qk.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last),
                weight, padding=tuple(size // 2 for size in weight.shape[2:]), groups=channels,
            ).flatten(2).transpose(1, 2)
        return torch.cat((qk, value), dim=-1)


def low_rank_gate_logits(x, down_weight, up_weight, up_bias):
    """Logits for the post-RMS gate: (x Wdown) Wup + b.

    Two ordinary GEMMs factorize the gate projection; there is no hidden
    activation or gate normalization. PyTorch owns sigmoid/multiply/autograd,
    so torch.compile can fuse the pointwise work. Gate location/activation
    follow Qiu et al., Gated Attention (2025); low rank is our cost choice.
    Projection operands/stores are BF16 in production, with FP32 master
    parameters and gradient accumulation. FP64 inputs retain the oracle.
    """
    dtype = torch.float64 if x.dtype == torch.float64 else torch.bfloat16
    hidden = tensor_core_linear(x, down_weight, output_dtype=dtype)
    return tensor_core_linear(hidden, up_weight, up_bias, output_dtype=dtype)


def sigmoid_output_gate(normalized, logits):
    """FP32 sigmoid/multiply with one output rounding; FP64 for the oracle."""
    with torch.autocast(normalized.device.type, enabled=False):
        calc_dtype = _calculation_dtype(normalized)
        gated = normalized.to(calc_dtype) * torch.sigmoid(logits.to(calc_dtype))
        return gated.to(normalized.dtype)


@lru_cache(maxsize=1)
def _biased_gemm_kernel():
    # CUDA PyTorch supplies Triton; keep CPU-only imports independent of it.
    from triton import jit
    import triton.language as tl

    @jit
    def kernel(
        X,
        W,
        B,
        Y,
        M,
        N,
        K: tl.constexpr,
        XM: tl.constexpr,
        XK: tl.constexpr,
        WK: tl.constexpr,
        WN: tl.constexpr,
        BS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        HAS_BIAS: tl.constexpr = True,
        BF16_INPUTS: tl.constexpr = False,
        SPLIT_K: tl.constexpr = 1,
    ):
        # Group eight row tiles to reuse operands in L2 (upstream Triton GEMM).
        pid = tl.program_id(0)
        num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
        group = pid // (8 * num_n)
        first_m = group * 8
        group_m = tl.minimum(num_m - first_m, 8)
        local = pid % (8 * num_n)
        rows = (first_m + local % group_m) * BM + tl.arange(0, BM)
        columns = (local // group_m) * BN + tl.arange(0, BN)
        inner = tl.arange(0, BK)
        split = tl.program_id(1)
        accumulator = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK * SPLIT_K)):
            indices = (start * SPLIT_K + split) * BK + inner
            left = tl.load(
                X + rows[:, None] * XM + indices[None, :] * XK,
                (rows[:, None] < M) & (indices[None, :] < K),
                0,
            )
            right = tl.load(
                W + columns[None, :] * WN + indices[:, None] * WK,
                (columns[None, :] < N) & (indices[:, None] < K),
                0,
            )
            if BF16_INPUTS:
                left, right = left.to(tl.bfloat16), right.to(tl.bfloat16)
            accumulator = tl.dot(left, right, accumulator)
        bias = tl.load(B + columns * BS, columns < N, 0) if HAS_BIAS else 0.
        tl.store(
            Y + split * M * N + rows[:, None] * N + columns[None, :],
            accumulator + (bias[None, :] if HAS_BIAS else 0.),
            (rows[:, None] < M) & (columns[None, :] < N),
        )

    return kernel


def _biased_gemm(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
    *,
    bf16_inputs: bool = False,
    split_k: int = 1,
) -> torch.Tensor:
    # AB + bias: FP32 accumulation/addition, then one output rounding.
    # Blocked GEMM follows Triton's official matrix-multiplication tutorial:
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    rows, inner = left.shape
    columns = right.shape[1]
    output = torch.empty((split_k, rows, columns), device=left.device, dtype=output_dtype)
    if rows == 0 or columns == 0:
        return output[0]
    # A 32-wide K tile avoids wasting half the work on low-rank gate logits.
    # Four warps also improve the full QKV/Wo shapes on SM80, without spills.
    bm, bn, bk = (128, 64 if inner <= 32 else 128, 32) if rows >= 128 else (32, 64, 32)
    if split_k > 1:
        assert bias is None and output_dtype == torch.float32
        bm, bn = 64, 64
    _biased_gemm_kernel()[(((rows + bm - 1) // bm) * ((columns + bn - 1) // bn), split_k)](
        left,
        right,
        bias,
        output,
        rows,
        columns,
        inner,
        *left.stride(),
        *right.stride(),
        0 if bias is None else bias.stride(0),
        bm,
        bn,
        bk,
        bias is not None,
        bf16_inputs,
        split_k,
        num_warps=4,
        num_stages=3,
    )
    return output[0] if split_k == 1 else output.sum(0)


def _wide_gemm(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """16-bit operands with FP32 accumulation, independent of ambient AMP policy."""
    matmul = torch.backends.cuda.matmul
    setting = ("allow_fp16_reduced_precision_reduction" if left.dtype == torch.float16
               else "allow_bf16_reduced_precision_reduction")
    previous = getattr(matmul, setting)
    setattr(matmul, setting, False)
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            if bias is not None:
                if output_dtype in (torch.float16, torch.bfloat16):
                    return _biased_gemm(left, right, bias, output_dtype)
                return torch.ops.aten.addmm.dtype(bias, left, right, torch.float32)
            operation = torch.bmm if left.ndim == 3 else torch.mm
            if output_dtype == torch.bfloat16 and left.dtype == torch.bfloat16:
                return operation(left, right)
            return operation(left, right, out_dtype=torch.float32)
    finally:
        setattr(matmul, setting, previous)


def _readout_linear_forward(value, weight, bias, output_dtype):
    """FP16 readout/Wo operands, FP32 dot and bias, one public output cast.

    The value is post-RMS/sigmoid, bounded by sqrt(head_dim)*abs(gamma).
    Only the forward operands use FP16; these tensors never form an autograd
    boundary. Master weights and the backward GEMM accumulation remain FP32.
    """
    value16, weight16 = value.to(torch.float16), weight.to(torch.float16)
    output = _wide_gemm(value16.reshape(-1, value.shape[-1]), weight16.mT,
                        bias, output_dtype=output_dtype)
    return output.to(output_dtype).reshape(*value.shape[:-1], weight.shape[0]), value16, weight16


def _readout_linear_backward(gradient, value16, weight16, has_bias):
    """BF16 VJP storage preserves exponent range; all dot products accumulate FP32."""
    grad = gradient.reshape(-1, weight16.shape[0])
    with torch.autocast(gradient.device.type, enabled=False):
        # Reuse the tiled projection GEMM; cast its operands in registers instead
        # of materializing BF16 copies of the saved FP16 forward activation.
        value = value16.reshape(-1, value16.shape[-1])
        dv = _biased_gemm(grad, weight16, None, torch.bfloat16, bf16_inputs=True).reshape_as(value16)
        # Parallel split-K, as in CUTLASS, gives a small D x D weight gradient
        # enough CTAs when the reduction spans a large batch of tokens.
        dw = _biased_gemm(grad.mT, value, None, torch.float32, bf16_inputs=True,
                         split_k=min(16, max(1, (grad.shape[0] + 1023) // 1024)))
        db = gradient.sum(tuple(range(gradient.ndim - 1)), dtype=torch.float32) if has_bias else None
    return dv, dw, db


class _ReadoutLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias, output_dtype):
        output, value16, weight16 = _readout_linear_forward(value, weight, bias, output_dtype)
        ctx.save_for_backward(value16, weight16)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return (*_readout_linear_backward(gradient, *ctx.saved_tensors, ctx.has_bias), None)


def readout_linear(value, weight, bias=None, *, output_dtype):
    """Project the FP32 normalized/gated readout without a half-gradient edge.

    GPU production: FP16 forward, BF16 backward operands/VJP storage and FP32
    accumulation. Autograd restores the input VJP to the FP32 input dtype.
    CPU and FP64 retain ordinary reference arithmetic. Callers must provide
    the normalized/gated value in FP32 (FP64 for the mathematical oracle).
    """
    if value.is_cuda and value.dtype != torch.float64:
        if value.dtype != torch.float32 or weight.dtype != torch.float32:
            raise TypeError("readout_linear requires FP32 activations and master weights")
        return _ReadoutLinear.apply(value, weight, bias, output_dtype)
    with torch.autocast(value.device.type, enabled=False):
        return functional.linear(value, weight, bias).to(output_dtype)


def _tensor_core_batches(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Broadcast matrix operands into the CUDA BMM layout."""

    batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    rows, inner, columns = left.shape[-2], left.shape[-1], right.shape[-1]
    left_batches = (
        left.to(dtype=torch.bfloat16)
        .expand(batch_shape + (rows, inner))
        .reshape(-1, rows, inner)
    )
    right_batches = (
        right.to(dtype=torch.bfloat16)
        .expand(batch_shape + (inner, columns))
        .reshape(-1, inner, columns)
    )
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
                    torch.bfloat16
                    if ctx.value_dtype == torch.bfloat16
                    else torch.float32
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


def split_qkv(projected: torch.Tensor, num_heads: int, rank: int):
    """Unpack independent Q, K and V into [B,H,N,R/D]."""
    batch, length, width = projected.shape
    value_dim = width - 2 * num_heads * rank
    if value_dim <= 0 or value_dim % num_heads:
        raise ValueError("packed QKV must have width 2*H*R + H*D")
    query, key, value = projected.split(
        (num_heads * rank, num_heads * rank, value_dim), -1
    )
    return tuple(
        t.reshape(batch, length, num_heads, -1).transpose(1, 2)
        for t in (query, key, value)
    )


def ridge_query_readout(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    core_map: torch.Tensor,
    valid_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Canonical explicit-frame oracle, returning [B,H,N,D].

    A_k=K/sqrt(n), A_q=Q/sqrt(n), R^T R=I+A_k^T A_k.
    P_k=A_k R^-1, P_q=A_q R^-1, Z=P_k^T V.
    T=core_map is learned per head, shared across samples.
    O=(P_q T) Z; T transforms queries in the key-statistic coordinates.
    The model supplies T=I+Delta, with a zero-initialized learned Delta.
    This primitive accepts any finite T, including singular matrices.
    Invalid Q/K/V rows must be zero before this boundary.
    """
    if query.ndim != 4 or query.shape != key.shape:
        raise ValueError("query and key must have identical [B,H,N,R] shapes")
    if value.ndim != 4 or value.shape[:3] != key.shape[:3]:
        raise ValueError("value must have shape [B,H,N,D]")
    batch, heads, length, rank = key.shape
    if min(batch, heads, length, rank, value.shape[-1]) <= 0:
        raise ValueError("QKV dimensions must be positive")
    if core_map.shape != (heads, rank, rank):
        raise ValueError("core_map must have shape [H,R,R]")
    if valid_counts is not None and valid_counts.shape != (batch,):
        raise ValueError("valid_counts must have shape [B]")
    calc_dtype = _calculation_dtype(query)
    with (
        torch.autocast(device_type=query.device.type, enabled=False),
        _ieee_fp32_matmul(query.device),
    ):
        query, key, value = (t.to(calc_dtype) for t in (query, key, value))
        counts = (
            query.new_full((batch,), float(length))
            if valid_counts is None
            else valid_counts.to(calc_dtype)
        )
        scale = counts.sqrt().view(batch, 1, 1, 1)
        # Accumulate statistics before normalization, matching the no-frame
        # implementation without rounding normalized token features to BF16.
        identity = torch.eye(rank, dtype=calc_dtype, device=query.device)
        gram = (key.mT @ key) / counts.view(batch, 1, 1, 1)
        factor, _ = torch.linalg.cholesky_ex(identity + gram, check_errors=False)
        key_frame = torch.linalg.solve_triangular(
            factor, key.mT / scale, upper=False
        ).mT
        query_frame = torch.linalg.solve_triangular(
            factor, query.mT / scale, upper=False
        ).mT
        state = key_frame.mT @ value
        mapping = core_map.to(dtype=calc_dtype)
        return query_frame @ (mapping @ state)


def head_rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 (FP64 for the oracle) RMS normalization per token and head.

    value is [B,N,H,D], weight is [D] and is shared across heads, as in FLA's
    Gated DeltaNet output RMSNorm. The fixed epsilon is 1e-6; neither a token
    mask nor another head enters the normalization statistics.
    """
    if weight.shape != (value.shape[-1],):
        raise ValueError("RMSNorm weight must have shape [D], shared across heads")
    calc_dtype = _calculation_dtype(value)
    with torch.autocast(device_type=value.device.type, enabled=False):
        value = value.to(calc_dtype)
        normalized = functional.rms_norm(value, (value.shape[-1],), eps=1e-6)
        return normalized * weight.to(calc_dtype)


__all__ = [
    "axial_rotary_tables",
    "rotary_qk",
    "convolve_qk",
    "ridge_query_readout",
    "head_rms_norm",
    "split_qkv",
    "tensor_core_linear",
    "tensor_core_matmul",
]
