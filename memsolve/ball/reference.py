from __future__ import annotations

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as functional
from torch.autograd.function import once_differentiable


_SOFTPLUS_ONE_OFFSET = math.log(math.expm1(1.0))


def _calculation_dtype(value: torch.Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


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

    from .cuda import _TensorCoreBmm

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
    fuse_weight_bias_grad: bool = False,
    output_mask: torch.Tensor | None = None,
    input_mask: torch.Tensor | None = None,
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

    if input_mask is not None:
        if input_mask.dtype != torch.bool or input_mask.shape != value.shape[:-1] or input_mask.device != value.device:
            raise ValueError("input_mask must be boolean with the input token shape and device")
        input_mask = input_mask.contiguous()

    if output_mask is not None:
        if output_mask.dtype != torch.bool or output_mask.shape != value.shape[:-1] or output_mask.device != value.device:
            raise ValueError("output_mask must be boolean with the input's token shape and device")
        output_mask = output_mask.contiguous()

    if (
        value.device.type == "cuda"
        and value.dtype != torch.float64
        and weight.dtype != torch.float64
    ):
        from .cuda import _TensorCoreLinear

        # Flatten tokens so the weight gradient is one GEMM across all batches.
        needs_backward = torch.is_grad_enabled() and (
            value.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )
        if needs_backward:
            return _TensorCoreLinear.apply(value, weight, bias, output_dtype, fuse_weight_bias_grad, output_mask, input_mask)
        # Fuse FP32 bias addition with the low-precision output cast.
        direct_dtype = (
            output_dtype
            if output_dtype in (torch.float16, torch.bfloat16)
            and (bias is not None or output_dtype == torch.bfloat16)
            else None
        )
        output = _TensorCoreLinear.apply(value, weight, bias, direct_dtype, fuse_weight_bias_grad, output_mask, input_mask)
        bias = None
        output_mask = None
    else:
        if input_mask is not None:
            value = value.masked_fill(~input_mask[..., None], 0)
        output = tensor_core_matmul(value, weight.mT)
    if bias is not None:
        output = output + bias.to(device=output.device, dtype=output.dtype)
    if output_mask is not None:
        output = output.masked_fill(~output_mask[..., None], 0)
    return output if output_dtype is None else output.to(dtype=output_dtype)


def rotary_position_factors(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    base: float,
    train_length: int,
    context_length: int | torch.Tensor,
    scaling: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-head adjacent-pair RoVE phases, with unit-amplitude YaRN scaling.

    YaRN interpolates slow frequencies and preserves fast frequencies. Its
    beta=32/1 transition follows the original correction-range index ramp;
    attention-temperature rescaling is not part of this orthogonal transport.
    ``context_length`` may contain one physical span per sample, independently
    of the absolute position offset. FP64 is reserved for the reference oracle.
    """
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=position_ids.device.type, enabled=False):
        pair = torch.arange(head_dim // 2, device=position_ids.device, dtype=calc_dtype)
        frequency = base ** (-2.0 * pair / head_dim)
        if scaling == "yarn":
            if isinstance(context_length, torch.Tensor):
                scale = (context_length.to(device=position_ids.device, dtype=calc_dtype)
                         / train_length).clamp_min(1.0).reshape(-1, 1)
            else:
                scale = max(1.0, context_length / train_length)
            low = max(0, math.floor(
                head_dim * math.log(train_length / (2 * math.pi * 32)) / (2 * math.log(base))
            ))
            high = min(head_dim // 2 - 1, math.ceil(
                head_dim * math.log(train_length / (2 * math.pi)) / (2 * math.log(base))
            ))
            ramp = ((pair - low) / max(high - low, 1e-4)).clamp(0, 1)
            frequency = frequency * (1.0 - ramp + ramp / scale)
        elif scaling != "none":
            raise ValueError("rotary scaling must be 'none' or 'yarn'")
        if frequency.ndim == 1:
            frequency = frequency.unsqueeze(0)
        angle = position_ids.to(dtype=calc_dtype).unsqueeze(-1) * frequency.unsqueeze(1)
        return angle.cos().contiguous(), angle.sin().contiguous()


def axial_rotary_position_factors(
    coordinates: torch.Tensor,
    *,
    head_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-amplitude 2D RoVE: half the channel pairs for each coordinate axis."""
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape [B, N, 2]")
    if head_dim <= 0 or head_dim % 4:
        raise ValueError("2D RoVE requires head_dim divisible by four")
    factors = [rotary_position_factors(
        coordinates[..., axis], head_dim=head_dim // 2, base=base,
        train_length=1, context_length=1, scaling="none", dtype=dtype,
    ) for axis in range(2)]
    return tuple(torch.cat(tables, dim=-1).contiguous() for tables in zip(*factors))


def rotate_content(
    content: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply shared-per-head planar rotations to [B,H,N,D] content."""
    with torch.autocast(device_type=content.device.type, enabled=False):
        value = content.to(dtype=_calculation_dtype(content))
        cosine, sine = cos[:, None], sin[:, None]
        if inverse:
            sine = -sine
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even * cosine - odd * sine,
                            even * sine + odd * cosine), dim=-1).flatten(-2)


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


class _BoundedComplement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (1.0 - torch.finfo(value.dtype).eps) * value.tanh()

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        # Differentiate the mathematical tanh, not its rounded output. The
        # usual 1-tanh(x)^2 loses representable gradients in the FP32 tails.
        # where (rather than abs) also preserves higher derivatives at zero.
        magnitude = torch.where(value >= 0, value, -value)
        exponent = torch.exp(-2.0 * magnitude)
        derivative = (1.0 - torch.finfo(value.dtype).eps) * (
            4.0 * exponent / (1.0 + exponent).square()
        )
        return gradient * derivative


def bounded_complement(raw: torch.Tensor) -> torch.Tensor:
    """Map learned complement coordinates to the strict contractive interior.

    ``tanh`` rounds to exactly ``+/-1`` in FP32 for otherwise reachable raw
    coordinates. Use native tanh for a cancellation-free value near zero,
    but its analytic exponential derivative so the tail VJP remains
    representable. Reserve one dtype ULP from the boundary.
    """

    if not raw.is_floating_point():
        raise TypeError("complement coordinates must be a floating-point tensor")

    calc_dtype = _calculation_dtype(raw)
    with torch.autocast(device_type=raw.device.type, enabled=False):
        return _BoundedComplement.apply(raw.to(dtype=calc_dtype))


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
    *,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Apply the complement or accretive-equilibrium compact token mix.

    The complement base and learned correction are

        correction = (I + K)^-1 (I - K) Z,
        Y = eta (C - P Z) + P correction.

    This equals the reflected-resolvent readout because
    (I + K)^-1 (I - K) = 2 (I + K)^-1 - I. The two frame
    products are fused as P (correction - eta Z). Shared K is solved
    once per head, with gradients retained and no cross-forward cache.
    Z = P^T C; generator=None means a zero learned-core map, equivalent to K=I.
    The negative complement correction -eta P Z is still present.
    With value transport, Z = P^T rot(C) and the entire compact correction
    is inverse-rotated before adding the unchanged local term eta C.

    For P=A R^-1 and R^T R=I+A^T A, the complement can be evaluated
    as eta*(C-A H), H=solve(I+A^T A, A^T C). Its derivative follows
    the SPD solve directly: dH=M^-1(dQ-dM H). Cholesky coordinates
    are an implementation detail and need not be differentiated separately.
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
                    f"shared generator must have shape {(heads, rank, rank)}, "
                    f"got {tuple(generator.shape)}"
                )
        elif generator.ndim == 4:
            if generator.shape != (batch, heads, rank, rank):
                raise ValueError(
                    f"batched generator must have shape {(batch, heads, rank, rank)}, "
                    f"got {tuple(generator.shape)}"
                )
        else:
            raise ValueError("generator must have shape [H, R, R] or [B, H, R, R]")

        # Use FP32 for both the compact correction products and the solve.
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

    if (rotary_cos is None) != (rotary_sin is None):
        raise ValueError("rotary_cos and rotary_sin must be supplied together")
    token_correction = (
        tensor_core_matmul(frame, compact) if rotary_cos is None
        else _compact_matmul(frame, compact)
    )
    if rotary_cos is not None:
        token_correction = rotate_content(
            token_correction, rotary_cos, rotary_sin, inverse=True
        )
    return eta_batch * content + token_correction


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
                f"shared generator must have shape {(heads, rank, rank)}, "
                f"got {tuple(generator.shape)}"
            )
        generator = generator.unsqueeze(0).expand(batch, -1, -1, -1)
    elif generator.shape != (batch, heads, rank, rank):
        raise ValueError(
            f"batched generator must have shape {(batch, heads, rank, rank)}, "
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
    "axial_rotary_position_factors",
    "bounded_complement",
    "compact_equilibrium_diagnostics",
    "qr_soft_frame",
    "rotary_position_factors",
    "rotate_content",
    "tensor_core_linear",
    "tensor_core_matmul",
]
