from __future__ import annotations

from functools import lru_cache
import importlib
import os
from pathlib import Path
from threading import Lock

import torch
from torch.autograd.function import once_differentiable

_LOAD_LOCK = Lock()
_SUPPORTED_ARCHITECTURES = frozenset((80, 86, 87, 89, 90, 100, 120))
_NATIVE_CONTRACT_VERSION = 17
_RUNTIME_PACKAGE = "memsolve_cuda_runtime"
_LOADED_ARCHITECTURE: int | None = None


def _native_operator_abi_is_registered() -> bool:
    namespace = getattr(torch.ops, "memsolve_equilibrium", None)
    return (
        namespace is not None
        and hasattr(namespace, "compact_lu")
        and hasattr(namespace, "compact_getrs")
        and hasattr(namespace, "compact_inverse")
        and hasattr(namespace, "linear_weight_bias")
    )


def _native_contract_version() -> int | None:
    """Return the registered native contract version, if the query exists."""

    namespace = getattr(torch.ops, "memsolve_equilibrium", None)
    if namespace is None or not hasattr(namespace, "contract_version"):
        return None
    try:
        version = namespace.contract_version()
    except RuntimeError:
        return None
    return version if isinstance(version, int) else None


def _check_native_contract() -> None:
    version = _native_contract_version()
    if version is None:
        raise RuntimeError(
            "the loaded MemSolve CUDA extension does not expose the required native "
            f"contract version {_NATIVE_CONTRACT_VERSION}; rebuild it with "
            "tools/build_cuda.sh and restart the process"
        )
    if version != _NATIVE_CONTRACT_VERSION:
        raise RuntimeError(
            "the loaded MemSolve CUDA extension has native contract version "
            f"{version}, but this package requires {_NATIVE_CONTRACT_VERSION}; "
            "rebuild it with tools/build_cuda.sh and restart the process"
        )


def is_available() -> bool:
    """Return whether the compact-solve runtime is registered."""

    return (
        _native_operator_abi_is_registered()
        and _native_contract_version() == _NATIVE_CONTRACT_VERSION
    )


def require_available() -> None:
    """Reject an explicit CUDA request until its extension is loaded."""

    with _LOAD_LOCK:
        if _LOADED_ARCHITECTURE is not None:
            return
        if not is_available():
            if _native_operator_abi_is_registered():
                _check_native_contract()
            raise RuntimeError(
                "the MemSolve accretive-equilibrium CUDA extension is not loaded; "
                "install the matching precompiled runtime wheel or build it with "
                "tools/build_cuda.sh, then call memsolve.ball.cuda.load(); "
                f"native contract version {_NATIVE_CONTRACT_VERSION} is required"
            )


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
            "the MemSolve CUDA fast path requires a CUDA device, " f"got {resolved_device}"
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


def _packaged_library_path(architecture: int) -> Path | None:
    """Return the matching release runtime library when its wheel is installed."""

    try:
        runtime = importlib.import_module(_RUNTIME_PACKAGE)
    except ModuleNotFoundError as error:
        if error.name == _RUNTIME_PACKAGE:
            return None
        raise RuntimeError(
            "the installed MemSolve CUDA runtime wheel could not be imported"
        ) from error

    from memsolve import __version__ as package_version

    expected = {
        "MEMSOLVE_VERSION": package_version,
        "NATIVE_CONTRACT_VERSION": _NATIVE_CONTRACT_VERSION,
        "TORCH_VERSION": torch.__version__,
        "CUDA_VERSION": torch.version.cuda or "",
        "CXX11_ABI": int(torch.compiled_with_cxx11_abi()),
    }
    for name, value in expected.items():
        if getattr(runtime, name, None) != value:
            raise RuntimeError(
                "the installed MemSolve CUDA runtime wheel is incompatible: "
                f"{name} is {getattr(runtime, name, None)!r}, expected {value!r}"
            )
    architectures = getattr(runtime, "ARCHITECTURES", ())
    if architecture not in architectures:
        raise RuntimeError(
            "the installed MemSolve CUDA runtime wheel does not contain "
            f"memsolve_equilibrium_sm{architecture}.so"
        )
    library_path = getattr(runtime, "library_path", None)
    if not callable(library_path):
        raise RuntimeError(
            "the installed MemSolve CUDA runtime wheel does not expose library_path()"
        )
    return Path(library_path(architecture))


def _development_library_path(architecture: int) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    return (
        repository_root
        / "build"
        / "cuda"
        / "lib"
        / f"memsolve_equilibrium_sm{architecture}.so"
    )


def _default_library_path(device: torch.device | int | None = None) -> Path:
    override = os.environ.get("MEMSOLVE_CUDA_LIBRARY")
    if override:
        return Path(override).expanduser()

    architecture = _device_architecture(device)
    development = _development_library_path(architecture)
    if development.is_file():
        return development
    packaged = _packaged_library_path(architecture)
    return packaged if packaged is not None else development


def load(
    path: str | os.PathLike[str] | None = None,
    *,
    device: torch.device | int | None = None,
) -> None:
    """Load the CUDA extension for the requested device and check its ABI."""

    global _LOADED_ARCHITECTURE
    requested_architecture = _device_architecture(device)
    with _LOAD_LOCK:
        if is_available():
            if (
                _LOADED_ARCHITECTURE is not None
                and _LOADED_ARCHITECTURE != requested_architecture
            ):
                raise RuntimeError(
                    "the loaded MemSolve CUDA extension targets "
                    f"SM{_LOADED_ARCHITECTURE}, not requested SM{requested_architecture}"
                )
            return
        if _native_operator_abi_is_registered():
            _check_native_contract()

        library = Path(path) if path is not None else _default_library_path(device)
        library = library.expanduser().resolve()
        expected_name = f"memsolve_equilibrium_sm{requested_architecture}.so"
        if library.name != expected_name:
            raise RuntimeError(
                "the strict MemSolve CUDA fast path requires "
                f"{expected_name} for SM{requested_architecture}, got {library.name}"
            )
        if not library.is_file():
            raise RuntimeError(
                "the MemSolve CUDA extension was not found at "
                f"{library}; build it with tools/build_cuda.sh or pass its path to load()"
            )

        torch.ops.load_library(str(library))
        _check_native_contract()
        _LOADED_ARCHITECTURE = requested_architecture


@lru_cache(maxsize=1)
def _biased_gemm_kernel():
    # Grouped GEMM/indexing adapted from PyTorch Inductor; see THIRD_PARTY_NOTICES.md.
    from triton import jit
    import triton.language as tl

    @jit
    def kernel(
        X, W, B, Y, MASK,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        XM: tl.constexpr, XK: tl.constexpr,
        WK: tl.constexpr, WN: tl.constexpr, BS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
        HAS_BIAS: tl.constexpr, HAS_MASK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m, grid_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
        group_width = GROUP_M * grid_n
        group = pid // group_width
        group_size = tl.minimum(grid_m - group * GROUP_M, GROUP_M)
        tile_m = group * GROUP_M + pid % group_size
        tile_n = pid % group_width // group_size
        tl.assume(tile_m >= 0)
        tl.assume(tile_n >= 0)
        rows = tile_m * BM + tl.arange(0, BM)
        columns = tile_n * BN + tl.arange(0, BN)
        inner = tl.arange(0, BK)
        if M >= BM and N >= BN:
            # Wrap edge loads; only valid output lanes are stored.
            if ((XM == 1 and XK == M) or (XM == K and XK == 1)) and K > 1:
                load_rows = tl.max_contiguous(tl.multiple_of(rows % M, BM), BM)
            else:
                load_rows = rows % M
            if ((WK == 1 and WN == K) or (WK == N and WN == 1)) and K > 1:
                load_columns = tl.max_contiguous(tl.multiple_of(columns % N, BN), BN)
            else:
                load_columns = columns % N
        else:
            load_rows, load_columns = rows, columns
        accumulator = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            indices = start * BK + inner
            if M < BM or N < BN:
                left = tl.load(
                    X + load_rows[:, None] * XM + indices[None, :] * XK,
                    (load_rows[:, None] < M) & (indices[None, :] < K), 0,
                )
                right = tl.load(
                    W + load_columns[None, :] * WN + indices[:, None] * WK,
                    (load_columns[None, :] < N) & (indices[:, None] < K), 0,
                )
            elif K % BK == 0:
                left = tl.load(X + load_rows[:, None] * XM + indices[None, :] * XK)
                right = tl.load(W + load_columns[None, :] * WN + indices[:, None] * WK)
            else:
                left = tl.load(
                    X + load_rows[:, None] * XM + indices[None, :] * XK,
                    indices[None, :] < K, 0,
                )
                right = tl.load(
                    W + load_columns[None, :] * WN + indices[:, None] * WK,
                    indices[:, None] < K, 0,
                )
            accumulator = tl.dot(left, right, accumulator)
        # Recompute store indices to shorten their live ranges across the loop.
        rows = tile_m * BM + tl.arange(0, BM)
        columns = tile_n * BN + tl.arange(0, BN)
        if HAS_BIAS:
            bias = tl.load(B + columns * BS, columns < N, 0).to(tl.float32)
            accumulator += bias[None, :]
        if HAS_MASK:
            valid = tl.load(MASK + rows, rows < M, 0)
            accumulator = tl.where(valid[:, None], accumulator, 0)
        tl.store(
            Y + rows[:, None] * N + columns[None, :],
            accumulator,
            (rows[:, None] < M) & (columns[None, :] < N),
        )
    return kernel


def _biased_gemm(
    left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor | None,
    output_dtype: torch.dtype, output_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # AB + bias: FP32 accumulation/addition, then one output rounding.
    rows, inner = left.shape
    columns = right.shape[1]
    output = torch.empty((rows, columns), device=left.device, dtype=output_dtype)
    if rows == 0 or columns == 0:
        return output
    if rows >= 128 and columns >= 128:
        bm, bn, warps, group = 128, 128, 4, 4
        bk, stages = (64, 3) if inner >= 512 else (32, 4)
    else:
        bm, bn, bk, warps, stages, group = 64, 64, 32, 4, 3, 1
    grid = ((rows + bm - 1) // bm) * ((columns + bn - 1) // bn)
    _biased_gemm_kernel()[(grid,)](
        left, right, bias, output, output_mask, rows, columns, inner,
        *left.stride(), *right.stride(), 0 if bias is None else bias.stride(0),
        bm, bn, bk, group, bias is not None, output_mask is not None,
        num_warps=warps, num_stages=stages,
    )
    return output


def _wide_gemm(
    left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor | None = None,
    *, output_dtype: torch.dtype = torch.float32, output_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16 operands with FP32 accumulation, independent of ambient AMP policy."""
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            if output_mask is not None:
                return _biased_gemm(left, right, bias, output_dtype, output_mask)
            if bias is not None:
                if output_dtype in (torch.float16, torch.bfloat16):
                    return _biased_gemm(left, right, bias, output_dtype)
                return torch.ops.aten.addmm.dtype(bias, left, right, torch.float32)
            operation = torch.bmm if left.ndim == 3 else torch.mm
            if output_dtype == torch.bfloat16:
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


@lru_cache(maxsize=1)
def _masked_cast_kernel():
    from triton import jit
    import triton.language as tl

    @jit
    def kernel(X, MASK, Y, SIZE: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = tl.load(MASK + i // WIDTH, i < SIZE, 0)
        value = tl.load(X + i, (i < SIZE) & valid, 0)
        tl.store(Y + i, value, i < SIZE)
    return kernel


def _masked_cast(value, mask, dtype):
    # Sanitize before GEMM: zero times NaN is still NaN in a weight gradient.
    if mask is None:
        return value.to(dtype)
    source = value.contiguous()
    output = torch.empty_like(source, dtype=dtype)
    _masked_cast_kernel()[((source.numel() + 1023) // 1024,)](
        source, mask, output, source.numel(), source.shape[-1], 1024,
    )
    return output


class _TensorCoreLinear(torch.autograd.Function):
    """BF16 Tensor Core linear map with FP32 accumulation and dtype-aware stores."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype | None,
        fuse_weight_bias_grad: bool,
        output_mask: torch.Tensor | None,
        input_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        value_bf16 = _masked_cast(value, input_mask, torch.bfloat16)
        weight_bf16 = weight.to(dtype=torch.bfloat16)
        ctx.save_for_backward(value_bf16, weight_bf16, output_mask, input_mask)
        ctx.value_shape = value.shape
        ctx.value_dtype = value.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.bias_device = None if bias is None else bias.device
        ctx.has_bias = bias is not None
        ctx.fuse_weight_bias_grad = fuse_weight_bias_grad

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
            output_mask=output_mask,
        ).reshape(value.shape[:-1] + (weight.shape[0],))
        return output if output_dtype is None else output.to(dtype=output_dtype)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        value, weight, output_mask, input_mask = ctx.saved_tensors
        # Match AOTAutograd: release ownership only when retain_graph is false.
        ctx.maybe_clear_saved_tensors()
        grad_output = _masked_cast(grad_output, output_mask, grad_output.dtype)
        flat_value = value.reshape(-1, value.shape[-1])
        flat_gradient = grad_output.to(dtype=torch.bfloat16).reshape(
            -1,
            weight.shape[0],
        )

        grad_weight = grad_bias = None
        # Bias uses the original upstream gradient, not a BF16-rounded FP16/
        # FP32 tensor. Fuse only when both VJPs consume the same BF16 values.
        fused = (
            ctx.fuse_weight_bias_grad and grad_output.dtype == torch.bfloat16
            and ctx.needs_input_grad[1] and ctx.has_bias and ctx.needs_input_grad[2]
        )
        if fused:
            grad_weight, grad_bias = torch.ops.memsolve_equilibrium.linear_weight_bias(
                flat_value.contiguous(), flat_gradient.contiguous(),
            )
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype)
            grad_bias = grad_bias.to(device=ctx.bias_device, dtype=ctx.bias_dtype)
        elif ctx.needs_input_grad[1]:
            grad_weight = _wide_gemm(
                flat_gradient.mT,
                flat_value,
            ).to(dtype=ctx.weight_dtype)

        if not fused and ctx.has_bias and ctx.needs_input_grad[2]:
            # Preserve the original FP32 bias-reduction boundary without
            # materializing a full FP32 copy of a half-precision upstream VJP.
            dimensions = tuple(range(grad_output.ndim - 1))
            grad_bias = (
                grad_output.sum(dim=dimensions, dtype=torch.float32)
                if dimensions
                else grad_output.to(dtype=torch.float32)
            ).to(device=ctx.bias_device, dtype=ctx.bias_dtype)

        del flat_value, value, grad_output
        grad_value = None
        if ctx.needs_input_grad[0]:
            grad_value = _wide_gemm(
                flat_gradient,
                weight,
                output_dtype=(
                    torch.bfloat16 if ctx.value_dtype == torch.bfloat16 else torch.float32
                ),
                output_mask=input_mask,
            ).reshape(ctx.value_shape)
            grad_value = grad_value.to(dtype=ctx.value_dtype)

        return grad_value, grad_weight, grad_bias, None, None, None, None


@lru_cache(maxsize=1)
def _no_frame_kernels():
    # Lazily import the Triton supplied by CUDA PyTorch, as for projections.
    from triton import jit
    import triton.language as tl

    @jit
    def _mixed_dot(a, b):
        # Two BF16 terms retain the compact FP32 coefficient's low bits
        # (Henry et al., arXiv:1904.06376); the token operand is already BF16.
        hi = b.to(tl.bfloat16)
        lo = (b - hi.to(tl.float32)).to(tl.bfloat16)
        return tl.dot(a, hi) + tl.dot(a, lo)

    @jit
    def _wide_dot(a, b):
        # Three BF16 products retain both rotated FP32 operands' low bits.
        # The omitted low-low product is second order (Henry et al. 2019).
        ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        # Reuse one accumulator instead of keeping three result tiles live.
        result = tl.dot(ah, bh)
        result = tl.dot(ah, bl, result)
        return tl.dot(al, bh, result)

    @jit
    def _rotate(value, COS, SIN, batch, tokens, channel_start,
                N: tl.constexpr, D: tl.constexpr,
                ROTARY_BATCHED: tl.constexpr, INVERSE: tl.constexpr = False):
        # Adjacent-pair split/join follows FlashAttention (BSD-3-Clause):
        # https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
        # Keep the rotation and inverse rotation in FP32 until the final store.
        pairs = channel_start // 2 + tl.arange(0, value.shape[1] // 2)
        table_batch = batch if ROTARY_BATCHED else 0
        offset = ((table_batch * N + tokens[:, None]) * (D // 2) + pairs[None, :])
        mask = (tokens[:, None] < N) & (pairs[None, :] < D // 2)
        cosine = tl.load(COS + offset, mask, 1.0)
        sine = tl.load(SIN + offset, mask, 0.0)
        if INVERSE:
            sine = -sine
        even, odd = tl.split(tl.reshape(value.to(tl.float32),
                                       (value.shape[0], value.shape[1] // 2, 2)))
        return tl.reshape(tl.join(even * cosine - odd * sine,
                                  even * sine + odd * cosine), value.shape)

    @jit
    def _statistics(
        X,
        E,
        COUNT,
        G,
        Q,
        DE,
        COS,
        SIN,
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        T: tl.constexpr,
        GRAD: tl.constexpr,
        HAS_COUNT: tl.constexpr,
        HAS_ROTARY: tl.constexpr,
        ROTARY_BATCHED: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
    ):
        s = tl.program_id(0)
        tile = tl.program_id(1)
        b = s // H
        h = s % H
        count = tl.load(COUNT + b) if HAS_COUNT else float(N)
        n = tile * BT + tl.arange(0, BT)
        r = tl.arange(0, BR)
        d = tl.arange(0, BD)
        a = tl.load(
            X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
            (n[:, None] < N) & (r[None, :] < R),
            0,
        )
        if not GRAD:
            g = tl.dot(tl.trans(a), a).to(tl.float32) / count
            tl.store(
                G + ((s * T + tile) * R + r[:, None]) * R + r[None, :],
                g, (r[:, None] < R) & (r[None, :] < R),
            )
        de = tl.full((), 0.0, tl.float32)
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            c = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D), 0,
            )
            if GRAD:
                e = tl.load(
                    E + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D), 0,
                )
                if HAS_ROTARY:
                    rotated = _rotate(e, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED)
                    q = _mixed_dot(tl.trans(a), rotated) / tl.sqrt(count)
                else:
                    q = tl.dot(tl.trans(a), e).to(tl.float32) / tl.sqrt(count)
                de += tl.sum(tl.sum(e.to(tl.float32) * c.to(tl.float32), 0), 0)
            else:
                if HAS_ROTARY:
                    rotated = _rotate(c, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED)
                    q = _mixed_dot(tl.trans(a), rotated) / tl.sqrt(count)
                else:
                    q = tl.dot(tl.trans(a), c).to(tl.float32) / tl.sqrt(count)
            tl.store(
                Q + ((s * T + tile) * R + r[:, None]) * D + d[None, :],
                q, (r[:, None] < R) & (d[None, :] < D),
            )
        if GRAD:
            tl.store(DE + s * T + tile, de)

    @jit
    def _readout(
        X,
        V,
        ETA,
        COUNT,
        Y,
        COS,
        SIN,
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        HAS_COUNT: tl.constexpr,
        HAS_ROTARY: tl.constexpr,
        ROTARY_BATCHED: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
        UNSCALED_COMPLEMENT: tl.constexpr = False,
    ):
        s = tl.program_id(0)
        tile = tl.program_id(1)
        b = s // H
        h = s % H
        count = tl.load(COUNT + b) if HAS_COUNT else float(N)
        n = tile * BT + tl.arange(0, BT)
        r = tl.arange(0, BR)
        d = tl.arange(0, BD)
        a = tl.load(
            X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
            (n[:, None] < N) & (r[None, :] < R),
            0,
        )
        eta = tl.load(ETA + h)
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            c = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            v = tl.load(
                V + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            correction = _mixed_dot(a, v)
            if UNSCALED_COMPLEMENT:
                correction = -eta * correction
            if HAS_ROTARY:
                correction = _rotate(correction, COS, SIN, b, n, channel_tile * BD,
                                     N, D, ROTARY_BATCHED, True)
            y = eta * c + correction / tl.sqrt(count)
            tl.store(
                Y + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                y,
                (n[:, None] < N) & (d[None, :] < D),
            )

    @jit
    def _token_backward(
        X,
        E,
        V,
        DG,
        DQ,
        ETA,
        COUNT,
        DX,
        COS,
        SIN,
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        HAS_COUNT: tl.constexpr,
        HAS_ROTARY: tl.constexpr,
        ROTARY_BATCHED: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
        BT: tl.constexpr,
        UNSCALED_COMPLEMENT: tl.constexpr = False,
    ):
        s = tl.program_id(0)
        tile = tl.program_id(1)
        b = s // H
        h = s % H
        count = tl.load(COUNT + b) if HAS_COUNT else float(N)
        n = tile * BT + tl.arange(0, BT)
        r = tl.arange(0, BR)
        d = tl.arange(0, BD)
        a = tl.load(
            X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
            (n[:, None] < N) & (r[None, :] < R),
            0,
        )
        g = tl.load(
            DG + (s * R + r[:, None]) * R + r[None, :],
            (r[:, None] < R) & (r[None, :] < R),
            0,
        )
        g = (g + tl.trans(g)).to(tl.float32)
        inv = 1.0 / tl.sqrt(count)
        eta = tl.load(ETA + h)
        da = _mixed_dot(a, g) * inv
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            e = tl.load(
                E + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D),
                0,
            )
            v = tl.load(
                V + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            # Finish E/V before loading C/DQ to limit live rotated operands.
            if HAS_ROTARY:
                rotated_e = _rotate(e, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED)
                dc = eta * e.to(tl.float32)
                direct = _wide_dot(rotated_e, tl.trans(v))
                da += -eta * direct if UNSCALED_COMPLEMENT else direct
                c = tl.load(
                    X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0,
                )
                dq = tl.load(
                    DQ + (s * R + r[:, None]) * D + d[None, :],
                    (r[:, None] < R) & (d[None, :] < D),
                    0,
                ).to(tl.float32)
                rotated_c = _rotate(c, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED)
                da += _wide_dot(rotated_c, tl.trans(dq))
                correction = _rotate(_mixed_dot(a, dq), COS, SIN, b, n, channel_tile * BD,
                                     N, D, ROTARY_BATCHED, True)
                dc += correction * inv
            else:
                direct = _mixed_dot(e, tl.trans(v))
                da += -eta * direct if UNSCALED_COMPLEMENT else direct
                dc = eta * e.to(tl.float32)
                c = tl.load(
                    X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0,
                )
                dq = tl.load(
                    DQ + (s * R + r[:, None]) * D + d[None, :],
                    (r[:, None] < R) & (d[None, :] < D),
                    0,
                ).to(tl.float32)
                da += _mixed_dot(c, tl.trans(dq))
                dc += _mixed_dot(a, dq) * inv
            tl.store(
                DX + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                dc, (n[:, None] < N) & (d[None, :] < D),
            )
        tl.store(
            DX + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
            da * inv, (n[:, None] < N) & (r[None, :] < R),
        )

    @jit
    def _chunk_statistics(
        X, E, COUNT, G, Q, DE, COS, SIN,
        N: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        T: tl.constexpr, GRAD: tl.constexpr, HAS_COUNT: tl.constexpr,
        HAS_ROTARY: tl.constexpr, ROTARY_BATCHED: tl.constexpr,
        BR: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr,
    ):
        # SSD-style chunk accumulation reduces partial-state traffic without
        # holding a full token chunk in registers. Statistics remain FP32.
        BK: tl.constexpr = 64
        s, tile = tl.program_id(0), tl.program_id(1)
        b, h = s // H, s % H
        count = tl.load(COUNT + b) if HAS_COUNT else float(N)
        r = tl.arange(0, BR)
        gram = tl.zeros((BR, BR), tl.float32)
        de = tl.full((), 0.0, tl.float32)
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            q_hi = tl.zeros((BR, BD), tl.float32)
            q_lo = tl.zeros((BR, BD), tl.float32)
            for step in range(tl.cdiv(BT, BK)):
                n = tile * BT + step * BK + tl.arange(0, BK)
                a = tl.load(
                    X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                    (n[:, None] < N) & (r[None, :] < R), 0,
                )
                if not GRAD and channel_tile == 0:
                    gram = tl.dot(tl.trans(a), a, gram)
                c = tl.load(
                    X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D), 0,
                )
                if GRAD:
                    value = tl.load(
                        E + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                        (n[:, None] < N) & (d[None, :] < D), 0,
                    )
                    de += tl.sum(tl.sum(value.to(tl.float32) * c.to(tl.float32), 0), 0)
                else:
                    value = c
                if HAS_ROTARY:
                    rotated = _rotate(value, COS, SIN, b, n, channel_tile * BD,
                                      N, D, ROTARY_BATCHED)
                    hi = rotated.to(tl.bfloat16)
                    lo = (rotated - hi.to(tl.float32)).to(tl.bfloat16)
                    # Keep the low-word sum separate from the high-word sum.
                    q_hi = tl.dot(tl.trans(a), hi, q_hi)
                    q_lo = tl.dot(tl.trans(a), lo, q_lo)
                else:
                    q_hi = tl.dot(tl.trans(a), value, q_hi)
            tl.store(
                Q + ((s * T + tile) * R + r[:, None]) * D + d[None, :],
                (q_hi + q_lo) / tl.sqrt(count),
                (r[:, None] < R) & (d[None, :] < D),
            )
        if not GRAD:
            tl.store(
                G + ((s * T + tile) * R + r[:, None]) * R + r[None, :],
                gram / count, (r[:, None] < R) & (r[None, :] < R),
            )
        if GRAD:
            tl.store(DE + s * T + tile, de)

    return _statistics, _readout, _token_backward, _chunk_statistics


def _no_frame_block_size(size: int) -> int:
    return max(16, 1 << (size - 1).bit_length())


def _no_frame_statistics(projected, heads, rank, gradient=None, counts=None,
                         rotary_cos=None, rotary_sin=None):
    batch, length, width = projected.shape
    head_dim = (width - heads * rank) // heads
    # Fuse short statistics without a partial reduction. Wide-rank gradient
    # tiles need eight warps to avoid register spills; channel tiles stay <=64.
    token_tile = _no_frame_block_size(length) if length <= 256 else 128
    # Rotated FP32 content needs more live registers than the BF16-only path.
    if rotary_cos is not None:
        token_tile = min(token_tile, 128)
    kernel = _no_frame_kernels()[0]
    # Use larger partials only when their grid still supplies two CTAs per SM.
    if length >= 512:
        chunk_tiles = (length + 511) // 512
        multiprocessors = torch.cuda.get_device_properties(projected.device).multi_processor_count
        if batch * heads * chunk_tiles >= 2 * multiprocessors:
            token_tile = 512
            kernel = _no_frame_kernels()[3]
    tiles = (length + token_tile - 1) // token_tile
    options = dict(device=projected.device, dtype=torch.float32)
    cross = torch.empty((batch * heads, tiles, rank, head_dim), **options)
    if gradient is None:
        gram = torch.empty((batch * heads, tiles, rank, rank), **options)
        eta_gradient = cross
    else:
        gram = cross
        eta_gradient = torch.empty((batch * heads, tiles), **options)
    kernel[(batch * heads, tiles)](
        projected,
        gradient,
        counts,
        gram,
        cross,
        eta_gradient,
        rotary_cos,
        rotary_sin,
        length,
        heads,
        rank,
        head_dim,
        tiles,
        gradient is not None,
        counts is not None,
        rotary_cos is not None,
        rotary_cos is not None and rotary_cos.shape[0] > 1,
        _no_frame_block_size(rank),
        min(64, _no_frame_block_size(head_dim)),
        token_tile,
        num_warps=(
            8
            if gradient is not None
            and ((token_tile == 256 and rank >= 32) or (rotary_cos is not None and rank == 32))
            else 4
        ),
        num_stages=1,
    )
    if gradient is None:
        del eta_gradient
        cross = (cross[:, 0] if tiles == 1 else cross.sum(1)).view(batch, heads, rank, head_dim)
        return (gram[:, 0] if tiles == 1 else gram.sum(1)).view(batch, heads, rank, rank), cross
    del gram
    cross = (cross[:, 0] if tiles == 1 else cross.sum(1)).view(batch, heads, rank, head_dim)
    return cross, (
        eta_gradient[:, 0].view(batch, heads).sum(0)
        if tiles == 1
        else eta_gradient.view(batch, heads, tiles).sum((0, 2))
    )


@lru_cache(maxsize=1)
def _compact_kernels():
    from triton import jit
    import triton.language as tl
    from triton.language.extra import libdevice

    @jit
    def core_forward(
        BASE, FACTOR, SYSTEM, DIFFERENCE,
        R: tl.constexpr, BR: tl.constexpr,
    ):
        h = tl.program_id(0)
        i = tl.arange(0, BR)
        mask = (i[:, None] < R) & (i[None, :] < R)
        offset = h * R * R + i[:, None] * R + i[None, :]
        b = tl.load(BASE + offset, mask, 0.0)
        shifted = b + 0.5413248546129181
        diagonal = tl.where(shifted > 20.0, shifted, libdevice.log1p(libdevice.exp(shifted)))
        f = tl.where(i[:, None] > i[None, :], b, tl.where(i[:, None] == i[None, :], diagonal, 0.0))
        tl.store(FACTOR + offset, f, mask)
        gram = tl.dot(f, tl.trans(f), input_precision="tf32x3")
        bt = tl.load(BASE + h * R * R + i[None, :] * R + i[:, None], mask, 0.0)
        skew = tl.where(i[:, None] < i[None, :], b, tl.where(i[:, None] > i[None, :], -bt, 0.0))
        k = gram + skew
        identity = (i[:, None] == i[None, :]).to(tl.float32)
        tl.store(SYSTEM + offset, identity + k, mask)
        tl.store(DIFFERENCE + offset, identity - k, mask)

    @jit
    def core_backward(
        ADJOINT, MAPPING, FACTOR, BASE, OUT,
        R: tl.constexpr, BR: tl.constexpr,
    ):
        h = tl.program_id(0)
        i = tl.arange(0, BR)
        mask = (i[:, None] < R) & (i[None, :] < R)
        offset = h * R * R + i[:, None] * R + i[None, :]
        u = tl.load(ADJOINT + offset, mask, 0.0)
        m = tl.load(MAPPING + offset, mask, 0.0)
        diagonal = (i[:, None] == i[None, :]) & (i[:, None] < R)
        m_plus_i = m + tl.where(diagonal, 1.0, 0.0)
        g = tl.dot(-u, tl.trans(m_plus_i), input_precision="tf32x3")
        gt = tl.trans(g)
        f = tl.load(FACTOR + offset, mask, 0.0)
        core_grad = tl.dot(g + gt, f, input_precision="tf32x3")
        b = tl.load(BASE + offset, mask, 0.0)
        shifted = b + 0.5413248546129181
        sigmoid = tl.where(shifted > 20.0, 1.0, 1.0 / (1.0 + libdevice.exp(-shifted)))
        d = tl.where(
            i[:, None] > i[None, :],
            core_grad,
            tl.where(i[:, None] == i[None, :], core_grad * sigmoid, g - gt),
        )
        tl.store(OUT + offset, d, mask)

    @jit
    def complement(RAW, ETA, DERIVATIVE, H: tl.constexpr, BLOCK: tl.constexpr):
        h = tl.arange(0, BLOCK)
        raw = tl.load(RAW + h, h < H, 0)
        exponent = libdevice.exp(-2.0 * tl.abs(raw))
        scale: tl.constexpr = 1.0 - 1.1920928955078125e-7
        # Native tanh avoids cancellation near zero. Keep the exponential
        # derivative separately: 1-tanh(raw)^2 loses the FP32 tail gradient.
        value = scale * libdevice.tanh(raw)
        derivative = scale * 4.0 * exponent / ((1.0 + exponent) * (1.0 + exponent))
        tl.store(ETA + h, value, h < H)
        tl.store(DERIVATIVE + h, derivative, h < H)

    return core_forward, core_backward, complement


def _compact_core_forward(base):
    core_factor = torch.empty_like(base, memory_format=torch.contiguous_format)
    system = torch.empty_like(core_factor)
    difference = torch.empty_like(core_factor)
    rank = base.shape[-1]
    _compact_kernels()[0][(base.shape[0],)](
        base, core_factor, system, difference,
        rank, _no_frame_block_size(rank), num_warps=8 if rank > 32 else 4, num_stages=1,
    )
    return core_factor, system, difference


def _compact_core_backward(adjoint, mapping, core_factor, base):
    output = torch.empty_like(base, memory_format=torch.contiguous_format)
    rank = base.shape[-1]
    _compact_kernels()[1][(base.shape[0],)](
        adjoint, mapping, core_factor, base, output,
        rank, _no_frame_block_size(rank), num_warps=4, num_stages=1,
    )
    return output


def _compact_complement(raw):
    eta, derivative = torch.empty_like(raw), torch.empty_like(raw)
    _compact_kernels()[2][(1,)](raw, eta, derivative, raw.numel(), _no_frame_block_size(raw.numel()), num_warps=4)
    return eta, derivative


def _compact_lu(system):
    return torch.ops.memsolve_equilibrium.compact_lu(system.contiguous())[:2]


def _compact_getrs(lu, pivots, rhs, *, transpose=False):
    return torch.ops.memsolve_equilibrium.compact_getrs(
        lu, pivots, rhs.contiguous(), transpose
    )


class _NoFrameSolve(torch.autograd.Function):
    """MathDx partial-pivot LU, reused by the implicit first-order VJP."""

    @staticmethod
    def forward(ctx, system, rhs):
        lu, pivots = _compact_lu(system)
        result = _compact_getrs(lu, pivots, rhs)
        ctx.save_for_backward(lu, pivots, result)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        lu, pivots, result = ctx.saved_tensors
        adjoint = _compact_getrs(lu, pivots, gradient, transpose=True)
        return -(adjoint @ result.mT), adjoint


@lru_cache(maxsize=1)
def _compact_product_kernel():
    from triton import jit
    import triton.language as tl

    @jit
    def product(A, B, C, ETA, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                SHARED_A: tl.constexpr, SHARED_B: tl.constexpr, K_TILE: tl.constexpr,
                BM: tl.constexpr, BC: tl.constexpr, SUBTRACT_STATE: tl.constexpr,
                NEGATE: tl.constexpr, SYMMETRIZE: tl.constexpr, SUBTRACT_FROM: tl.constexpr):
        s, tile = tl.program_id(0), tl.program_id(1)
        a_batch = s % H if SHARED_A else s
        b_batch = s % H if SHARED_B else s
        m = tl.arange(0, BM)
        n = tile * BC + tl.arange(0, BC)
        kk = tl.arange(0, K_TILE)
        acc = tl.full((BM, BC), 0, tl.float32)
        for start in range(tl.cdiv(K, K_TILE)):
            k = start * K_TILE + kk
            a = tl.load(A + a_batch*M*K + m[:, None]*AM + k[None, :]*AK,
                        (m[:, None] < M) & (k[None, :] < K), 0)
            b = tl.load(B + b_batch*K*N + k[:, None]*BK + n[None, :]*BN,
                        (k[:, None] < K) & (n[None, :] < N), 0)
            acc = tl.dot(a, b, acc, input_precision="tf32x3")
        if SUBTRACT_STATE:
            state = tl.load(B + b_batch*K*N + m[:, None]*BK + n[None, :]*BN,
                            (m[:, None] < M) & (n[None, :] < N), 0)
            acc -= tl.load(ETA + s % H) * state
        if SYMMETRIZE:
            acc = 0.5 * (acc + tl.trans(acc))
        if NEGATE:
            acc = -acc
        if SUBTRACT_FROM:
            previous = tl.load(C + (s*M + m[:, None])*N + n[None, :],
                               (m[:, None] < M) & (n[None, :] < N), 0)
            acc = previous - acc
        tl.store(C + (s*M + m[:, None])*N + n[None, :], acc,
                 (m[:, None] < M) & (n[None, :] < N))
    return product


def _compact_product(left, right, *, subtract_eta=None, negative=False,
                     symmetrize=False, subtract_from=None, out=None):
    # out is unsaved scratch: right aliasing needs a square left map; left
    # aliasing needs one full output tile per system. Each CTA reads before storing.
    # TF32x3 compact products, as in FLA; broadcast the shared head map in-kernel.
    batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    shape = (*batch_shape, left.shape[-2], right.shape[-1])
    if out is not None:
        output = out
    elif subtract_from is None:
        output = torch.empty(shape, device=left.device, dtype=torch.float32)
    else:
        # Subtract the product in place from the fresh compact adjoint.
        if tuple(subtract_from.shape) != shape or not subtract_from.is_contiguous():
            raise ValueError("compact product accumulator must match its contiguous output")
        output = subtract_from
    heads = batch_shape[-1]
    systems = output.numel() // (left.shape[-2] * right.shape[-1])
    # Use 16-wide inner tiles for r16/r48 to avoid padding the rank dimension.
    columns = min(64, _no_frame_block_size(right.shape[-1]))
    if symmetrize and (left.shape[-2] != right.shape[-1] or columns < right.shape[-1]):
        raise ValueError("symmetric compact epilogue requires one complete square tile")
    _compact_product_kernel()[(systems, (right.shape[-1]+columns-1)//columns)](
        left, right, output, subtract_eta, heads, left.shape[-2], right.shape[-1], left.shape[-1],
        *left.stride()[-2:], *right.stride()[-2:], left.ndim == 3, right.ndim == 3,
        16 if left.shape[-1] % 32 == 16 else 32,
        _no_frame_block_size(left.shape[-2]), columns, subtract_eta is not None,
        negative, symmetrize, subtract_from is not None,
        num_warps=4, num_stages=1,
    )
    return output


@lru_cache(maxsize=1)
def _compact_adjoint_kernels():
    from triton import jit
    import triton.language as tl

    @jit
    def eta_trace(G, DIRECT, DERIVATIVE, OUT, R: tl.constexpr, BR: tl.constexpr):
        h = tl.program_id(0)
        i = tl.arange(0, BR)
        trace = tl.sum(tl.load(G + h*R*R + i*(R+1), i < R, 0), 0)
        tl.store(OUT+h, (tl.load(DIRECT+h)-trace)*tl.load(DERIVATIVE+h))

    @jit
    def cholesky_product(F, H, ETA, INVERSE, HEADS: tl.constexpr, R: tl.constexpr,
                         BR: tl.constexpr, COMPLEMENT: tl.constexpr, FUSE_CONJUGATE: tl.constexpr):
        # H=U Z^T is fresh, unsaved storage, owned in full by one CTA.
        # L^T dL = -M H^T - M^T H, M=F-eta I. Apply the Cholesky
        # symmetric lower projection here, and for R<=32 fuse T^T @ sym @ T in registers.
        s = tl.program_id(0)
        h = s % HEADS
        i = tl.arange(0, BR)
        eta = tl.load(ETA+h)
        mask_rr = (i[:, None] < R) & (i[None, :] < R)
        if COMPLEMENT:
            value = tl.load(H+s*R*R+i[:, None]*R+i[None, :], mask_rr, 0)
            product = eta * (value + tl.trans(value))
        else:
            # Accumulate products serially to reduce register pressure at r48/r64.
            k = tl.arange(0, 16)
            product = tl.full((BR, BR), 0, tl.float32)
            for start in range(tl.cdiv(R, 16)):
                kk = start*16+k
                f = tl.load(F+h*R*R+i[:, None]*R+kk[None, :],
                            (i[:, None] < R) & (kk[None, :] < R), 0)
                diagonal = (i[:, None] == kk[None, :]) & (i[:, None] < R)
                f -= tl.where(diagonal, eta, 0)
                ht = tl.load(H+s*R*R+i[None, :]*R+kk[:, None],
                             (kk[:, None] < R) & (i[None, :] < R), 0)
                product = tl.dot(-f, ht, product, input_precision="tf32x3")
                ft = tl.load(F+h*R*R+kk[None, :]*R+i[:, None],
                             (i[:, None] < R) & (kk[None, :] < R), 0)
                ft -= tl.where(diagonal, eta, 0)
                hn = tl.load(H+s*R*R+kk[:, None]*R+i[None, :],
                             (kk[:, None] < R) & (i[None, :] < R), 0)
                product = tl.dot(-ft, hn, product, input_precision="tf32x3")
        lower = tl.where(
            i[:, None] > i[None, :],
            0.5 * product,
            tl.where(i[:, None] == i[None, :], 0.25 * product, 0.0),
        )
        symmetric = lower + tl.trans(lower)
        if FUSE_CONJUGATE:
            t = tl.load(INVERSE + s*R*R + i[:, None]*R + i[None, :], mask_rr, 0)
            mid = tl.dot(symmetric, t, input_precision="tf32x3")
            tt = tl.load(INVERSE + s*R*R + i[None, :]*R + i[:, None], mask_rr, 0)
            gram = tl.dot(tt, mid, input_precision="tf32x3")
            tl.store(H+s*R*R+i[:, None]*R+i[None, :], gram, mask_rr)
        else:
            tl.store(H+s*R*R+i[:, None]*R+i[None, :], symmetric, mask_rr)

    @jit
    def compact_chain(
        INVERSE, X, MAPPING, ETA, FIRST, SECOND,
        H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        XM: tl.constexpr, XN: tl.constexpr,
        K_TILE: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
        COMPLEMENT: tl.constexpr, TRANSPOSE_MAP: tl.constexpr,
        STORE_FIRST: tl.constexpr = True,
    ):
        s, tile = tl.program_id(0), tl.program_id(1)
        h = s % H
        i = tl.arange(0, BR)
        col = tile * BC + tl.arange(0, BC)
        mask_rd = (i[:, None] < R) & (col[None, :] < D)
        if R <= 32:
            j = tl.arange(0, BR)
            mask_rr = (i[:, None] < R) & (j[None, :] < R)
            t = tl.load(INVERSE + s * R * R + i[:, None] * R + j[None, :], mask_rr, 0.0)
            x = tl.load(
                X + s * R * D + j[:, None] * XM + col[None, :] * XN,
                (j[:, None] < R) & (col[None, :] < D),
                0.0,
            )
            u = tl.dot(t, x, input_precision="tf32x3")
            if STORE_FIRST:
                tl.store(FIRST + (s * R + i[:, None]) * D + col[None, :], u, mask_rd)
            if COMPLEMENT:
                mid = -tl.load(ETA + h) * u if TRANSPOSE_MAP else u
            else:
                if TRANSPOSE_MAP:
                    m_mat = tl.load(MAPPING + h * R * R + j[None, :] * R + i[:, None], mask_rr, 0.0)
                else:
                    m_mat = tl.load(MAPPING + h * R * R + i[:, None] * R + j[None, :], mask_rr, 0.0)
                diagonal = (i[:, None] == j[None, :]) & (i[:, None] < R)
                m_mat -= tl.where(diagonal, tl.load(ETA + h), 0.0)
                mid = tl.dot(m_mat, u, input_precision="tf32x3")
            tt = tl.load(INVERSE + s * R * R + j[None, :] * R + i[:, None], mask_rr, 0.0)
            out = tl.dot(tt, mid, input_precision="tf32x3")
            tl.store(SECOND + (s * R + i[:, None]) * D + col[None, :], out, mask_rd)
        else:
            kk = tl.arange(0, K_TILE)
            u = tl.full((BR, BC), 0.0, tl.float32)
            for start in range(tl.cdiv(R, K_TILE)):
                k = start * K_TILE + kk
                t = tl.load(
                    INVERSE + s * R * R + i[:, None] * R + k[None, :],
                    (i[:, None] < R) & (k[None, :] < R),
                    0.0,
                )
                x = tl.load(
                    X + s * R * D + k[:, None] * XM + col[None, :] * XN,
                    (k[:, None] < R) & (col[None, :] < D),
                    0.0,
                )
                u = tl.dot(t, x, u, input_precision="tf32x3")
            tl.store(FIRST + (s * R + i[:, None]) * D + col[None, :], u, mask_rd)
            tl.debug_barrier()
            if COMPLEMENT:
                mid = -tl.load(ETA + h) * u if TRANSPOSE_MAP else u
            else:
                eta = tl.load(ETA + h)
                mid = tl.full((BR, BC), 0.0, tl.float32)
                for start in range(tl.cdiv(R, K_TILE)):
                    k = start * K_TILE + kk
                    if TRANSPOSE_MAP:
                        m_tile = tl.load(
                            MAPPING + h * R * R + k[None, :] * R + i[:, None],
                            (i[:, None] < R) & (k[None, :] < R),
                            0.0,
                        )
                    else:
                        m_tile = tl.load(
                            MAPPING + h * R * R + i[:, None] * R + k[None, :],
                            (i[:, None] < R) & (k[None, :] < R),
                            0.0,
                        )
                    diagonal = (i[:, None] == k[None, :]) & (i[:, None] < R)
                    m_tile -= tl.where(diagonal, eta, 0.0)
                    u_tile = tl.load(
                        FIRST + (s * R + k[:, None]) * D + col[None, :],
                        (k[:, None] < R) & (col[None, :] < D),
                        0.0,
                    )
                    mid = tl.dot(m_tile, u_tile, mid, input_precision="tf32x3")
            tl.store(SECOND + (s * R + i[:, None]) * D + col[None, :], mid, mask_rd)
            tl.debug_barrier()
            out = tl.full((BR, BC), 0.0, tl.float32)
            for start in range(tl.cdiv(R, K_TILE)):
                k = start * K_TILE + kk
                tt = tl.load(
                    INVERSE + s * R * R + k[None, :] * R + i[:, None],
                    (i[:, None] < R) & (k[None, :] < R),
                    0.0,
                )
                mid_tile = tl.load(
                    SECOND + (s * R + k[:, None]) * D + col[None, :],
                    (k[:, None] < R) & (col[None, :] < D),
                    0.0,
                )
                out = tl.dot(tt, mid_tile, out, input_precision="tf32x3")
            tl.store(SECOND + (s * R + i[:, None]) * D + col[None, :], out, mask_rd)

    @jit
    def compact_chain_bwd_r32(
        INVERSE, GRAD, STATE, MAPPING, ETA, PRODUCT, CROSS_GRAD,
        H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        GM: tl.constexpr, GN: tl.constexpr,
        BR: tl.constexpr, BC: tl.constexpr, COMPLEMENT: tl.constexpr,
    ):
        s = tl.program_id(0)
        h = s % H
        i = tl.arange(0, BR)
        j = tl.arange(0, BR)
        mask_rr = (i[:, None] < R) & (j[None, :] < R)
        eta = tl.load(ETA + h)
        prod_acc = tl.full((BR, BR), 0.0, tl.float32)
        c = tl.arange(0, BC)
        for start in range(tl.cdiv(D, BC)):
            col = start * BC + c
            mask_rd = (i[:, None] < R) & (col[None, :] < D)
            t = tl.load(INVERSE + s * R * R + i[:, None] * R + j[None, :], mask_rr, 0.0)
            g = tl.load(
                GRAD + s * R * D + j[:, None] * GM + col[None, :] * GN,
                (j[:, None] < R) & (col[None, :] < D),
                0.0,
            )
            u = tl.dot(t, g, input_precision="tf32x3")
            st = tl.load(
                STATE + (s * R + j[:, None]) * D + col[None, :],
                (j[:, None] < R) & (col[None, :] < D),
                0.0,
            )
            prod_acc = tl.dot(u, tl.trans(st), prod_acc, input_precision="tf32x3")
            if COMPLEMENT:
                mid = -eta * u
            else:
                m_mat = tl.load(MAPPING + h * R * R + j[None, :] * R + i[:, None], mask_rr, 0.0)
                diagonal = (i[:, None] == j[None, :]) & (i[:, None] < R)
                m_mat -= tl.where(diagonal, eta, 0.0)
                mid = tl.dot(m_mat, u, input_precision="tf32x3")
            tt = tl.load(INVERSE + s * R * R + j[None, :] * R + i[:, None], mask_rr, 0.0)
            dq = tl.dot(tt, mid, input_precision="tf32x3")
            tl.store(CROSS_GRAD + (s * R + i[:, None]) * D + col[None, :], dq, mask_rd)
        tl.store(PRODUCT + s * R * R + i[:, None] * R + j[None, :], prod_acc, mask_rr)

    @jit
    def compact_conjugate_tiled(
        INVERSE, H_MAT, R: tl.constexpr, K_TILE: tl.constexpr, BR: tl.constexpr,
    ):
        s = tl.program_id(0)
        i = tl.arange(0, BR)
        mask_rr = (i[:, None] < R) & (i[None, :] < R)
        kk = tl.arange(0, K_TILE)
        inter = tl.full((BR, BR), 0.0, tl.float32)
        for start in range(tl.cdiv(R, K_TILE)):
            k = start * K_TILE + kk
            tt = tl.load(
                INVERSE + s * R * R + k[None, :] * R + i[:, None],
                (i[:, None] < R) & (k[None, :] < R),
                0.0,
            )
            sym_tile = tl.load(
                H_MAT + s * R * R + k[:, None] * R + i[None, :],
                (k[:, None] < R) & (i[None, :] < R),
                0.0,
            )
            inter = tl.dot(tt, sym_tile, inter, input_precision="tf32x3")
        tl.store(H_MAT + s * R * R + i[:, None] * R + i[None, :], inter, mask_rr)
        tl.debug_barrier()
        gram_grad = tl.full((BR, BR), 0.0, tl.float32)
        for start in range(tl.cdiv(R, K_TILE)):
            k = start * K_TILE + kk
            inter_tile = tl.load(
                H_MAT + s * R * R + i[:, None] * R + k[None, :],
                (i[:, None] < R) & (k[None, :] < R),
                0.0,
            )
            t = tl.load(
                INVERSE + s * R * R + k[:, None] * R + i[None, :],
                (k[:, None] < R) & (i[None, :] < R),
                0.0,
            )
            gram_grad = tl.dot(inter_tile, t, gram_grad, input_precision="tf32x3")
        tl.store(H_MAT + s * R * R + i[:, None] * R + i[None, :], gram_grad, mask_rr)

    return eta_trace, cholesky_product, compact_chain, compact_chain_bwd_r32, compact_conjugate_tiled


def _compact_chain(inverse, rhs, mapping, eta, *, complement, transpose_map, store_first=True):
    batch, heads, rank, columns = rhs.shape
    second = torch.empty((batch, heads, rank, columns), device=rhs.device, dtype=torch.float32)
    first = torch.empty_like(second) if store_first or rank > 32 else second
    block_r = _no_frame_block_size(rank)
    block_c = min(64, _no_frame_block_size(columns))
    _compact_adjoint_kernels()[2][(batch * heads, (columns + block_c - 1) // block_c)](
        inverse, rhs, mapping, eta, first, second,
        heads, rank, columns, *rhs.stride()[-2:],
        16, block_r, block_c, complement, transpose_map, store_first,
        num_warps=4, num_stages=1,
    )
    return (first if store_first else None), second


def _no_frame_compact(gram, cross, base, raw, mode, *, record=True):
    # Factor I+G once; apply the soft frame through a fused compact chain.
    inverse = torch.ops.memsolve_equilibrium.compact_inverse(gram)
    del gram
    eta, eta_derivative = _compact_complement(raw)
    if not record:
        del eta_derivative
        eta_derivative = None
    coordinates = core_factor = lu = pivots = mapping = None
    if mode != 2:
        coordinates = base if record else None
        core_factor, system, difference = _compact_core_forward(base)
        if not record:
            del core_factor
            core_factor = None
        lu, pivots = _compact_lu(system)
        del system
        mapping = _compact_getrs(lu, pivots, difference)
        del difference
        if not record:
            del lu, pivots
            lu = pivots = None
    # Complement keeps an unscaled coefficient: token kernels apply -eta
    # after the compensated product, including eta=0 without division.
    state, coefficient = _compact_chain(
        inverse, cross, mapping, eta, complement=mode == 2, transpose_map=False,
        store_first=record,
    )
    del cross
    if not record:
        return coefficient, eta, ()
    tape = (inverse, state, coordinates, lu, pivots, mapping, eta_derivative, core_factor)
    return coefficient, eta, tape


def _no_frame_compact_backward(coefficient, eta, tape, gradient, eta_gradient, mode):
    inverse, state, coordinates, lu, pivots, mapping, eta_derivative, core_factor = tape
    if isinstance(tape, list):
        tape.clear()
    del tape, coefficient
    batch, heads, rank, columns = state.shape
    block_r = _no_frame_block_size(rank)
    kernels = _compact_adjoint_kernels()
    if rank <= 32:
        product = torch.empty((batch, heads, rank, rank), device=state.device, dtype=torch.float32)
        cross_gradient = torch.empty((batch, heads, rank, columns), device=state.device, dtype=torch.float32)
        block_c = min(64, _no_frame_block_size(columns))
        kernels[3][(batch * heads,)](
            inverse, gradient, state, mapping, eta, product, cross_gradient,
            heads, rank, columns, *gradient.stride()[-2:],
            block_r, block_c, mode == 2,
            num_warps=4, num_stages=1,
        )
        del gradient, state
    else:
        u, cross_gradient = _compact_chain(
            inverse, gradient, mapping, eta, complement=mode == 2, transpose_map=True
        )
        del gradient
        product = _compact_product(u, state.mT)
        del u, state
    map_gradient = product.sum(0)
    raw_gradient = torch.empty_like(eta_gradient)
    kernels[0][(heads,)](map_gradient, eta_gradient, eta_derivative, raw_gradient,
                        rank, block_r, num_warps=4)
    del eta_gradient, eta_derivative
    base_gradient = None
    if mode != 2:
        adjoint = _compact_getrs(lu, pivots, map_gradient, transpose=True)
        del lu, pivots, map_gradient
        base_gradient = _compact_core_backward(adjoint, mapping, core_factor, coordinates)
        del adjoint, core_factor, coordinates
    else:
        del lu, pivots, map_gradient, core_factor, coordinates
    fuse_conjugate = rank <= 32
    kernels[1][(batch * heads,)](
        mapping, product, eta, inverse, heads, rank, block_r,
        mode == 2, fuse_conjugate, num_warps=4, num_stages=1,
    )
    del mapping
    if not fuse_conjugate:
        kernels[4][(batch * heads,)](
            inverse, product, rank, 16 if rank % 32 == 16 else 32, block_r,
            num_warps=4, num_stages=1,
        )
    del inverse
    return product, cross_gradient, base_gradient, raw_gradient


def _no_frame_forward(projected, base, raw, counts, *, record, sequence_parallel=None,
                      rotary_cos=None, rotary_sin=None):
    heads, rank = base.shape[:2]
    batch, length, width = projected.shape
    head_dim = (width - heads * rank) // heads
    mode = 2 if base.numel() == 0 else 1
    stats = list(
        _no_frame_statistics(projected, heads, rank, counts=counts,
                             rotary_cos=rotary_cos, rotary_sin=rotary_sin)
    )
    if sequence_parallel is not None:
        stats[:] = sequence_parallel.sum_tensors(*stats)
    with torch.no_grad():
        coefficient, eta, tape = _no_frame_compact(
            stats.pop(0), stats.pop(0), base, raw, mode, record=record
        )
    output = torch.empty(
        (batch, length, heads * head_dim),
        device=projected.device,
        dtype=projected.dtype,
    )
    _no_frame_kernels()[1][(batch * heads, (length + 127) // 128)](
        projected,
        coefficient,
        eta,
        counts,
        output,
        rotary_cos,
        rotary_sin,
        length,
        heads,
        rank,
        head_dim,
        counts is not None,
        rotary_cos is not None,
        rotary_cos is not None and rotary_cos.shape[0] > 1,
        _no_frame_block_size(rank),
        min(128, _no_frame_block_size(head_dim)),
        128,
        mode == 2,
        num_warps=4,
        num_stages=1,
    )
    if not record:
        return output, ()
    return output, (coefficient, eta, *tape)


class _NoFrameMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, base, raw, counts, sequence_parallel, rotary_cos, rotary_sin):
        ctx.sequence_parallel = sequence_parallel
        output, saved = _no_frame_forward(
            projected, base, raw, counts, record=True, sequence_parallel=sequence_parallel,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
        )
        ctx.save_for_backward(projected, counts, rotary_cos, rotary_sin, *saved)
        ctx.heads, ctx.rank = base.shape[:2]
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        projected, counts, rotary_cos, rotary_sin, coefficient, eta, *tape = ctx.saved_tensors
        # Keep retained graphs intact; otherwise compact tape can die before dX.
        ctx.maybe_clear_saved_tensors()
        heads, rank = ctx.heads, ctx.rank
        batch, length, width = projected.shape
        head_dim = (width - heads * rank) // heads
        gradient = gradient.contiguous()
        stats = list(
            _no_frame_statistics(
                projected,
                heads,
                rank,
                gradient,
                counts,
                rotary_cos,
                rotary_sin,
            )
        )
        sequence_parallel = ctx.sequence_parallel
        if sequence_parallel is not None:
            stats[:] = sequence_parallel.sum_tensors(*stats)
        mode = 2 if tape[2] is None else 1
        with torch.no_grad():
            gram_gradient, cross_gradient, base_gradient, raw_gradient = _no_frame_compact_backward(
                None, eta, tape, stats.pop(0), stats.pop(0), mode
            )
        if sequence_parallel is not None:
            # Compact adjoints already include every shard. Parameter SUM
            # must count these replicated gradients exactly once.
            raw_gradient.div_(sequence_parallel.world_size)
            if base_gradient is not None:
                base_gradient.div_(sequence_parallel.world_size)
        projected_gradient = torch.empty_like(projected)
        # Narrow channel tiles bound the live FP32 rotation/adjoint operands.
        token_tile, channel_tile = 128, 32 if rank <= 32 else 16
        if rotary_cos is not None:
            token_tile, channel_tile = 128, 16
        _no_frame_kernels()[2][(batch * heads, (length + token_tile - 1) // token_tile)](
            projected,
            gradient,
            coefficient,
            gram_gradient,
            cross_gradient,
            eta,
            counts,
            projected_gradient,
            rotary_cos,
            rotary_sin,
            length,
            heads,
            rank,
            head_dim,
            counts is not None,
            rotary_cos is not None,
            rotary_cos is not None and rotary_cos.shape[0] > 1,
            _no_frame_block_size(rank),
            min(channel_tile, _no_frame_block_size(head_dim)),
            token_tile,
            mode == 2,
            num_warps=4,
            num_stages=1,
        )
        del projected, gradient, coefficient, gram_gradient, cross_gradient, eta
        return projected_gradient, base_gradient, raw_gradient, None, None, None, None


def _short_no_frame_forward(projected, base, raw, counts, rotary_cos=None, rotary_sin=None):
    """Use the smaller token system for the cancellation-sensitive base term.

    When N <= r, C-A(I+A^T A)^-1 A^T C = (I+A A^T)^-1 C.
    The latter avoids subtracting nearly equal values for a strong relation.
    Only at most r-by-r systems are formed; no explicit soft frame is needed.
    """
    from .reference import accretive_generator, bounded_complement, _ieee_fp32_matmul, rotate_content

    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    head_dim = (width - heads * rank) // heads
    with (
        torch.autocast(device_type="cuda", enabled=False),
        _ieee_fp32_matmul(projected.device),
    ):
        normalization = (
            counts
            if counts is not None
            else torch.full(
                (batch,),
                float(length),
                device=projected.device,
                dtype=torch.float32,
            )
        )
        relation, content = projected.float().split(
            (heads * rank, heads * head_dim), dim=-1
        )
        relation = relation.reshape(batch, length, heads, rank).transpose(1, 2)
        relation = relation / normalization.sqrt().view(batch, 1, 1, 1)
        content = content.reshape(batch, length, heads, head_dim).transpose(1, 2)
        if rotary_cos is not None:
            content = rotate_content(content, rotary_cos, rotary_sin)
        identity_token = torch.eye(length, device=projected.device, dtype=torch.float32)
        factor_token, _info = torch.linalg.cholesky_ex(
            identity_token + relation @ relation.mT, check_errors=False
        )
        residual = torch.linalg.solve_triangular(factor_token, content, upper=False)
        residual = torch.linalg.solve_triangular(factor_token.mT, residual, upper=True)
        eta = bounded_complement(raw)
        result = eta[None, :, None, None] * residual
        if base.numel():
            identity = torch.eye(rank, device=projected.device, dtype=torch.float32)
            factor, _info = torch.linalg.cholesky_ex(
                identity + relation.mT @ relation, check_errors=False
            )
            state = torch.linalg.solve_triangular(
                factor, relation.mT @ content, upper=False
            )
            generator = accretive_generator(base)
            correction = (
                _NoFrameSolve.apply(identity + generator, identity - generator)
                @ state
            )
            coefficient = torch.linalg.solve_triangular(
                factor.mT, correction, upper=True
            )
            result = result + relation @ coefficient
        if rotary_cos is not None:
            result = rotate_content(result, rotary_cos, rotary_sin, inverse=True)
        return (
            result.transpose(1, 2)
            .reshape(batch, length, heads * head_dim)
            .to(torch.bfloat16)
        )


class _ShortNoFrameMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, base, raw, counts, rotary_cos, rotary_sin):
        with torch.enable_grad():
            leaves = [
                value.detach().requires_grad_(True)
                for value in (projected, base, raw)
            ]
            result = _short_no_frame_forward(*leaves, counts, rotary_cos, rotary_sin)
        ctx.save_for_backward(result, *leaves)
        return result.detach()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        from .reference import _ieee_fp32_matmul

        result, *leaves = ctx.saved_tensors
        with (
            torch.enable_grad(),
            torch.autocast(device_type="cuda", enabled=False),
            _ieee_fp32_matmul(result.device),
        ):
            gradients = torch.autograd.grad(
                result, leaves, gradient, allow_unused=True, retain_graph=True
            )
        return (*gradients, None, None, None)


def _validate_no_frame_inputs(projected, base, raw, counts):
    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    require(projected.is_cuda, "projected must be a CUDA tensor")
    require(
        projected.ndim == 3 and projected.is_contiguous(),
        "projected must be contiguous [B, N, H*R + D]",
    )
    require(
        projected.dtype == torch.bfloat16,
        "projected must use bfloat16 under the mixed precision CUDA contract",
    )
    for name, value in (
        ("core_base_raw", base),
        ("eta_raw", raw),
    ):
        require(
            value.device == projected.device and value.is_contiguous(),
            f"{name} must be contiguous on the same CUDA device",
        )
        require(value.dtype == torch.float32, f"{name} must use float32")
    require(base.ndim == 3, "core_base_raw must have shape [H,R,R] or [H,R,0]")
    batch, length, width = projected.shape
    heads, rank, columns = base.shape
    require(batch > 0 and length > 0 and heads > 0, "B, N, and H must be positive")
    require(
        rank in (16, 32, 48, 64), "the CUDA fast path supports rank in {16,32,48,64}"
    )
    require(columns in (0, rank), "core_base_raw must have shape [H,R,R] or [H,R,0]")
    content_dim = width - heads * rank
    require(
        content_dim > 0 and content_dim % heads == 0,
        "projected content width must be positive and divisible by H",
    )
    require(tuple(raw.shape) == (heads,), "eta_raw must have shape [H]")
    if counts is not None:
        require(
            counts.device == projected.device
            and counts.dtype == torch.float32
            and counts.is_contiguous(),
            "valid_counts must be contiguous float32 on the same CUDA device",
        )
        require(tuple(counts.shape) == (batch,), "valid_counts must have shape [B]")


def fast_mix(
    projected: torch.Tensor,
    core_base_raw: torch.Tensor,
    eta_raw: torch.Tensor,
    valid_counts: torch.Tensor | None = None,
    *, sequence_parallel=None, rotary_cos=None, rotary_sin=None,
) -> torch.Tensor:
    """Apply either mode through compact products without materializing P.

    Solve the base term in token space when N <= r, otherwise in rank space.
    """
    if valid_counts is not None and valid_counts.requires_grad:
        raise ValueError("the MemSolve CUDA fast path does not support gradients for valid_counts")
    require_available()
    _validate_no_frame_inputs(
        projected, core_base_raw, eta_raw, valid_counts
    )
    if (rotary_cos is None) != (rotary_sin is None):
        raise ValueError("rotary_cos and rotary_sin must be supplied together")
    if rotary_cos is not None:
        batch, length, width = projected.shape
        heads, rank = core_base_raw.shape[:2]
        head_dim = (width - heads * rank) // heads
        if head_dim % 2:
            raise ValueError("rotary content requires an even head dimension")
        for name, table in (("rotary_cos", rotary_cos), ("rotary_sin", rotary_sin)):
            if (table.device != projected.device or table.dtype != torch.float32
                    or not table.is_contiguous() or table.requires_grad):
                raise ValueError(f"{name} must be non-differentiable contiguous FP32 on the input device")
            if table.shape not in ((1, length, head_dim // 2), (batch, length, head_dim // 2)):
                raise ValueError(f"{name} must have shape [1 or B,N,Dhead/2]")
        if rotary_cos.shape != rotary_sin.shape:
            raise ValueError("rotary_cos and rotary_sin must share their shape")
    record = torch.is_grad_enabled() and any(
        value.requires_grad
        for value in (projected, core_base_raw, eta_raw)
    )
    arguments = (projected, core_base_raw, eta_raw, valid_counts)
    if sequence_parallel is not None:
        sequence_parallel.validate(projected.shape[1], core_base_raw.shape[1])
        if valid_counts is None:
            raise ValueError("sequence parallelism requires global valid_counts")
    if sequence_parallel is None and projected.shape[1] <= core_base_raw.shape[1]:
        return (_ShortNoFrameMix.apply(*arguments, rotary_cos, rotary_sin) if record
                else _short_no_frame_forward(*arguments, rotary_cos, rotary_sin))
    if record:
        return _NoFrameMix.apply(*arguments, sequence_parallel, rotary_cos, rotary_sin)
    return _no_frame_forward(*arguments, record=False, sequence_parallel=sequence_parallel,
                             rotary_cos=rotary_cos, rotary_sin=rotary_sin)[0]
