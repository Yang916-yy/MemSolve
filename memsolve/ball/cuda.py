from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import importlib
import math
import os
from pathlib import Path
from threading import Lock, local
from typing import Literal, NamedTuple

import torch
from torch.autograd.function import once_differentiable
from torch.fx.experimental.symbolic_shapes import has_static_value

from .config import _validate_recompute_readout

_LOAD_LOCK = Lock()
_RECOMPUTE_LOCAL = local()
_SUPPORTED_ARCHITECTURES = frozenset((80, 86, 87, 89, 90, 100, 120))
_NATIVE_CONTRACT_VERSION = 17
_RUNTIME_PACKAGE = "memsolve_cuda_runtime"
_LOADED_ARCHITECTURE: int | frozenset[int] | None = None


class _RecomputeSpec(NamedTuple):
    projected: torch.Tensor
    coefficient: torch.Tensor
    eta: torch.Tensor
    counts: torch.Tensor | None
    rotary_cos: torch.Tensor | None
    rotary_sin: torch.Tensor | None
    heads: int
    rank: int
    head_dim: int
    mode: int


class _RecomputeScopeState:
    __slots__ = ("policy", "spec")

    def __init__(self, policy: Literal["auto"] | bool) -> None:
        self.policy = _validate_recompute_readout(policy)
        self.spec: _RecomputeSpec | None = None

    @property
    def enabled(self) -> bool:
        return self.policy is not False


@contextmanager
def _recompute_scope(policy: Literal["auto"] | bool = "auto"):
    previous = getattr(_RECOMPUTE_LOCAL, "active", None)
    state = _RecomputeScopeState(policy)
    _RECOMPUTE_LOCAL.active = state
    try:
        yield state
    finally:
        _RECOMPUTE_LOCAL.active = previous


def _resolved_device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


@lru_cache(maxsize=16)
def _multiprocessor_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@lru_cache(maxsize=16)
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _is_hopper_or_newer(device: torch.device) -> bool:
    return _device_capability(_resolved_device_index(device))[0] >= 9


def _is_hopper_wgmma(device: torch.device) -> bool:
    return _device_capability(_resolved_device_index(device))[0] == 9


def _triton_cuda_allocator(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_triton_allocator() -> None:
    from triton.runtime import _allocation

    if isinstance(_allocation._allocator.get(), _allocation.NullAllocator):
        _allocation.set_allocator(_triton_cuda_allocator)


def _tma_aligned_2d(tensor: torch.Tensor, row_stride_elems: int) -> bool:
    return (
        tensor.data_ptr() % 16 == 0
        and (row_stride_elems * tensor.element_size()) % 16 == 0
    )


@lru_cache(maxsize=16)
def _side_stream_bundle(
    device_index: int,
) -> tuple[torch.cuda.Stream, torch.cuda.Event, torch.cuda.Event]:
    with torch.cuda.device(device_index):
        return (
            torch.cuda.Stream(device=device_index),
            torch.cuda.Event(enable_timing=False),
            torch.cuda.Event(enable_timing=False),
        )


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

    if _LOADED_ARCHITECTURE is not None:
        return
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
    if architecture in (103, 110):
        architecture = 100
    elif architecture == 121:
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
    multi_arch = bool(getattr(runtime, "MULTI_ARCH", False)) or (0 in architectures) or ("multi" in architectures)
    if architecture not in architectures and not multi_arch:
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


def _native_supported_architectures() -> frozenset[int] | None:
    """Return the SMs compiled into the loaded native extension, if queried."""

    namespace = getattr(torch.ops, "memsolve_equilibrium", None)
    if namespace is None or not hasattr(namespace, "supported_architectures"):
        return None
    try:
        values = namespace.supported_architectures()
    except RuntimeError:
        return None
    try:
        architectures = frozenset(int(sm) for sm in values)
    except (TypeError, ValueError):
        return None
    return architectures or None


def _architecture_is_supported(
    loaded: int | frozenset[int],
    requested: int,
) -> bool:
    if isinstance(loaded, frozenset):
        return requested in loaded
    return loaded == 0 or loaded == requested


def _format_loaded_architectures(loaded: int | frozenset[int]) -> str:
    if isinstance(loaded, frozenset):
        return ", ".join(f"SM{arch}" for arch in sorted(loaded))
    return "all supported SMs" if loaded == 0 else f"SM{loaded}"


def _normalize_loaded_architectures(
    architectures: frozenset[int],
) -> int | frozenset[int]:
    if architectures >= _SUPPORTED_ARCHITECTURES:
        return 0
    if len(architectures) == 1:
        return next(iter(architectures))
    return architectures


def _is_same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


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
    unified = development.parent / "memsolve_equilibrium.so"
    if unified.is_file():
        return unified
    if development.is_file():
        return development
    packaged = _packaged_library_path(architecture)
    if packaged is not None:
        return packaged
    return development


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
            loaded = _LOADED_ARCHITECTURE
            if loaded is None:
                queried = _native_supported_architectures()
                if queried is not None:
                    loaded = _normalize_loaded_architectures(queried)
            if loaded is not None:
                if not _architecture_is_supported(loaded, requested_architecture):
                    raise RuntimeError(
                        "the loaded MemSolve CUDA extension targets "
                        f"{_format_loaded_architectures(loaded)}, "
                        f"not requested SM{requested_architecture}"
                    )
                if _LOADED_ARCHITECTURE is None:
                    _LOADED_ARCHITECTURE = loaded
            return
        if _native_operator_abi_is_registered():
            _check_native_contract()

        raw_library = (
            Path(path).expanduser()
            if path is not None
            else _default_library_path(device).expanduser()
        )
        expected_name = f"memsolve_equilibrium_sm{requested_architecture}.so"
        allowed_names = (expected_name, "memsolve_equilibrium.so")
        if raw_library.name not in allowed_names:
            raise RuntimeError(
                "the strict MemSolve CUDA fast path requires "
                f"{expected_name} or memsolve_equilibrium.so for "
                f"SM{requested_architecture}, got {raw_library.name}"
            )
        library = raw_library.resolve()
        if not library.is_file():
            raise RuntimeError(
                "the MemSolve CUDA extension was not found at "
                f"{library}; build it with tools/build_cuda.sh or pass its path to load()"
            )

        torch.ops.load_library(str(library))
        _check_native_contract()
        queried = _native_supported_architectures()
        if queried is not None:
            loaded = _normalize_loaded_architectures(queried)
            if not _architecture_is_supported(loaded, requested_architecture):
                raise RuntimeError(
                    "the loaded MemSolve CUDA extension targets "
                    f"{_format_loaded_architectures(loaded)}, "
                    f"not requested SM{requested_architecture}"
                )
            _LOADED_ARCHITECTURE = loaded
        elif (
            raw_library.name == "memsolve_equilibrium.so"
            or library.name == "memsolve_equilibrium.so"
            or _is_same_file(library, library.parent / "memsolve_equilibrium.so")
        ):
            _LOADED_ARCHITECTURE = 0
        else:
            _LOADED_ARCHITECTURE = requested_architecture


_DTYPE_TO_CODE: dict[torch.dtype | None, int] = {
    None: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
    torch.float64: 4,
}
_CODE_TO_DTYPE: dict[int, torch.dtype | None] = {
    code: dtype for dtype, code in _DTYPE_TO_CODE.items()
}


@lru_cache(maxsize=1)
def _biased_gemm_kernel():
    # Grouped GEMM/indexing adapted from PyTorch Inductor; see THIRD_PARTY_NOTICES.md.
    from triton import jit
    import triton.language as tl

    @jit
    def kernel(
        X, W, B, Y, MASK, INPUT_MASK,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        XM: tl.constexpr, XK: tl.constexpr,
        WK: tl.constexpr, WN: tl.constexpr, BS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
        HAS_BIAS: tl.constexpr, HAS_MASK: tl.constexpr, HAS_INPUT_MASK: tl.constexpr,
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
        accumulator = tl.full((BM, BN), 0, tl.float32)
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
        for start in range(tl.cdiv(K, BK)):
            indices = start * BK + inner
            if M < BM or N < BN:
                left = tl.load(
                    X + load_rows[:, None] * XM + indices[None, :] * XK,
                    (load_rows[:, None] < M) & (indices[None, :] < K), 0,
                ).to(tl.bfloat16)
                right = tl.load(
                    W + load_columns[None, :] * WN + indices[:, None] * WK,
                    (load_columns[None, :] < N) & (indices[:, None] < K), 0,
                    eviction_policy="evict_last",
                ).to(tl.bfloat16)
            elif K % BK == 0:
                left = tl.load(
                    X + load_rows[:, None] * XM + indices[None, :] * XK,
                ).to(tl.bfloat16)
                right = tl.load(
                    W + load_columns[None, :] * WN + indices[:, None] * WK,
                    eviction_policy="evict_last",
                ).to(tl.bfloat16)
            else:
                left = tl.load(
                    X + load_rows[:, None] * XM + indices[None, :] * XK,
                    indices[None, :] < K, 0,
                ).to(tl.bfloat16)
                right = tl.load(
                    W + load_columns[None, :] * WN + indices[:, None] * WK,
                    indices[:, None] < K, 0,
                    eviction_policy="evict_last",
                ).to(tl.bfloat16)
            accumulator = tl.dot(left, right, accumulator)
        # Recompute store indices to shorten their live ranges across the loop.
        rows = tile_m * BM + tl.arange(0, BM)
        columns = tile_n * BN + tl.arange(0, BN)
        if HAS_INPUT_MASK:
            valid_in = tl.load(INPUT_MASK + rows, rows < M, 0)
            accumulator = tl.where(valid_in[:, None], accumulator, 0)
        if HAS_BIAS:
            bias = tl.load(B + columns * BS, columns < N, 0, eviction_policy="evict_last").to(tl.float32)
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


@lru_cache(maxsize=1)
def _biased_gemm_hopper_kernel():
    from triton import jit
    import triton.language as tl
    import triton.language.extra.cuda as tlc

    @jit
    def kernel(
        X, W, B, Y, MASK, INPUT_MASK,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        XM: tl.constexpr, XK: tl.constexpr,
        WK: tl.constexpr, WN: tl.constexpr, BS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
        HAS_BIAS: tl.constexpr, HAS_MASK: tl.constexpr, HAS_INPUT_MASK: tl.constexpr,
        USE_TMA: tl.constexpr, W_TRANSPOSED: tl.constexpr, USE_PDL: tl.constexpr,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        pid = tl.program_id(0)
        grid_m, grid_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
        group_width = GROUP_M * grid_n
        group = pid // group_width
        group_size = tl.minimum(grid_m - group * GROUP_M, GROUP_M)
        tile_m = group * GROUP_M + pid % group_size
        tile_n = pid % group_width // group_size
        tl.assume(tile_m >= 0)
        tl.assume(tile_n >= 0)
        accumulator = tl.full((BM, BN), 0, tl.float32)
        zero = pid * 0
        if USE_TMA:
            m_rt = zero + M
            n_rt = zero + N
            k_rt = zero + K
            xm_rt = zero + XM
            ws_rt = zero + (WN if W_TRANSPOSED else WK)
            desc_x = tl.make_tensor_descriptor(
                X, shape=[m_rt, k_rt], strides=[xm_rt, 1], block_shape=[BM, BK],
            )
            if W_TRANSPOSED:
                desc_w = tl.make_tensor_descriptor(
                    W, shape=[n_rt, k_rt], strides=[ws_rt, 1], block_shape=[BN, BK],
                )
            else:
                desc_w = tl.make_tensor_descriptor(
                    W, shape=[k_rt, n_rt], strides=[ws_rt, 1], block_shape=[BK, BN],
                )
            for start in range(tl.cdiv(K, BK)):
                k_off = start * BK
                left = desc_x.load([tile_m * BM, k_off])
                if W_TRANSPOSED:
                    right = tl.trans(desc_w.load([tile_n * BN, k_off]))
                else:
                    right = desc_w.load([k_off, tile_n * BN])
                accumulator = tl.dot(left, right, accumulator)
        else:
            rows = tile_m * BM + tl.arange(0, BM)
            columns = tile_n * BN + tl.arange(0, BN)
            inner = tl.arange(0, BK)
            for start in range(tl.cdiv(K, BK)):
                indices = start * BK + inner
                left = tl.load(
                    X + rows[:, None] * XM + indices[None, :] * XK,
                    (rows[:, None] < M) & (indices[None, :] < K), 0,
                ).to(tl.bfloat16)
                right = tl.load(
                    W + columns[None, :] * WN + indices[:, None] * WK,
                    (columns[None, :] < N) & (indices[:, None] < K), 0,
                    eviction_policy="evict_last",
                ).to(tl.bfloat16)
                accumulator = tl.dot(left, right, accumulator)
        rows = tile_m * BM + tl.arange(0, BM)
        columns = tile_n * BN + tl.arange(0, BN)
        if HAS_INPUT_MASK:
            valid_in = tl.load(INPUT_MASK + rows, rows < M, 0)
            accumulator = tl.where(valid_in[:, None], accumulator, 0)
        if HAS_BIAS:
            bias = tl.load(B + columns * BS, columns < N, 0, eviction_policy="evict_last").to(tl.float32)
            accumulator += bias[None, :]
        if HAS_MASK:
            valid = tl.load(MASK + rows, rows < M, 0)
            accumulator = tl.where(valid[:, None], accumulator, 0)
        if USE_PDL:
            tlc.gdc_launch_dependents()
        if USE_TMA:
            desc_y = tl.make_tensor_descriptor(
                Y, shape=[zero + M, zero + N], strides=[zero + N, 1], block_shape=[BM, BN],
            )
            desc_y.store([tile_m * BM, tile_n * BN], accumulator.to(Y.dtype.element_ty))
        else:
            tl.store(
                Y + rows[:, None] * N + columns[None, :],
                accumulator,
                (rows[:, None] < M) & (columns[None, :] < N),
            )
    return kernel


def _biased_gemm(
    left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor | None,
    output_dtype: torch.dtype, output_mask: torch.Tensor | None = None,
    input_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # AB + bias: FP32 accumulation/addition, then one output rounding.
    rows, inner = left.shape
    columns = right.shape[1]
    output = torch.empty((rows, columns), device=left.device, dtype=output_dtype)
    if rows == 0 or columns == 0:
        return output
    if output_mask is not None and not output_mask.is_contiguous():
        output_mask = output_mask.contiguous()
    if input_mask is not None:
        if not input_mask.is_contiguous():
            input_mask = input_mask.contiguous()
        if (
            output_mask is not None
            and (
                input_mask is output_mask
                or (
                    not torch.compiler.is_compiling()
                    and type(input_mask) is torch.Tensor
                    and type(output_mask) is torch.Tensor
                    and input_mask.data_ptr() == output_mask.data_ptr()
                    and input_mask.numel() == output_mask.numel()
                )
            )
        ):
            input_mask = None
    is_hopper = _is_hopper_or_newer(left.device)
    w_transposed = right.stride(0) == 1 and right.stride(1) >= inner
    w_row_major = right.stride(1) == 1 and right.stride(0) >= columns
    w_stride = right.stride(1) if w_transposed else right.stride(0)
    use_tma = (
        is_hopper
        and left.dtype == torch.bfloat16
        and right.dtype == torch.bfloat16
        and output_dtype in (torch.bfloat16, torch.float16, torch.float32)
        and left.stride(1) == 1
        and left.stride(0) >= inner
        and (w_transposed or w_row_major)
        and _tma_aligned_2d(left, left.stride(0))
        and _tma_aligned_2d(right, w_stride)
        and _tma_aligned_2d(output, columns)
    )
    if use_tma:
        _ensure_triton_allocator()
    if rows >= 128 and columns >= 128 and columns % 8 == 0 and inner % 8 == 0:
        bm, bn, group = 128, 128, 4
        warps = 8 if is_hopper else 4
        if left.element_size() > 2 or right.element_size() > 2:
            bk, stages = 32, 3
        elif is_hopper:
            bk, stages = 64, 3
        else:
            bk, stages = (64, 3) if inner >= 512 else (32, 4)
    else:
        bm, bn, bk, warps, stages, group = 64, 64, 32, 4, 3, 1
    grid = ((rows + bm - 1) // bm) * ((columns + bn - 1) // bn)
    if is_hopper:
        _biased_gemm_hopper_kernel()[(grid,)](
            left, right, bias, output, output_mask, input_mask,
            rows, columns, inner,
            *left.stride(), *right.stride(), 0 if bias is None else bias.stride(0),
            bm, bn, bk, group,
            bias is not None, output_mask is not None, input_mask is not None,
            use_tma, w_transposed, True,
            num_warps=warps, num_stages=stages, launch_pdl=True,
        )
    else:
        _biased_gemm_kernel()[(grid,)](
            left, right, bias, output, output_mask, input_mask,
            rows, columns, inner,
            *left.stride(), *right.stride(), 0 if bias is None else bias.stride(0),
            bm, bn, bk, group,
            bias is not None, output_mask is not None, input_mask is not None,
            num_warps=warps, num_stages=stages,
        )
    return output


def _wide_gemm(
    left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor | None = None,
    *, output_dtype: torch.dtype = torch.float32,
    output_mask: torch.Tensor | None = None,
    input_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16 operands with FP32 accumulation, independent of ambient AMP policy."""
    with torch.autocast(device_type="cuda", enabled=False):
        if output_mask is not None or input_mask is not None:
            return _biased_gemm(
                left, right, bias, output_dtype, output_mask, input_mask,
            )
        if bias is not None and output_dtype in (torch.float16, torch.bfloat16):
            return _biased_gemm(left, right, bias, output_dtype)
        if left.dtype != torch.bfloat16:
            left = left.to(dtype=torch.bfloat16)
        if bias is not None:
            return torch.ops.aten.addmm.dtype(bias, left, right, torch.float32)
        operation = torch.bmm if left.ndim == 3 else torch.mm
        if output_dtype == torch.bfloat16:
            return operation(left, right)
        return operation(left, right, out_dtype=torch.float32)


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


class _TensorCoreBmmFunction(torch.autograd.Function):
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


@torch.library.custom_op("memsolve::tensor_core_bmm", mutates_args=())
def _tensor_core_bmm_op(
    left_batches: torch.Tensor,
    right_batches: torch.Tensor,
) -> torch.Tensor:
    return _wide_gemm(left_batches, right_batches).contiguous()


@_tensor_core_bmm_op.register_fake
def _(
    left_batches: torch.Tensor,
    right_batches: torch.Tensor,
) -> torch.Tensor:
    return left_batches.new_empty(
        (left_batches.shape[0], left_batches.shape[1], right_batches.shape[2]),
        dtype=torch.float32,
    )


@torch.library.custom_op("memsolve::tensor_core_bmm_backward", mutates_args=())
def _tensor_core_bmm_bwd_op(
    grad_output: torch.Tensor,
    left_batches: torch.Tensor | None,
    right_batches: torch.Tensor | None,
    need_dl: bool,
    need_dr: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_batches = grad_output.to(dtype=torch.bfloat16)
    if need_dl:
        assert right_batches is not None
        grad_left = _wide_gemm(grad_batches, right_batches.mT).contiguous()
    else:
        grad_left = grad_output.new_empty((0,), dtype=torch.float32)
    if need_dr:
        assert left_batches is not None
        grad_right = _wide_gemm(left_batches.mT, grad_batches).contiguous()
    else:
        grad_right = grad_output.new_empty((0,), dtype=torch.float32)
    return grad_left, grad_right


@_tensor_core_bmm_bwd_op.register_fake
def _(
    grad_output: torch.Tensor,
    left_batches: torch.Tensor | None,
    right_batches: torch.Tensor | None,
    need_dl: bool,
    need_dr: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if need_dl:
        assert right_batches is not None
        grad_left = grad_output.new_empty(
            (grad_output.shape[0], grad_output.shape[1], right_batches.shape[1]),
            dtype=torch.float32,
        )
    else:
        grad_left = grad_output.new_empty((0,), dtype=torch.float32)
    if need_dr:
        assert left_batches is not None
        grad_right = grad_output.new_empty(
            (grad_output.shape[0], left_batches.shape[2], grad_output.shape[2]),
            dtype=torch.float32,
        )
    else:
        grad_right = grad_output.new_empty((0,), dtype=torch.float32)
    return grad_left, grad_right


def _tensor_core_bmm_setup_ctx(ctx, inputs, output):
    left_batches, right_batches = inputs
    del output
    ctx.save_for_backward(
        left_batches if right_batches.requires_grad else None,
        right_batches if left_batches.requires_grad else None,
    )


def _tensor_core_bmm_bwd(ctx, grad_output):
    left_batches, right_batches = ctx.saved_tensors
    need_dl = ctx.needs_input_grad[0]
    need_dr = ctx.needs_input_grad[1]
    dl, dr = _tensor_core_bmm_bwd_op(
        grad_output, left_batches, right_batches, need_dl, need_dr,
    )
    return (
        dl.to(dtype=torch.bfloat16) if need_dl else None,
        dr.to(dtype=torch.bfloat16) if need_dr else None,
    )


_tensor_core_bmm_op.register_autograd(
    _tensor_core_bmm_bwd, setup_context=_tensor_core_bmm_setup_ctx
)


class _TensorCoreBmm:
    """BF16 Tensor Core BMM dispatching between custom_op (compile) and Function (eager)."""

    @staticmethod
    def apply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if not torch.compiler.is_compiling() and type(left) is torch.Tensor and type(right) is torch.Tensor:
            return _TensorCoreBmmFunction.apply(left, right)
        batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        rows, inner, columns = left.shape[-2], left.shape[-1], right.shape[-1]
        left_b = left.to(dtype=torch.bfloat16).expand(batch_shape + (rows, inner)).reshape(-1, rows, inner)
        right_b = right.to(dtype=torch.bfloat16).expand(batch_shape + (inner, columns)).reshape(-1, inner, columns)
        return _tensor_core_bmm_op(left_b, right_b).reshape(batch_shape + (rows, columns))


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


def _record_linear_ctx(
    ctx: torch.autograd.function.FunctionCtx,
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    fuse_weight_bias_grad: bool,
) -> None:
    ctx.value_shape = value.shape
    ctx.value_dtype = value.dtype
    ctx.weight_dtype = weight.dtype
    ctx.out_features, ctx.in_features = weight.shape
    ctx.bias_dtype = None if bias is None else bias.dtype
    ctx.bias_device = None if bias is None else bias.device
    ctx.has_bias = bias is not None
    ctx.fuse_weight_bias_grad = fuse_weight_bias_grad


def _linear_forward_output(
    value: torch.Tensor,
    weight_bf16: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype | None,
    output_mask: torch.Tensor | None,
    input_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    flat_value = value.reshape(-1, value.shape[-1])
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
        input_mask=input_mask,
    ).reshape(value.shape[:-1] + (weight_bf16.shape[0],))
    del flat_value
    return output if output_dtype is None else output.to(dtype=output_dtype)


def _linear_backward_common(
    ctx,
    grad_output: torch.Tensor,
    value: torch.Tensor | None,
    weight_bf16: torch.Tensor | None,
    output_mask: torch.Tensor | None,
    input_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    need_dx = ctx.needs_input_grad[0]
    need_dw = ctx.needs_input_grad[1]
    need_db = ctx.has_bias and ctx.needs_input_grad[2]

    grad_weight = grad_bias = None
    if need_dw or need_db:
        grad_output = _masked_cast(grad_output, output_mask, grad_output.dtype)
        flat_gradient = (
            grad_output.reshape(-1, ctx.out_features)
            if grad_output.dtype == torch.bfloat16
            else grad_output.to(dtype=torch.bfloat16).reshape(-1, ctx.out_features)
        )
        # Bias uses the original upstream gradient, not a BF16-rounded FP16/
        # FP32 tensor. Fuse only when both VJPs consume the same BF16 values.
        fused = (
            ctx.fuse_weight_bias_grad and grad_output.dtype == torch.bfloat16
            and need_dw and need_db
        )
        if fused:
            require_available()
            flat_value = _masked_cast(value, input_mask, torch.bfloat16).reshape(
                -1, ctx.in_features
            )
            del value
            grad_weight, grad_bias = torch.ops.memsolve_equilibrium.linear_weight_bias(
                flat_value.contiguous(), flat_gradient.contiguous(),
            )
            del flat_value
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype)
            grad_bias = grad_bias.to(device=ctx.bias_device, dtype=ctx.bias_dtype)
        else:
            if need_dw:
                flat_value = _masked_cast(value, input_mask, torch.bfloat16).reshape(
                    -1, ctx.in_features
                )
                del value
                grad_weight = _wide_gemm(
                    flat_gradient.mT,
                    flat_value,
                ).to(dtype=ctx.weight_dtype)
                del flat_value
            else:
                del value
            if need_db:
                # Preserve the original FP32 bias-reduction boundary without
                # materializing a full FP32 copy of a half-precision upstream VJP.
                dimensions = tuple(range(grad_output.ndim - 1))
                grad_bias = (
                    grad_output.sum(dim=dimensions, dtype=torch.float32)
                    if dimensions
                    else grad_output.to(dtype=torch.float32)
                ).to(device=ctx.bias_device, dtype=ctx.bias_dtype)
        del grad_output
        grad_input_mask = None
    else:
        del value
        flat_gradient = grad_output.reshape(-1, ctx.out_features)
        del grad_output
        grad_input_mask = output_mask

    grad_value = None
    if need_dx:
        grad_value = _wide_gemm(
            flat_gradient,
            weight_bf16,
            output_dtype=(
                torch.bfloat16 if ctx.value_dtype == torch.bfloat16 else torch.float32
            ),
            output_mask=input_mask,
            input_mask=grad_input_mask,
        ).reshape(ctx.value_shape)
        del weight_bf16
        grad_value = grad_value.to(dtype=ctx.value_dtype)

    return grad_value, grad_weight, grad_bias


class _LinearCtxProxy:
    __slots__ = (
        "value_shape",
        "value_dtype",
        "weight_dtype",
        "out_features",
        "in_features",
        "bias_dtype",
        "bias_device",
        "has_bias",
        "fuse_weight_bias_grad",
        "needs_input_grad",
    )


class _TensorCoreLinearFunction(torch.autograd.Function):
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
        weight_bf16 = weight.to(dtype=torch.bfloat16)
        ctx.save_for_backward(
            value if weight.requires_grad else None,
            weight_bf16 if value.requires_grad else None,
            output_mask,
            input_mask,
        )
        _record_linear_ctx(ctx, value, weight, bias, fuse_weight_bias_grad)
        return _linear_forward_output(
            value, weight_bf16, bias, output_dtype, output_mask, input_mask,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        value, weight_bf16, output_mask, input_mask = ctx.saved_tensors
        # Match AOTAutograd: release ownership only when retain_graph is false.
        ctx.maybe_clear_saved_tensors()
        grad_value, grad_weight, grad_bias = _linear_backward_common(
            ctx, grad_output, value, weight_bf16, output_mask, input_mask,
        )
        return grad_value, grad_weight, grad_bias, None, None, None, None


def _ensure_contig(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


@torch.library.custom_op("memsolve::tensor_core_linear", mutates_args=())
def _tensor_core_linear_op(
    flat_value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output_mask: torch.Tensor | None,
    input_mask: torch.Tensor | None,
    out_dtype_code: int,
    fuse_weight_bias_grad: bool,
    save_weight_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del fuse_weight_bias_grad
    output_dtype = _CODE_TO_DTYPE[out_dtype_code]
    weight_bf16 = (
        weight
        if weight.dtype == torch.bfloat16
        else _ensure_contig(weight.detach().to(dtype=torch.bfloat16))
    )
    out = _ensure_contig(
        _linear_forward_output(
            flat_value, weight_bf16, bias, output_dtype, output_mask, input_mask,
        )
    )
    saved_weight = (
        weight_bf16
        if save_weight_bf16
        else weight.new_empty((0, 0), dtype=torch.bfloat16)
    )
    return out, saved_weight


@_tensor_core_linear_op.register_fake
def _(
    flat_value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output_mask: torch.Tensor | None,
    input_mask: torch.Tensor | None,
    out_dtype_code: int,
    fuse_weight_bias_grad: bool,
    save_weight_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del bias, output_mask, input_mask, fuse_weight_bias_grad
    output_dtype = _CODE_TO_DTYPE[out_dtype_code]
    res_dtype = torch.float32 if output_dtype is None else output_dtype
    out = flat_value.new_empty((flat_value.shape[0], weight.shape[0]), dtype=res_dtype)
    saved_weight = weight.new_empty(
        (weight.shape[0], weight.shape[1]) if save_weight_bf16 else (0, 0),
        dtype=torch.bfloat16,
    )
    return out, saved_weight


@torch.library.custom_op("memsolve::tensor_core_linear_backward", mutates_args=())
def _tensor_core_linear_bwd_op(
    grad_output: torch.Tensor,
    flat_value: torch.Tensor | None,
    weight_bf16: torch.Tensor | None,
    output_mask: torch.Tensor | None,
    input_mask: torch.Tensor | None,
    in_features: int,
    value_dtype_code: int,
    weight_dtype_code: int,
    bias_dtype_code: int,
    fuse_weight_bias_grad: bool,
    need_dx: bool,
    need_dw: bool,
    need_db: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proxy = _LinearCtxProxy()
    proxy.value_shape = (grad_output.shape[0], in_features)
    proxy.value_dtype = _CODE_TO_DTYPE[value_dtype_code]
    proxy.weight_dtype = _CODE_TO_DTYPE[weight_dtype_code]
    proxy.out_features = grad_output.shape[1]
    proxy.in_features = in_features
    proxy.bias_dtype = torch.float32 if bias_dtype_code == 0 else _CODE_TO_DTYPE[bias_dtype_code]
    proxy.bias_device = grad_output.device
    proxy.has_bias = bias_dtype_code != 0
    proxy.fuse_weight_bias_grad = fuse_weight_bias_grad
    proxy.needs_input_grad = (need_dx, need_dw, need_db)

    dx, dw, db = _linear_backward_common(
        proxy,
        grad_output,
        flat_value,
        weight_bf16 if need_dx else None,
        output_mask,
        input_mask,
    )
    dx_out = _ensure_contig(dx) if dx is not None else grad_output.new_empty((0,), dtype=proxy.value_dtype)
    dw_out = _ensure_contig(dw) if dw is not None else grad_output.new_empty((0,), dtype=proxy.weight_dtype)
    db_out = _ensure_contig(db) if db is not None else grad_output.new_empty((0,), dtype=proxy.bias_dtype)
    return dx_out, dw_out, db_out


@_tensor_core_linear_bwd_op.register_fake
def _(
    grad_output: torch.Tensor,
    flat_value: torch.Tensor | None,
    weight_bf16: torch.Tensor | None,
    output_mask: torch.Tensor | None,
    input_mask: torch.Tensor | None,
    in_features: int,
    value_dtype_code: int,
    weight_dtype_code: int,
    bias_dtype_code: int,
    fuse_weight_bias_grad: bool,
    need_dx: bool,
    need_dw: bool,
    need_db: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del flat_value, weight_bf16, output_mask, input_mask, fuse_weight_bias_grad
    val_dtype = _CODE_TO_DTYPE[value_dtype_code]
    w_dtype = _CODE_TO_DTYPE[weight_dtype_code]
    b_dtype = torch.float32 if bias_dtype_code == 0 else _CODE_TO_DTYPE[bias_dtype_code]
    dx = (
        grad_output.new_empty((grad_output.shape[0], in_features), dtype=val_dtype)
        if need_dx
        else grad_output.new_empty((0,), dtype=val_dtype)
    )
    dw = (
        grad_output.new_empty((grad_output.shape[1], in_features), dtype=w_dtype)
        if need_dw
        else grad_output.new_empty((0,), dtype=w_dtype)
    )
    db = (
        grad_output.new_empty((grad_output.shape[1],), dtype=b_dtype)
        if need_db
        else grad_output.new_empty((0,), dtype=b_dtype)
    )
    return dx, dw, db


def _tensor_core_linear_setup_ctx(ctx, inputs, output):
    (
        flat_value,
        weight,
        bias,
        output_mask,
        input_mask,
        _out_dtype_code,
        fuse_weight_bias_grad,
        save_weight_bf16,
    ) = inputs
    _out, saved_weight = output
    saved_w_for_bwd = (
        (weight if weight.dtype == torch.bfloat16 else saved_weight)
        if (flat_value.requires_grad and (save_weight_bf16 or weight.dtype == torch.bfloat16))
        else None
    )
    ctx.save_for_backward(
        flat_value if weight.requires_grad else None,
        saved_w_for_bwd,
        output_mask,
        input_mask,
    )
    ctx.in_features = weight.shape[1]
    ctx.value_dtype_code = _DTYPE_TO_CODE[flat_value.dtype]
    ctx.weight_dtype_code = _DTYPE_TO_CODE[weight.dtype]
    ctx.bias_dtype_code = 0 if bias is None else _DTYPE_TO_CODE[bias.dtype]
    ctx.bias_device = None if bias is None else bias.device
    ctx.fuse_weight_bias_grad = fuse_weight_bias_grad


def _tensor_core_linear_bwd(ctx, grad_output, _grad_saved_weight):
    flat_value, weight_bf16, output_mask, input_mask = ctx.saved_tensors
    need_dx = ctx.needs_input_grad[0]
    need_dw = ctx.needs_input_grad[1]
    need_db = (ctx.bias_dtype_code != 0) and ctx.needs_input_grad[2]
    dx, dw, db = _tensor_core_linear_bwd_op(
        grad_output,
        flat_value,
        weight_bf16,
        output_mask,
        input_mask,
        ctx.in_features,
        ctx.value_dtype_code,
        ctx.weight_dtype_code,
        ctx.bias_dtype_code,
        ctx.fuse_weight_bias_grad,
        need_dx,
        need_dw,
        need_db,
    )
    if need_db and ctx.bias_device is not None and db.device != ctx.bias_device:
        db = db.to(device=ctx.bias_device)
    return (
        dx if need_dx else None,
        dw if need_dw else None,
        db if need_db else None,
        None,
        None,
        None,
        None,
        None,
    )


_tensor_core_linear_op.register_autograd(
    _tensor_core_linear_bwd, setup_context=_tensor_core_linear_setup_ctx
)


class _TensorCoreLinear:
    """BF16 Tensor Core linear map dispatching between custom_op (compile) and Function (eager)."""

    @staticmethod
    def apply(
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype | None,
        fuse_weight_bias_grad: bool,
        output_mask: torch.Tensor | None,
        input_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling() and type(value) is torch.Tensor:
            return _TensorCoreLinearFunction.apply(
                value, weight, bias, output_dtype, fuse_weight_bias_grad, output_mask, input_mask,
            )
        flat_value = value.reshape(-1, value.shape[-1])
        flat_out_mask = None if output_mask is None else output_mask.reshape(-1)
        flat_in_mask = None if input_mask is None else input_mask.reshape(-1)
        save_weight_bf16 = bool(
            torch.is_grad_enabled()
            and value.requires_grad
            and weight.dtype != torch.bfloat16
        )
        out, _ = _tensor_core_linear_op(
            flat_value,
            weight,
            bias,
            flat_out_mask,
            flat_in_mask,
            _DTYPE_TO_CODE[output_dtype],
            fuse_weight_bias_grad,
            save_weight_bf16,
        )
        return out.reshape(value.shape[:-1] + (weight.shape[0],))


class _RecomputeOutputLinear(torch.autograd.Function):
    """Output projection that reconstructs `mixed` via `_readout` during backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        mixed: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype | None,
        fuse_weight_bias_grad: bool,
        output_mask: torch.Tensor | None,
        spec: _RecomputeSpec,
    ) -> torch.Tensor:
        if output_mask is not None and not output_mask.is_contiguous():
            output_mask = output_mask.contiguous()
        weight_bf16 = weight.to(dtype=torch.bfloat16)
        (
            projected,
            coefficient,
            eta,
            counts,
            rotary_cos,
            rotary_sin,
            heads,
            rank,
            head_dim,
            mode,
        ) = spec
        if weight.requires_grad:
            ctx.save_for_backward(
                weight_bf16 if mixed.requires_grad else None,
                output_mask,
                projected,
                coefficient,
                eta,
                counts,
                rotary_cos,
                rotary_sin,
            )
        else:
            ctx.save_for_backward(
                weight_bf16 if mixed.requires_grad else None,
                output_mask,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        ctx.heads = heads
        ctx.rank = rank
        ctx.head_dim = head_dim
        ctx.mode = mode
        _record_linear_ctx(ctx, mixed, weight, bias, fuse_weight_bias_grad)
        return _linear_forward_output(
            mixed, weight_bf16, bias, output_dtype, output_mask, None,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        (
            weight_bf16,
            output_mask,
            projected,
            coefficient,
            eta,
            counts,
            rotary_cos,
            rotary_sin,
        ) = ctx.saved_tensors
        ctx.maybe_clear_saved_tensors()
        mixed = None
        if ctx.needs_input_grad[1]:
            with torch.no_grad():
                mixed = _run_readout(
                    projected,
                    coefficient,
                    eta,
                    counts,
                    rotary_cos,
                    rotary_sin,
                    heads=ctx.heads,
                    rank=ctx.rank,
                    head_dim=ctx.head_dim,
                    mode=ctx.mode,
                )
        del projected, coefficient, eta, counts, rotary_cos, rotary_sin
        grad_mixed, grad_weight, grad_bias = _linear_backward_common(
            ctx, grad_output, mixed, weight_bf16, output_mask, None,
        )
        del mixed
        return grad_mixed, grad_weight, grad_bias, None, None, None, None


def _recompute_output_linear(
    mixed: torch.Tensor,
    spec: _RecomputeSpec,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype | None = None,
    fuse_weight_bias_grad: bool = False,
    output_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _RecomputeOutputLinear.apply(
        mixed, weight, bias, output_dtype, fuse_weight_bias_grad, output_mask, spec,
    )


@lru_cache(maxsize=1)
def _no_frame_kernels():
    # Lazily import the Triton supplied by CUDA PyTorch, as for projections.
    from triton import jit
    import triton.language as tl
    import triton.language.extra.cuda as tlc

    @jit
    def _short_rsqrt(count):
        y0 = tl.rsqrt(count)
        return tl.math.fma(0.5 * y0, tl.math.fma(-count * y0, y0, 1.0), y0)

    @jit
    def _mixed_dot(a, b):
        # Two BF16 terms retain the compact FP32 coefficient's low bits
        # (Henry et al., arXiv:1904.06376); the token operand is already BF16.
        # Accumulate the O(2^-8) low term first into a single FP32 tile so
        # high-order bits do not swamp low-order bits before FMA addition.
        hi = b.to(tl.bfloat16)
        lo = (b - hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(a, lo)
        return tl.dot(a, hi, acc)

    @jit
    def _mixed_dot_acc(a, b, acc):
        hi = b.to(tl.bfloat16)
        lo = (b - hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(a, lo, acc)
        return tl.dot(a, hi, acc)

    @jit
    def _mixed_dot_lhs(a, b):
        # Transposed counterpart of _mixed_dot where `a` is FP32 and `b` is BF16,
        # used on Hopper SM90 so M=BD>=64 maps directly to native WGMMA shapes.
        hi = a.to(tl.bfloat16)
        lo = (a - hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(lo, b)
        return tl.dot(hi, b, acc)

    @jit
    def _wide_dot(a, b):
        # Three BF16 products retain both rotated FP32 operands' low bits.
        # The omitted low-low product is second order (Henry et al. 2019).
        # Sum the two small cross-terms first, then FMA-add the main ah*bh term.
        ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        result = tl.dot(al, bh)
        result = tl.dot(ah, bl, result)
        return tl.dot(ah, bh, result)

    @jit
    def _wide_dot_acc(a, b, acc):
        ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(al, bh, acc)
        acc = tl.dot(ah, bl, acc)
        return tl.dot(ah, bh, acc)

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
        cosine = tl.load(COS + offset, mask, 1.0, eviction_policy="evict_last")
        sine = tl.load(SIN + offset, mask, 0.0, eviction_policy="evict_last")
        if INVERSE:
            sine = -sine
        even, odd = tl.split(tl.reshape(value.to(tl.float32),
                                       (value.shape[0], value.shape[1] // 2, 2)))
        real = tl.math.fma(even, cosine, -(odd * sine))
        imag = tl.math.fma(even, sine, odd * cosine)
        return tl.reshape(tl.join(real, imag), value.shape)

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
        N_rt,
        R_rt,
        D_rt,
        W_rt,
        HD_rt,
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
        SCALE_OUTPUT: tl.constexpr = True,
        USE_TMA: tl.constexpr = False,
        TRANSPOSE_Q: tl.constexpr = False,
        USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        tile = tl.program_id(1)
        b = s // H
        h = s % H
        if SCALE_OUTPUT:
            count = tl.load(COUNT + b, eviction_policy="evict_last") if HAS_COUNT else float(N)
            inv_count = tl.math.div_rn(1.0, count)
            inv_sqrt = tl.math.div_rn(1.0, tl.math.sqrt_rn(count))
        n = tile * BT + tl.arange(0, BT)
        r = tl.arange(0, BR)
        if USE_TMA:
            x_base = X + b.to(tl.int64) * N_rt * W_rt
            desc_a = tl.make_tensor_descriptor(
                x_base + h * R, shape=[N_rt, R_rt], strides=[W_rt, 1], block_shape=[BT, BR],
            )
            a = desc_a.load([tile * BT, 0])
            desc_c = tl.make_tensor_descriptor(
                x_base + H * R + h * D, shape=[N_rt, D_rt], strides=[W_rt, 1], block_shape=[BT, BD],
            )
            if GRAD:
                e_base = E + b.to(tl.int64) * N_rt * HD_rt
                desc_e = tl.make_tensor_descriptor(
                    e_base + h * D, shape=[N_rt, D_rt], strides=[HD_rt, 1], block_shape=[BT, BD],
                )
        else:
            a = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                (n[:, None] < N) & (r[None, :] < R),
                0,
            )
        if not GRAD:
            g = tl.dot(tl.trans(a), a).to(tl.float32)
            if SCALE_OUTPUT:
                g = g * inv_count
            tl.store(
                G + ((s * T + tile) * R + r[:, None]) * R + r[None, :],
                g, (r[:, None] < R) & (r[None, :] < R),
            )
        de = tl.full((), 0.0, tl.float32)
        num_c_tiles: tl.constexpr = tl.cdiv(D, BD)
        for channel_tile in range(num_c_tiles):
            d = channel_tile * BD + tl.arange(0, BD)
            if USE_TMA:
                c = desc_c.load([tile * BT, channel_tile * BD])
            else:
                c = tl.load(
                    X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D), 0,
                )
            if GRAD:
                if USE_TMA:
                    e = desc_e.load([tile * BT, channel_tile * BD])
                else:
                    e = tl.load(
                        E + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                        (n[:, None] < N) & (d[None, :] < D), 0,
                    )
                if HAS_ROTARY:
                    rotated = _rotate(e, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED)
                    if TRANSPOSE_Q:
                        q = tl.trans(_mixed_dot_lhs(tl.trans(rotated), a))
                    else:
                        q = _mixed_dot(tl.trans(a), rotated)
                else:
                    if TRANSPOSE_Q:
                        q = tl.trans(tl.dot(tl.trans(e), a).to(tl.float32))
                    else:
                        q = tl.dot(tl.trans(a), e).to(tl.float32)
                de += tl.sum(tl.sum(e.to(tl.float32) * c.to(tl.float32), 0), 0)
            else:
                if HAS_ROTARY:
                    rotated = _rotate(c, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED)
                    if TRANSPOSE_Q:
                        q = tl.trans(_mixed_dot_lhs(tl.trans(rotated), a))
                    else:
                        q = _mixed_dot(tl.trans(a), rotated)
                else:
                    if TRANSPOSE_Q:
                        q = tl.trans(tl.dot(tl.trans(c), a).to(tl.float32))
                    else:
                        q = tl.dot(tl.trans(a), c).to(tl.float32)
            if SCALE_OUTPUT:
                q = q * inv_sqrt
            if USE_PDL and not GRAD and channel_tile + 1 == num_c_tiles:
                tlc.gdc_launch_dependents()
            tl.store(
                Q + ((s * T + tile) * R + r[:, None]) * D + d[None, :],
                q, (r[:, None] < R) & (d[None, :] < D),
            )
        if GRAD:
            if USE_PDL:
                tlc.gdc_launch_dependents()
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
        N_rt,
        R_rt,
        D_rt,
        W_rt,
        HD_rt,
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
        USE_TMA: tl.constexpr = False,
        USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        tile = tl.program_id(1)
        b = s // H
        h = s % H
        n = tile * BT + tl.arange(0, BT)
        r = tl.arange(0, BR)
        if USE_TMA:
            x_base = X + b.to(tl.int64) * N_rt * W_rt
            y_base = Y + b.to(tl.int64) * N_rt * HD_rt
            desc_a = tl.make_tensor_descriptor(
                x_base + h * R, shape=[N_rt, R_rt], strides=[W_rt, 1], block_shape=[BT, BR],
            )
            a = desc_a.load([tile * BT, 0])
            desc_c = tl.make_tensor_descriptor(
                x_base + H * R + h * D, shape=[N_rt, D_rt], strides=[W_rt, 1], block_shape=[BT, BD],
            )
            desc_y = tl.make_tensor_descriptor(
                y_base + h * D, shape=[N_rt, D_rt], strides=[HD_rt, 1], block_shape=[BT, BD],
            )
        else:
            a = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                (n[:, None] < N) & (r[None, :] < R),
                0,
            )
        num_c_tiles: tl.constexpr = tl.cdiv(D, BD)
        for channel_tile in range(num_c_tiles):
            d = channel_tile * BD + tl.arange(0, BD)
            v = tl.load(
                V + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            correction = _mixed_dot(a, v)
            if UNSCALED_COMPLEMENT:
                eta_c = tl.load(ETA + h, eviction_policy="evict_last")
                correction = -eta_c * correction
            if HAS_ROTARY:
                correction = _rotate(correction, COS, SIN, b, n, channel_tile * BD,
                                     N, D, ROTARY_BATCHED, True)
            if USE_TMA:
                c = desc_c.load([tile * BT, channel_tile * BD]).to(tl.float32)
            else:
                c = tl.load(
                    X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0,
                ).to(tl.float32)
            inv_sqrt = (
                _short_rsqrt(tl.load(COUNT + b, eviction_policy="evict_last"))
                if HAS_COUNT
                else tl.math.div_rn(1.0, tl.math.sqrt_rn(float(N)))
            )
            eta = tl.load(ETA + h, eviction_policy="evict_last")
            y = tl.math.fma(correction, inv_sqrt, eta * c)
            if USE_PDL and channel_tile + 1 == num_c_tiles:
                tlc.gdc_launch_dependents()
            if USE_TMA:
                desc_y.store([tile * BT, channel_tile * BD], y.to(Y.dtype.element_ty))
            else:
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
        N_rt,
        R_rt,
        D_rt,
        W_rt,
        HD_rt,
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
        USE_TMA: tl.constexpr = False,
        USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        tile = tl.program_id(1)
        b = s // H
        h = s % H
        if HAS_COUNT:
            count = tl.load(COUNT + b, eviction_policy="evict_last")
            y0 = tl.rsqrt(count)
            inv = tl.math.fma(0.5 * y0, tl.math.fma(-count * y0, y0, 1.0), y0)
        else:
            inv = tl.math.div_rn(1.0, tl.math.sqrt_rn(float(N)))
        eta = tl.load(ETA + h, eviction_policy="evict_last")
        n = tile * BT + tl.arange(0, BT)
        r = tl.arange(0, BR)
        if USE_TMA:
            x_base = X + b.to(tl.int64) * N_rt * W_rt
            e_base = E + b.to(tl.int64) * N_rt * HD_rt
            dx_base = DX + b.to(tl.int64) * N_rt * W_rt
            desc_a = tl.make_tensor_descriptor(
                x_base + h * R, shape=[N_rt, R_rt], strides=[W_rt, 1], block_shape=[BT, BR],
            )
            a = desc_a.load([tile * BT, 0])
            desc_c = tl.make_tensor_descriptor(
                x_base + H * R + h * D, shape=[N_rt, D_rt], strides=[W_rt, 1], block_shape=[BT, BD],
            )
            desc_e = tl.make_tensor_descriptor(
                e_base + h * D, shape=[N_rt, D_rt], strides=[HD_rt, 1], block_shape=[BT, BD],
            )
            desc_dc = tl.make_tensor_descriptor(
                dx_base + H * R + h * D, shape=[N_rt, D_rt], strides=[W_rt, 1], block_shape=[BT, BD],
            )
        else:
            a = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                (n[:, None] < N) & (r[None, :] < R),
                0,
            )
        g = tl.load(
            DG + (s * R + r[:, None]) * R + r[None, :],
            (r[:, None] < R) & (r[None, :] < R),
            0,
            eviction_policy="evict_last",
        )
        g = (g + tl.trans(g)).to(tl.float32)
        da = _mixed_dot(a, g) * inv
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            if USE_TMA:
                e = desc_e.load([tile * BT, channel_tile * BD])
            else:
                e = tl.load(
                    E + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0,
                )
            v = tl.load(
                V + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            v_eff = (-eta * v) if UNSCALED_COMPLEMENT else v
            # Finish E/V before loading C/DQ and accumulate directly into `da`
            # so a second [BT, BR] accumulator tile is never materialized.
            if HAS_ROTARY:
                rotated_e = _rotate(e, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED)
                dc = eta * e.to(tl.float32)
                da = _wide_dot_acc(rotated_e, tl.trans(v_eff), da)
                if USE_TMA:
                    c = desc_c.load([tile * BT, channel_tile * BD])
                else:
                    c = tl.load(
                        X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                        (n[:, None] < N) & (d[None, :] < D),
                        0,
                    )
                dq = tl.load(
                    DQ + (s * R + r[:, None]) * D + d[None, :],
                    (r[:, None] < R) & (d[None, :] < D),
                    0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                rotated_c = _rotate(c, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED)
                da = _wide_dot_acc(rotated_c, tl.trans(dq), da)
                correction = _rotate(_mixed_dot(a, dq), COS, SIN, b, n, channel_tile * BD,
                                     N, D, ROTARY_BATCHED, True)
                dc = tl.math.fma(correction, inv, dc)
            else:
                da = _mixed_dot_acc(e, tl.trans(v_eff), da)
                dc = eta * e.to(tl.float32)
                if USE_TMA:
                    c = desc_c.load([tile * BT, channel_tile * BD])
                else:
                    c = tl.load(
                        X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                        (n[:, None] < N) & (d[None, :] < D),
                        0,
                    )
                dq = tl.load(
                    DQ + (s * R + r[:, None]) * D + d[None, :],
                    (r[:, None] < R) & (d[None, :] < D),
                    0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                da = _mixed_dot_acc(c, tl.trans(dq), da)
                dc = tl.math.fma(_mixed_dot(a, dq), inv, dc)
            if USE_TMA:
                desc_dc.store([tile * BT, channel_tile * BD], dc.to(DX.dtype.element_ty))
            else:
                tl.store(
                    DX + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                    dc, (n[:, None] < N) & (d[None, :] < D),
                )
        if USE_PDL:
            tlc.gdc_launch_dependents()
        if USE_TMA:
            desc_da = tl.make_tensor_descriptor(
                dx_base + h * R, shape=[N_rt, R_rt], strides=[W_rt, 1], block_shape=[BT, BR],
            )
            desc_da.store([tile * BT, 0], (da * inv).to(DX.dtype.element_ty))
        else:
            tl.store(
                DX + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                da * inv, (n[:, None] < N) & (r[None, :] < R),
            )

    @jit
    def _chunk_statistics(
        X, E, COUNT, G, Q, DE, COS, SIN,
        N_rt, R_rt, D_rt, W_rt, HD_rt,
        N: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        T: tl.constexpr, GRAD: tl.constexpr, HAS_COUNT: tl.constexpr,
        HAS_ROTARY: tl.constexpr, ROTARY_BATCHED: tl.constexpr,
        BR: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr,
        SCALE_OUTPUT: tl.constexpr = True,
        USE_TMA: tl.constexpr = False,
        TRANSPOSE_Q: tl.constexpr = False,
        USE_PDL: tl.constexpr = False,
    ):
        # SSD-style chunk accumulation reduces partial-state traffic without
        # holding a full token chunk in registers. Statistics remain FP32.
        if USE_PDL:
            tlc.gdc_wait()
        BK: tl.constexpr = 64
        s, tile = tl.program_id(0), tl.program_id(1)
        b, h = s // H, s % H
        if SCALE_OUTPUT:
            count = tl.load(COUNT + b, eviction_policy="evict_last") if HAS_COUNT else float(N)
            inv_count = tl.math.div_rn(1.0, count)
            inv_sqrt = tl.math.div_rn(1.0, tl.math.sqrt_rn(count))
        r = tl.arange(0, BR)
        if USE_TMA:
            x_base = X + b.to(tl.int64) * N_rt * W_rt
            desc_a = tl.make_tensor_descriptor(
                x_base + h * R, shape=[N_rt, R_rt], strides=[W_rt, 1], block_shape=[BK, BR],
            )
            desc_c = tl.make_tensor_descriptor(
                x_base + H * R + h * D, shape=[N_rt, D_rt], strides=[W_rt, 1], block_shape=[BK, BD],
            )
            if GRAD:
                e_base = E + b.to(tl.int64) * N_rt * HD_rt
                desc_e = tl.make_tensor_descriptor(
                    e_base + h * D, shape=[N_rt, D_rt], strides=[HD_rt, 1], block_shape=[BK, BD],
                )
        gram = tl.zeros((BR, BR), tl.float32)
        de = tl.full((), 0.0, tl.float32)
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            if TRANSPOSE_Q:
                q_hi = tl.zeros((BD, BR), tl.float32)
                if HAS_ROTARY:
                    q_lo = tl.zeros((BD, BR), tl.float32)
            else:
                q_hi = tl.zeros((BR, BD), tl.float32)
                if HAS_ROTARY:
                    q_lo = tl.zeros((BR, BD), tl.float32)
            for step in range(tl.cdiv(BT, BK)):
                row_off = tile * BT + step * BK
                n = row_off + tl.arange(0, BK)
                if USE_TMA:
                    a = desc_a.load([row_off, 0])
                else:
                    a = tl.load(
                        X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                        (n[:, None] < N) & (r[None, :] < R), 0,
                    )
                if not GRAD and channel_tile == 0:
                    gram = tl.dot(tl.trans(a), a, gram)
                if USE_TMA:
                    c = desc_c.load([row_off, channel_tile * BD])
                else:
                    c = tl.load(
                        X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                        (n[:, None] < N) & (d[None, :] < D), 0,
                    )
                if GRAD:
                    if USE_TMA:
                        value = desc_e.load([row_off, channel_tile * BD])
                    else:
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
                    if TRANSPOSE_Q:
                        rt = tl.trans(rotated)
                        hi = rt.to(tl.bfloat16)
                        lo = (rt - hi.to(tl.float32)).to(tl.bfloat16)
                        q_lo = tl.dot(lo, a, q_lo)
                        q_hi = tl.dot(hi, a, q_hi)
                    else:
                        hi = rotated.to(tl.bfloat16)
                        lo = (rotated - hi.to(tl.float32)).to(tl.bfloat16)
                        # Keep the low-word sum separate from the high-word sum.
                        q_lo = tl.dot(tl.trans(a), lo, q_lo)
                        q_hi = tl.dot(tl.trans(a), hi, q_hi)
                else:
                    if TRANSPOSE_Q:
                        q_hi = tl.dot(tl.trans(value), a, q_hi)
                    else:
                        q_hi = tl.dot(tl.trans(a), value, q_hi)
            q_total = (q_lo + q_hi) if HAS_ROTARY else q_hi
            if TRANSPOSE_Q:
                q_total = tl.trans(q_total)
            if SCALE_OUTPUT:
                q_total = q_total * inv_sqrt
            tl.store(
                Q + ((s * T + tile) * R + r[:, None]) * D + d[None, :],
                q_total,
                (r[:, None] < R) & (d[None, :] < D),
            )
        if not GRAD:
            if SCALE_OUTPUT:
                gram = gram * inv_count
            if USE_PDL:
                tlc.gdc_launch_dependents()
            tl.store(
                G + ((s * T + tile) * R + r[:, None]) * R + r[None, :],
                gram, (r[:, None] < R) & (r[None, :] < R),
            )
        if GRAD:
            if USE_PDL:
                tlc.gdc_launch_dependents()
            tl.store(DE + s * T + tile, de)

    return _statistics, _readout, _token_backward, _chunk_statistics


@lru_cache(maxsize=1)
def _stats_reduction_kernels():
    from triton import jit
    import triton.language as tl
    import triton.language.extra.cuda as tlc

    @jit
    def reduce_fwd(
        G_IN, Q_IN, COUNT, G_OUT, Q_OUT, T_rt, N_f32,
        H: tl.constexpr, R_SQ: tl.constexpr, R_D: tl.constexpr,
        GRAM_BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
        HAS_COUNT: tl.constexpr, USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        blk = tl.program_id(1)
        b = s // H
        count = tl.load(COUNT + b, eviction_policy="evict_last") if HAS_COUNT else N_f32
        if blk < GRAM_BLOCKS:
            idx = blk * BLOCK + tl.arange(0, BLOCK)
            mask = idx < R_SQ
            acc = tl.zeros((BLOCK,), tl.float32)
            for t in range(T_rt):
                acc += tl.load(G_IN + (s * T_rt + t) * R_SQ + idx, mask, 0.0)
            inv_count = tl.math.div_rn(1.0, count)
            if USE_PDL:
                tlc.gdc_launch_dependents()
            tl.store(G_OUT + s * R_SQ + idx, acc * inv_count, mask)
        else:
            c_blk = blk - GRAM_BLOCKS
            idx = c_blk * BLOCK + tl.arange(0, BLOCK)
            mask = idx < R_D
            acc = tl.zeros((BLOCK,), tl.float32)
            for t in range(T_rt):
                acc += tl.load(Q_IN + (s * T_rt + t) * R_D + idx, mask, 0.0)
            inv_sqrt = tl.math.div_rn(1.0, tl.math.sqrt_rn(count))
            if USE_PDL:
                tlc.gdc_launch_dependents()
            tl.store(Q_OUT + s * R_D + idx, acc * inv_sqrt, mask)

    @jit
    def reduce_bwd(
        Q_IN, COUNT, Q_OUT, T_rt, N_f32,
        H: tl.constexpr, R_D: tl.constexpr, BLOCK: tl.constexpr,
        HAS_COUNT: tl.constexpr, USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        blk = tl.program_id(1)
        b = s // H
        count = tl.load(COUNT + b, eviction_policy="evict_last") if HAS_COUNT else N_f32
        idx = blk * BLOCK + tl.arange(0, BLOCK)
        mask = idx < R_D
        acc = tl.zeros((BLOCK,), tl.float32)
        for t in range(T_rt):
            acc += tl.load(Q_IN + (s * T_rt + t) * R_D + idx, mask, 0.0)
        inv_sqrt = tl.math.div_rn(1.0, tl.math.sqrt_rn(count))
        if USE_PDL:
            tlc.gdc_launch_dependents()
        tl.store(Q_OUT + s * R_D + idx, acc * inv_sqrt, mask)

    return reduce_fwd, reduce_bwd


def _no_frame_block_size(size: int) -> int:
    return max(16, 1 << (size - 1).bit_length())


def _no_frame_statistics(projected, heads, rank, gradient=None, counts=None,
                         rotary_cos=None, rotary_sin=None):
    batch, length, width = projected.shape
    head_dim = (width - heads * rank) // heads
    # Fuse short statistics without a partial reduction. Wide-rank gradient
    # tiles need eight warps to avoid register spills; short channel tiles stay <=64.
    token_tile = _no_frame_block_size(length) if length <= 256 else 128
    # Rotated FP32 content needs more live registers than the BF16-only path.
    if rotary_cos is not None:
        token_tile = min(token_tile, 128)
    kernels = _no_frame_kernels()
    kernel = kernels[0]
    use_chunk = False
    # Use larger partials only when their grid still supplies two CTAs per SM.
    if length >= 512:
        chunk_tiles = (length + 511) // 512
        multiprocessors = _multiprocessor_count(_resolved_device_index(projected.device))
        if batch * heads * chunk_tiles >= 2 * multiprocessors:
            token_tile = 512
            kernel = kernels[3]
            use_chunk = True
    channel_limit = (
        128 if use_chunk and (rank <= 32 or rotary_cos is None) else 64
    )
    block_r = _no_frame_block_size(rank)
    block_d = min(channel_limit, _no_frame_block_size(head_dim))
    tiles = (length + token_tile - 1) // token_tile
    scale_in_kernel = tiles == 1
    is_hopper = _is_hopper_or_newer(projected.device)
    use_tma = (
        is_hopper
        and head_dim % 8 == 0
        and projected.data_ptr() % 16 == 0
        and (gradient is None or gradient.data_ptr() % 16 == 0)
    )
    if use_tma:
        _ensure_triton_allocator()
    transpose_q = _is_hopper_wgmma(projected.device) and block_d >= 64 and block_r < 64
    options = dict(device=projected.device, dtype=torch.float32)
    cross = torch.empty((batch * heads, tiles, rank, head_dim), **options)
    if gradient is None:
        gram = torch.empty((batch * heads, tiles, rank, rank), **options)
        eta_gradient = cross
    else:
        gram = cross
        eta_gradient = torch.empty((batch * heads, tiles), **options)
    launch_kwargs = {"launch_pdl": True} if is_hopper else {}
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
        rank,
        head_dim,
        width,
        heads * head_dim,
        length,
        heads,
        rank,
        head_dim,
        tiles,
        gradient is not None,
        counts is not None,
        rotary_cos is not None,
        rotary_cos is not None and rotary_cos.shape[0] > 1,
        block_r,
        block_d,
        token_tile,
        scale_in_kernel,
        use_tma,
        transpose_q,
        is_hopper,
        num_warps=(
            8
            if block_d > 64
            or block_r >= 64
            or head_dim % 16 != 0
            or (token_tile == 256 and block_d >= 64)
            or (
                gradient is not None
                and (
                    (token_tile == 256 and rank >= 32)
                    or (rotary_cos is not None and (rank >= 32 or block_d >= 64))
                )
            )
            else 4
        ),
        num_stages=3 if use_tma else 1,
        **launch_kwargs,
    )
    if gradient is None:
        del eta_gradient
        if tiles == 1:
            return (
                gram[:, 0].view(batch, heads, rank, rank),
                cross[:, 0].view(batch, heads, rank, head_dim),
            )
        if tiles < 32:
            gram_out = gram.view(batch, heads, tiles, rank, rank).sum(dim=2)
            cross_out = cross.view(batch, heads, tiles, rank, head_dim).sum(dim=2)
            del gram, cross
            if counts is None:
                gram_out.mul_(1.0 / float(length))
                cross_out.mul_(1.0 / math.sqrt(float(length)))
            else:
                gram_out.mul_((1.0 / counts).view(batch, 1, 1, 1))
                cross_out.mul_((1.0 / torch.sqrt(counts)).view(batch, 1, 1, 1))
            return gram_out, cross_out
        gram_out = torch.empty((batch, heads, rank, rank), **options)
        cross_out = torch.empty((batch, heads, rank, head_dim), **options)
        r_sq = rank * rank
        r_d = rank * head_dim
        block_red = 256
        gram_blocks = (r_sq + block_red - 1) // block_red
        cross_blocks = (r_d + block_red - 1) // block_red
        _stats_reduction_kernels()[0][(batch * heads, gram_blocks + cross_blocks)](
            gram, cross, counts, gram_out, cross_out, tiles, float(length),
            heads, r_sq, r_d, gram_blocks, block_red, counts is not None, is_hopper,
            num_warps=4, num_stages=1, **launch_kwargs,
        )
        del gram, cross
        return gram_out, cross_out
    del gram
    if tiles == 1:
        cross_out = cross[:, 0].view(batch, heads, rank, head_dim)
        eta_out = eta_gradient[:, 0].view(batch, heads).sum(0)
        return cross_out, eta_out
    if tiles < 32:
        cross_out = cross.view(batch, heads, tiles, rank, head_dim).sum(dim=2)
        del cross
        if counts is None:
            cross_out.mul_(1.0 / math.sqrt(float(length)))
        else:
            cross_out.mul_((1.0 / torch.sqrt(counts)).view(batch, 1, 1, 1))
        return cross_out, eta_gradient.view(batch, heads, tiles).sum((0, 2))
    cross_out = torch.empty((batch, heads, rank, head_dim), **options)
    r_d = rank * head_dim
    block_red = 256
    cross_blocks = (r_d + block_red - 1) // block_red
    _stats_reduction_kernels()[1][(batch * heads, cross_blocks)](
        cross, counts, cross_out, tiles, float(length),
        heads, r_d, block_red, counts is not None, is_hopper,
        num_warps=4, num_stages=1, **launch_kwargs,
    )
    del cross
    return cross_out, eta_gradient.view(batch, heads, tiles).sum((0, 2))


@lru_cache(maxsize=1)
def _compact_kernels():
    from triton import jit
    import triton.language as tl
    from triton.language.extra import libdevice

    @jit
    def core_forward(
        BASE, FACTOR, SYSTEM, DIFFERENCE,
        R: tl.constexpr, BR: tl.constexpr,
        STORE_FACTOR: tl.constexpr = True,
    ):
        h = tl.program_id(0)
        i = tl.arange(0, BR)
        mask = (i[:, None] < R) & (i[None, :] < R)
        offset = h * R * R + i[:, None] * R + i[None, :]
        b = tl.load(BASE + offset, mask, 0.0)
        b_diag = tl.load(BASE + h * R * R + i * (R + 1), i < R, 0.0)
        shifted = b_diag + 0.5413248546129181
        diagonal = tl.where(shifted > 20.0, shifted, libdevice.log1p(libdevice.exp(shifted)))
        f = tl.where(i[:, None] > i[None, :], b, tl.where(i[:, None] == i[None, :], diagonal[:, None], 0.0))
        if STORE_FACTOR:
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
        HAS_FACTOR: tl.constexpr = True,
    ):
        h = tl.program_id(0)
        i = tl.arange(0, BR)
        mask = (i[:, None] < R) & (i[None, :] < R)
        offset = h * R * R + i[:, None] * R + i[None, :]
        k = tl.arange(0, 16)
        g = tl.full((BR, BR), 0.0, tl.float32)
        for start in range(tl.cdiv(R, 16)):
            kk = start * 16 + k
            mask_rk = (i[:, None] < R) & (kk[None, :] < R)
            u_tile = tl.load(ADJOINT + h * R * R + i[:, None] * R + kk[None, :], mask_rk, 0.0)
            m_tile = tl.load(MAPPING + h * R * R + i[:, None] * R + kk[None, :], mask_rk, 0.0)
            diag_rk = (i[:, None] == kk[None, :]) & (i[:, None] < R)
            m_tile += tl.where(diag_rk, 1.0, 0.0)
            g = tl.dot(-u_tile, tl.trans(m_tile), g, input_precision="tf32x3")
        gt = tl.trans(g)
        sym_g = g + gt
        skew_g = g - gt
        b_diag = tl.load(BASE + h * R * R + i * (R + 1), i < R, 0.0)
        shifted = b_diag + 0.5413248546129181
        if HAS_FACTOR:
            f = tl.load(FACTOR + offset, mask, 0.0)
        else:
            b = tl.load(BASE + offset, mask, 0.0)
            diag_f = tl.where(shifted > 20.0, shifted, libdevice.log1p(libdevice.exp(shifted)))
            f = tl.where(i[:, None] > i[None, :], b, tl.where(i[:, None] == i[None, :], diag_f[:, None], 0.0))
        core_grad = tl.dot(sym_g, f, input_precision="tf32x3")
        sigmoid = tl.where(shifted > 20.0, 1.0, 1.0 / (1.0 + libdevice.exp(-shifted)))
        d = tl.where(
            i[:, None] > i[None, :],
            core_grad,
            tl.where(i[:, None] == i[None, :], core_grad * sigmoid[:, None], skew_g),
        )
        tl.store(OUT + offset, d, mask)

    @jit
    def complement(
        RAW, ETA, DERIVATIVE, H: tl.constexpr, BLOCK: tl.constexpr,
        STORE_DERIVATIVE: tl.constexpr = True,
    ):
        h = tl.arange(0, BLOCK)
        raw = tl.load(RAW + h, h < H, 0)
        scale: tl.constexpr = 1.0 - 1.1920928955078125e-7
        # Native tanh avoids cancellation near zero. Keep the exponential
        # derivative separately: 1-tanh(raw)^2 loses the FP32 tail gradient.
        value = scale * libdevice.tanh(raw)
        tl.store(ETA + h, value, h < H)
        if STORE_DERIVATIVE:
            exponent = libdevice.exp(-2.0 * tl.abs(raw))
            derivative = scale * 4.0 * exponent / ((1.0 + exponent) * (1.0 + exponent))
            tl.store(DERIVATIVE + h, derivative, h < H)

    return core_forward, core_backward, complement


def _compact_core_forward(base, *, store_factor: bool = True):
    system = torch.empty_like(base, memory_format=torch.contiguous_format)
    difference = torch.empty_like(system)
    core_factor = torch.empty_like(system) if store_factor else None
    rank = base.shape[-1]
    _compact_kernels()[0][(base.shape[0],)](
        base, system if core_factor is None else core_factor, system, difference,
        rank, _no_frame_block_size(rank), store_factor,
        num_warps=8 if rank > 32 else 4, num_stages=1,
    )
    return core_factor, system, difference


def _compact_core_backward(adjoint, mapping, core_factor, base, *, out=None):
    output = (
        torch.empty_like(base, memory_format=torch.contiguous_format)
        if out is None
        else out
    )
    rank = base.shape[-1]
    has_factor = core_factor is not None
    _compact_kernels()[1][(base.shape[0],)](
        adjoint, mapping, base if core_factor is None else core_factor, base, output,
        rank, _no_frame_block_size(rank), has_factor,
        num_warps=8 if rank > 32 else 4, num_stages=1,
    )
    return output


def _compact_complement(raw, *, record: bool = True, out_eta=None, out_derivative=None):
    eta = torch.empty_like(raw) if out_eta is None else out_eta
    derivative = (
        (torch.empty_like(raw) if out_derivative is None else out_derivative)
        if record
        else None
    )
    _compact_kernels()[2][(1,)](
        raw, eta, eta if derivative is None else derivative,
        raw.numel(), _no_frame_block_size(raw.numel()), record, num_warps=4,
    )
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
    def eta_trace(
        G, DIRECT, DERIVATIVE, OUT, R: tl.constexpr, BR: tl.constexpr,
        SUBTRACT_TRACE: tl.constexpr = True,
    ):
        h = tl.program_id(0)
        if SUBTRACT_TRACE:
            i = tl.arange(0, BR)
            trace = tl.sum(tl.load(G + h*R*R + i*(R+1), i < R, 0), 0)
            tl.store(OUT+h, (tl.load(DIRECT+h)-trace)*tl.load(DERIVATIVE+h))
        else:
            tl.store(OUT+h, tl.load(DIRECT+h)*tl.load(DERIVATIVE+h))

    @jit
    def cholesky_product(F, H, ETA, INVERSE, HEADS: tl.constexpr, R: tl.constexpr,
                         BR: tl.constexpr, COMPLEMENT: tl.constexpr, FUSE_CONJUGATE: tl.constexpr,
                         SUBTRACT_ETA: tl.constexpr = True):
        # H=U Z^T is fresh, unsaved storage, owned in full by one CTA.
        # L^T dL = -M H^T - M^T H, M=F-eta I. Apply the Cholesky
        # symmetric lower projection here and fuse T^T @ sym @ T in registers.
        s = tl.program_id(0)
        h = s % HEADS
        i = tl.arange(0, BR)
        eta = tl.load(ETA+h)
        mask_rr = (i[:, None] < R) & (i[None, :] < R)
        if COMPLEMENT:
            value = tl.load(H+s*R*R+i[:, None]*R+i[None, :], mask_rr, 0)
            product = eta * (value + tl.trans(value))
        else:
            # Accumulate products serially to bound live tiles at r48/r64.
            k = tl.arange(0, 16)
            product = tl.full((BR, BR), 0, tl.float32)
            for start in range(tl.cdiv(R, 16)):
                kk = start*16+k
                f = tl.load(F+h*R*R+i[:, None]*R+kk[None, :],
                            (i[:, None] < R) & (kk[None, :] < R), 0)
                if SUBTRACT_ETA:
                    diagonal = (i[:, None] == kk[None, :]) & (i[:, None] < R)
                    f -= tl.where(diagonal, eta, 0)
                ht = tl.load(H+s*R*R+i[None, :]*R+kk[:, None],
                             (kk[:, None] < R) & (i[None, :] < R), 0)
                product = tl.dot(-f, ht, product, input_precision="tf32x3")
                ft = tl.load(F+h*R*R+kk[None, :]*R+i[:, None],
                             (i[:, None] < R) & (kk[None, :] < R), 0)
                if SUBTRACT_ETA:
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
            gram = tl.dot(tl.trans(t), mid, input_precision="tf32x3")
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
        SUBTRACT_ETA: tl.constexpr = True,
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
                m_mat = tl.load(MAPPING + h * R * R + i[:, None] * R + j[None, :], mask_rr, 0.0)
                if TRANSPOSE_MAP:
                    m_mat = tl.trans(m_mat)
                if SUBTRACT_ETA:
                    diagonal = (i[:, None] == j[None, :]) & (i[:, None] < R)
                    m_mat -= tl.where(diagonal, tl.load(ETA + h), 0.0)
                mid = tl.dot(m_mat, u, input_precision="tf32x3")
            out = tl.dot(tl.trans(t), mid, input_precision="tf32x3")
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
                if SUBTRACT_ETA:
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
                    if SUBTRACT_ETA:
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
        SUBTRACT_ETA: tl.constexpr = True,
    ):
        s = tl.program_id(0)
        h = s % H
        i = tl.arange(0, BR)
        j = tl.arange(0, BR)
        mask_rr = (i[:, None] < R) & (j[None, :] < R)
        eta = tl.load(ETA + h)
        t = tl.load(INVERSE + s * R * R + i[:, None] * R + j[None, :], mask_rr, 0.0)
        if not COMPLEMENT:
            m_mat = tl.trans(
                tl.load(MAPPING + h * R * R + i[:, None] * R + j[None, :], mask_rr, 0.0)
            )
            if SUBTRACT_ETA:
                diagonal = (i[:, None] == j[None, :]) & (i[:, None] < R)
                m_mat -= tl.where(diagonal, eta, 0.0)
        prod_acc = tl.full((BR, BR), 0.0, tl.float32)
        c = tl.arange(0, BC)
        for start in range(tl.cdiv(D, BC)):
            col = start * BC + c
            mask_rd = (i[:, None] < R) & (col[None, :] < D)
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
                mid = tl.dot(m_mat, u, input_precision="tf32x3")
            dq = tl.dot(tl.trans(t), mid, input_precision="tf32x3")
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


def _compact_chain(
    inverse, rhs, mapping, eta, *, complement, transpose_map, store_first=True,
    subtract_eta: bool = True,
):
    batch, heads, rank, columns = rhs.shape
    second = torch.empty((batch, heads, rank, columns), device=rhs.device, dtype=torch.float32)
    first = torch.empty_like(second) if store_first or rank > 32 else second
    block_r = _no_frame_block_size(rank)
    block_c = min(64, _no_frame_block_size(columns))
    _compact_adjoint_kernels()[2][(batch * heads, (columns + block_c - 1) // block_c)](
        inverse, rhs, mapping, eta, first, second,
        heads, rank, columns, *rhs.stride()[-2:],
        16, block_r, block_c, complement, transpose_map, store_first, subtract_eta,
        num_warps=4, num_stages=1,
    )
    if not store_first and first is not second:
        del first
    return (first if store_first else None), second


def _should_overlap_core_stream(tokens: int) -> bool:
    del tokens
    return torch.cuda.is_current_stream_capturing()


def _launch_compact_core_forward(base, raw, *, record: bool, allow_overlap: bool = True):
    device_index = _resolved_device_index(raw.device)
    main_stream = torch.cuda.current_stream(device_index) if allow_overlap else None
    side_stream, fork_event, join_event = (
        _side_stream_bundle(device_index) if allow_overlap else (None, None, None)
    )
    if not allow_overlap or side_stream == main_stream:
        eta, eta_derivative = _compact_complement(raw, record=record)
        _, system, difference = _compact_core_forward(base, store_factor=False)
        lu, pivots = _compact_lu(system)
        del system
        mapping = _compact_getrs(lu, pivots, difference)
        del difference
        if not record:
            del lu, pivots
            lu = pivots = None
        return eta, eta_derivative, lu, pivots, mapping, None, False

    capturing = torch.cuda.is_current_stream_capturing()
    eta = torch.empty_like(raw)
    eta_derivative = torch.empty_like(raw) if record else None
    fork_event.record(main_stream)
    side_stream.wait_event(fork_event)
    if not capturing:
        base.record_stream(side_stream)
        raw.record_stream(side_stream)
        eta.record_stream(side_stream)
        if eta_derivative is not None:
            eta_derivative.record_stream(side_stream)
    with torch.cuda.stream(side_stream):
        _compact_complement(
            raw, record=record, out_eta=eta, out_derivative=eta_derivative,
        )
        _, system, difference = _compact_core_forward(base, store_factor=False)
        lu, pivots = _compact_lu(system)
        del system
        mapping = _compact_getrs(lu, pivots, difference)
        del difference
        if not record:
            del lu, pivots
            lu = pivots = None
    join_event.record(side_stream)
    return eta, eta_derivative, lu, pivots, mapping, join_event, capturing


def _no_frame_compact(
    gram, cross, base, raw, mode, *, record=True, core_state=None,
    subtract_eta_in_chain: bool = True,
):
    # Factor I+G once; overlap head-local core LU/solve on a side stream when worthwhile.
    if mode != 2 and core_state is None:
        core_state = _launch_compact_core_forward(
            base, raw, record=record,
            allow_overlap=torch.cuda.is_current_stream_capturing(),
        )
    inverse = torch.ops.memsolve_equilibrium.compact_inverse(gram)
    del gram
    coordinates = lu = pivots = mapping = None
    if mode == 2:
        eta, eta_derivative = _compact_complement(raw, record=record)
    else:
        eta, eta_derivative, lu, pivots, mapping, join_event, capturing = core_state
        core_state = None
        coordinates = base if record else None
        if join_event is not None:
            main_stream = torch.cuda.current_stream(_resolved_device_index(inverse.device))
            main_stream.wait_event(join_event)
            if not capturing:
                mapping.record_stream(main_stream)
                if lu is not None:
                    lu.record_stream(main_stream)
                    pivots.record_stream(main_stream)
    # Complement keeps an unscaled coefficient: token kernels apply -eta
    # after the compensated product, including eta=0 without division.
    state, coefficient = _compact_chain(
        inverse, cross, mapping, eta, complement=mode == 2, transpose_map=False,
        store_first=record, subtract_eta=subtract_eta_in_chain,
    )
    del cross
    if not record:
        return coefficient, eta, ()
    tape = (inverse, state, coordinates, lu, pivots, mapping, eta_derivative)
    return coefficient, eta, tape


def _no_frame_compact_backward(
    coefficient, eta, tape, gradient, eta_gradient, mode,
    *, defer_join: bool = False, allow_overlap: bool | None = None,
    subtract_eta_in_chain: bool = True,
):
    inverse, state, coordinates, lu, pivots, mapping, eta_derivative = tape[:7]
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
            block_r, block_c, mode == 2, subtract_eta_in_chain,
            num_warps=4, num_stages=1,
        )
        del gradient, state
    else:
        u, cross_gradient = _compact_chain(
            inverse, gradient, mapping, eta, complement=mode == 2, transpose_map=True,
            subtract_eta=subtract_eta_in_chain,
        )
        del gradient
        product = _compact_product(u, state.mT)
        del u, state
    map_gradient = product.sum(0)
    raw_gradient = torch.empty_like(eta_gradient)
    base_gradient = None
    pending_join = None
    if mode != 2:
        if allow_overlap is None:
            allow_overlap = torch.cuda.is_current_stream_capturing()
        base_gradient = torch.empty_like(coordinates, memory_format=torch.contiguous_format)
        if allow_overlap:
            device_index = _resolved_device_index(product.device)
            main_stream = torch.cuda.current_stream(device_index)
            side_stream, fork_event, join_event = _side_stream_bundle(device_index)
        else:
            main_stream = side_stream = fork_event = join_event = None
        if allow_overlap and side_stream != main_stream:
            capturing = torch.cuda.is_current_stream_capturing()
            fork_event.record(main_stream)
            side_stream.wait_event(fork_event)
            if not capturing:
                for tensor in (
                    map_gradient, eta_gradient, eta_derivative,
                    lu, pivots, mapping, coordinates,
                    raw_gradient, base_gradient,
                ):
                    tensor.record_stream(side_stream)
            with torch.cuda.stream(side_stream):
                kernels[0][(heads,)](
                    map_gradient, eta_gradient, eta_derivative, raw_gradient,
                    rank, block_r, subtract_eta_in_chain, num_warps=4,
                )
                adjoint = _compact_getrs(lu, pivots, map_gradient, transpose=True)
                _compact_core_backward(
                    adjoint, mapping, None, coordinates, out=base_gradient,
                )
                del adjoint
            join_event.record(side_stream)
            pending_join = join_event
        else:
            kernels[0][(heads,)](
                map_gradient, eta_gradient, eta_derivative, raw_gradient,
                rank, block_r, subtract_eta_in_chain, num_warps=4,
            )
            adjoint = _compact_getrs(lu, pivots, map_gradient, transpose=True)
            _compact_core_backward(
                adjoint, mapping, None, coordinates, out=base_gradient,
            )
            del adjoint
        del eta_gradient, eta_derivative, lu, pivots, map_gradient, coordinates
    else:
        kernels[0][(heads,)](
            map_gradient, eta_gradient, eta_derivative, raw_gradient,
            rank, block_r, subtract_eta_in_chain, num_warps=4,
        )
        del eta_gradient, eta_derivative, lu, pivots, map_gradient, coordinates
    fuse_conjugate = rank <= 32
    kernels[1][(batch * heads,)](
        mapping, product, eta, inverse, heads, rank, block_r,
        mode == 2, fuse_conjugate, subtract_eta_in_chain,
        num_warps=4, num_stages=1,
    )
    del mapping
    if not fuse_conjugate:
        kernels[4][(batch * heads,)](
            inverse, product, rank, 16 if rank % 32 == 16 else 32, block_r,
            num_warps=4, num_stages=1,
        )
    del inverse
    if defer_join:
        return product, cross_gradient, base_gradient, raw_gradient, pending_join
    if pending_join is not None:
        torch.cuda.current_stream(_resolved_device_index(product.device)).wait_event(
            pending_join
        )
    return product, cross_gradient, base_gradient, raw_gradient


def _run_readout(
    projected, coefficient, eta, counts, rotary_cos, rotary_sin,
    *, heads: int, rank: int, head_dim: int, mode: int,
) -> torch.Tensor:
    batch, length, width = projected.shape
    output = torch.empty(
        (batch, length, heads * head_dim),
        device=projected.device,
        dtype=projected.dtype,
    )
    is_hopper = _is_hopper_or_newer(projected.device)
    block_r = _no_frame_block_size(rank)
    if block_r >= 64 and head_dim > 64:
        block_d = 32 if (rotary_cos is not None or head_dim % 64 != 0) else 64
        num_warps = 8
    elif not is_hopper:
        block_d = min(128, _no_frame_block_size(head_dim))
        num_warps = 8 if (block_d > 64 and (block_r > 16 or rotary_cos is not None)) else 4
    else:
        block_d = min(128, _no_frame_block_size(head_dim))
        num_warps = 8 if block_d > 64 else 4
    use_tma = (
        is_hopper
        and head_dim % 8 == 0
        and projected.data_ptr() % 16 == 0
        and output.data_ptr() % 16 == 0
    )
    if use_tma:
        _ensure_triton_allocator()
    launch_kwargs = {"launch_pdl": True} if is_hopper else {}
    _no_frame_kernels()[1][(batch * heads, (length + 127) // 128)](
        projected,
        coefficient,
        eta,
        counts,
        output,
        rotary_cos,
        rotary_sin,
        length,
        rank,
        head_dim,
        width,
        heads * head_dim,
        length,
        heads,
        rank,
        head_dim,
        counts is not None,
        rotary_cos is not None,
        rotary_cos is not None and rotary_cos.shape[0] > 1,
        block_r,
        block_d,
        128,
        mode == 2,
        use_tma,
        is_hopper,
        num_warps=num_warps,
        num_stages=3 if use_tma else 1,
        **launch_kwargs,
    )
    return output


def _run_token_backward(
    projected, gradient, coefficient, gram_gradient, cross_gradient, eta,
    counts, rotary_cos, rotary_sin, *, heads: int, rank: int, head_dim: int, mode: int,
) -> torch.Tensor:
    batch, length, width = projected.shape
    projected_gradient = torch.empty_like(projected)
    # Narrow channel tiles bound the live FP32 rotation/adjoint operands.
    is_hopper = _is_hopper_or_newer(projected.device)
    token_tile, channel_tile = 128, 32 if rank <= 32 else 16
    if rotary_cos is not None:
        token_tile, channel_tile = 128, 16
    elif rank > 32 and head_dim % 8 != 0 and not is_hopper:
        token_tile = 64
    use_tma = (
        is_hopper
        and head_dim % 8 == 0
        and projected.data_ptr() % 16 == 0
        and gradient.data_ptr() % 16 == 0
        and projected_gradient.data_ptr() % 16 == 0
    )
    if use_tma:
        _ensure_triton_allocator()
    launch_kwargs = {"launch_pdl": True} if is_hopper else {}
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
        rank,
        head_dim,
        width,
        heads * head_dim,
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
        use_tma,
        is_hopper,
        num_warps=8 if (rank > 32 or head_dim % 16 != 0) else 4,
        num_stages=3 if use_tma else 1,
        **launch_kwargs,
    )
    return projected_gradient


def _no_frame_forward(projected, base, raw, counts, *, record, sequence_parallel=None,
                      rotary_cos=None, rotary_sin=None):
    heads, rank = base.shape[:2]
    batch, length, width = projected.shape
    head_dim = (width - heads * rank) // heads
    mode = 2 if base.numel() == 0 else 1
    core_state = None
    if mode != 2:
        with torch.no_grad():
            core_state = _launch_compact_core_forward(
                base, raw, record=record,
                allow_overlap=_should_overlap_core_stream(batch * length),
            )
    stats = list(
        _no_frame_statistics(projected, heads, rank, counts=counts,
                             rotary_cos=rotary_cos, rotary_sin=rotary_sin)
    )
    if sequence_parallel is not None:
        stats[:] = sequence_parallel.sum_tensors(*stats)
    with torch.no_grad():
        coefficient, eta, tape = _no_frame_compact(
            stats.pop(0), stats.pop(0), base, raw, mode,
            record=record, core_state=core_state,
        )
    core_state = None
    output = _run_readout(
        projected, coefficient, eta, counts, rotary_cos, rotary_sin,
        heads=heads, rank=rank, head_dim=head_dim, mode=mode,
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
        heads, rank = base.shape[:2]
        ctx.heads, ctx.rank = heads, rank
        scope = getattr(_RECOMPUTE_LOCAL, "active", None)
        if scope is not None and (
            scope.policy is True
            or (scope.policy == "auto" and projected.shape[1] > 512)
        ):
            head_dim = (projected.shape[2] - heads * rank) // heads
            mode = 2 if base.numel() == 0 else 1
            scope.spec = _RecomputeSpec(
                projected.detach(),
                saved[0].detach(),
                saved[1].detach(),
                counts,
                rotary_cos,
                rotary_sin,
                heads,
                rank,
                head_dim,
                mode,
            )
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
            gram_gradient, cross_gradient, base_gradient, raw_gradient, pending_join = (
                _no_frame_compact_backward(
                    None, eta, tape, stats.pop(0), stats.pop(0), mode,
                    defer_join=True,
                    allow_overlap=_should_overlap_core_stream(batch * length),
                )
            )
        projected_gradient = _run_token_backward(
            projected, gradient, coefficient, gram_gradient, cross_gradient, eta,
            counts, rotary_cos, rotary_sin,
            heads=heads, rank=rank, head_dim=head_dim, mode=mode,
        )
        del projected, gradient, coefficient, gram_gradient, cross_gradient, eta
        if pending_join is not None:
            torch.cuda.current_stream(
                _resolved_device_index(projected_gradient.device)
            ).wait_event(pending_join)
        if sequence_parallel is not None:
            # Compact adjoints already include every shard. Parameter SUM
            # must count these replicated gradients exactly once.
            raw_gradient.div_(sequence_parallel.world_size)
            if base_gradient is not None:
                base_gradient.div_(sequence_parallel.world_size)
        return projected_gradient, base_gradient, raw_gradient, None, None, None, None


@lru_cache(maxsize=1)
def _short_kernels():
    from triton import jit
    import triton.language as tl
    import triton.language.extra.cuda as tlc

    @jit
    def _short_rsqrt(count):
        y0 = tl.rsqrt(count)
        return tl.math.fma(0.5 * y0, tl.math.fma(-count * y0, y0, 1.0), y0)

    @jit
    def _short_rcp(count, inv_sqrt):
        r0 = inv_sqrt * inv_sqrt
        return tl.math.fma(r0, tl.math.fma(-count, r0, 1.0), r0)

    @jit
    def _short_mixed_dot(a, b):
        hi = b.to(tl.bfloat16)
        lo = (b - hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(a, lo)
        return tl.dot(a, hi, acc)

    @jit
    def _short_mixed_dot_lhs(a, b):
        hi = a.to(tl.bfloat16)
        lo = (a - hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(lo, b)
        return tl.dot(hi, b, acc)

    @jit
    def _short_wide_dot(a, b):
        ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        result = tl.dot(al, bh)
        result = tl.dot(ah, bl, result)
        return tl.dot(ah, bh, result)

    @jit
    def _short_rotate(
        value, COS, SIN, batch, tokens, channel_start,
        N: tl.constexpr, D: tl.constexpr,
        ROTARY_BATCHED: tl.constexpr, INVERSE: tl.constexpr = False,
    ):
        pairs = channel_start // 2 + tl.arange(0, value.shape[1] // 2)
        table_batch = batch if ROTARY_BATCHED else 0
        offset = ((table_batch * N + tokens[:, None]) * (D // 2) + pairs[None, :])
        mask = (tokens[:, None] < N) & (pairs[None, :] < D // 2)
        cosine = tl.load(COS + offset, mask, 1.0, eviction_policy="evict_last")
        sine = tl.load(SIN + offset, mask, 0.0, eviction_policy="evict_last")
        if INVERSE:
            sine = -sine
        even, odd = tl.split(tl.reshape(value.to(tl.float32),
                                        (value.shape[0], value.shape[1] // 2, 2)))
        real = tl.math.fma(even, cosine, -(odd * sine))
        imag = tl.math.fma(even, sine, odd * cosine)
        return tl.reshape(tl.join(real, imag), value.shape)

    @jit
    def short_stats(
        X, COUNT, COS, SIN, G_TOK, G_RANK, Q_RANK, C_ROT,
        N: tl.constexpr, P: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        HAS_COUNT: tl.constexpr, HAS_ROTARY: tl.constexpr, ROTARY_BATCHED: tl.constexpr,
        MODE: tl.constexpr, BP: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
        USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        b = s // H
        h = s % H
        if HAS_COUNT:
            count = tl.load(COUNT + b, eviction_policy="evict_last")
            inv_sqrt = _short_rsqrt(count)
            inv_count = _short_rcp(count, inv_sqrt)
        else:
            inv_count = tl.math.div_rn(1.0, float(N))
            inv_sqrt = tl.math.div_rn(1.0, tl.math.sqrt_rn(float(N)))
        n = tl.arange(0, BP)
        r = tl.arange(0, BR)
        a_bf16 = tl.load(
            X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
            (n[:, None] < N) & (r[None, :] < R),
            0,
        )
        g_tok = tl.dot(a_bf16, tl.trans(a_bf16)).to(tl.float32) * inv_count
        tl.store(
            G_TOK + (s * P + n[:, None]) * P + n[None, :],
            g_tok,
            (n[:, None] < P) & (n[None, :] < P),
        )
        if MODE == 1:
            g_rank = tl.dot(tl.trans(a_bf16), a_bf16).to(tl.float32) * inv_count
            tl.store(
                G_RANK + (s * R + r[:, None]) * R + r[None, :],
                g_rank,
                (r[:, None] < R) & (r[None, :] < R),
            )
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            c_bf16 = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D),
                0,
            )
            if HAS_ROTARY:
                c_rot = _short_rotate(
                    c_bf16, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED, False,
                )
                if MODE == 1:
                    q = _short_mixed_dot(tl.trans(a_bf16), c_rot) * inv_sqrt
            else:
                c_rot = c_bf16.to(tl.float32)
                if MODE == 1:
                    q = tl.dot(tl.trans(a_bf16), c_bf16).to(tl.float32) * inv_sqrt
            tl.store(
                C_ROT + (s * N + n[:, None]) * D + d[None, :],
                c_rot,
                (n[:, None] < N) & (d[None, :] < D),
            )
            if MODE == 1:
                tl.store(
                    Q_RANK + (s * R + r[:, None]) * D + d[None, :],
                    q,
                    (r[:, None] < R) & (d[None, :] < D),
                )
        if USE_PDL:
            tlc.gdc_launch_dependents()

    @jit
    def short_readout(
        X, C_ROT, T_TOK, V, ETA, COUNT, COS, SIN, RESIDUAL, Y,
        N: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        T_STRIDE: tl.constexpr, RES_ROWS: tl.constexpr,
        HAS_COUNT: tl.constexpr, HAS_ROTARY: tl.constexpr, ROTARY_BATCHED: tl.constexpr,
        MODE: tl.constexpr, STORE_RES: tl.constexpr, LOAD_RES: tl.constexpr,
        BP: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
        USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        channel_tile = tl.program_id(1)
        b = s // H
        h = s % H
        n = tl.arange(0, BP)
        d = channel_tile * BD + tl.arange(0, BD)
        if LOAD_RES:
            res = tl.load(
                RESIDUAL + (s * RES_ROWS + n[:, None]) * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D),
                0.0,
            )
        else:
            j = tl.arange(0, BP)
            t_tok = tl.load(
                T_TOK + (s * T_STRIDE + n[:, None]) * T_STRIDE + j[None, :],
                (n[:, None] < N) & (j[None, :] < N),
                0.0,
                eviction_policy="evict_last",
            )
            c_rot = tl.load(
                C_ROT + (s * N + j[:, None]) * D + d[None, :],
                (j[:, None] < N) & (d[None, :] < D),
                0.0,
            )
            w = tl.dot(t_tok, c_rot, input_precision="tf32x3")
            res = tl.dot(tl.trans(t_tok), w, input_precision="tf32x3")
            if STORE_RES:
                tl.store(
                    RESIDUAL + (s * RES_ROWS + n[:, None]) * D + d[None, :],
                    res,
                    (n[:, None] < N) & (d[None, :] < D),
                )
        eta = tl.load(ETA + h, eviction_policy="evict_last")
        y_rot = eta * res
        if MODE == 1:
            r = tl.arange(0, BR)
            inv_sqrt = (
                _short_rsqrt(tl.load(COUNT + b, eviction_policy="evict_last"))
                if HAS_COUNT
                else tl.math.div_rn(1.0, tl.math.sqrt_rn(float(N)))
            )
            a_bf16 = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                (n[:, None] < N) & (r[None, :] < R),
                0,
            )
            v = tl.load(
                V + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0.0,
                eviction_policy="evict_last",
            )
            y_rot = tl.math.fma(_short_mixed_dot(a_bf16, v), inv_sqrt, y_rot)
        if HAS_ROTARY:
            y_out = _short_rotate(
                y_rot, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED, True,
            )
        else:
            y_out = y_rot
        if USE_PDL:
            tlc.gdc_launch_dependents()
        tl.store(
            Y + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
            y_out,
            (n[:, None] < N) & (d[None, :] < D),
        )

    @jit
    def short_bwd_prep(
        X, DY, T_TOK, RESIDUAL, ETA, COUNT, COS, SIN, U_TOK, E_ROT, DV, SYM_P, DE_OUT,
        N: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        T_STRIDE: tl.constexpr, RES_ROWS: tl.constexpr,
        HAS_COUNT: tl.constexpr, HAS_ROTARY: tl.constexpr, ROTARY_BATCHED: tl.constexpr,
        MODE: tl.constexpr, BP: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
        USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        b = s // H
        h = s % H
        n = tl.arange(0, BP)
        j = tl.arange(0, BP)
        r = tl.arange(0, BR)
        eta = tl.load(ETA + h, eviction_policy="evict_last")
        t_tok = tl.load(
            T_TOK + (s * T_STRIDE + n[:, None]) * T_STRIDE + j[None, :],
            (n[:, None] < N) & (j[None, :] < N),
            0.0,
            eviction_policy="evict_last",
        )
        if MODE == 1:
            inv_sqrt = (
                _short_rsqrt(tl.load(COUNT + b, eviction_policy="evict_last"))
                if HAS_COUNT
                else tl.math.div_rn(1.0, tl.math.sqrt_rn(float(N)))
            )
            a_bf16 = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                (n[:, None] < N) & (r[None, :] < R),
                0,
            )
        p_acc = tl.zeros((BP, BP), tl.float32)
        de_acc = tl.full((), 0.0, tl.float32)
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            dy_bf16 = tl.load(
                DY + (b * N + n[:, None]) * (H * D) + h * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D),
                0,
            )
            if HAS_ROTARY:
                e = _short_rotate(
                    dy_bf16, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED, False,
                )
            else:
                e = dy_bf16.to(tl.float32)
            res = tl.load(
                RESIDUAL + (s * RES_ROWS + n[:, None]) * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D),
                0.0,
            )
            de_acc += tl.sum(tl.sum(e * res, 0), 0)
            w = tl.dot(t_tok, eta * e, input_precision="tf32x3")
            u_tok = tl.dot(tl.trans(t_tok), w, input_precision="tf32x3")
            tl.store(
                U_TOK + (s * N + n[:, None]) * D + d[None, :],
                u_tok,
                (n[:, None] < N) & (d[None, :] < D),
            )
            p_acc = tl.dot(u_tok, tl.trans(res), p_acc, input_precision="tf32x3")
            if MODE == 1:
                tl.store(
                    E_ROT + (s * N + n[:, None]) * D + d[None, :],
                    e,
                    (n[:, None] < N) & (d[None, :] < D),
                )
                if HAS_ROTARY:
                    dv = _short_mixed_dot(tl.trans(a_bf16), e) * inv_sqrt
                else:
                    dv = tl.dot(tl.trans(a_bf16), dy_bf16).to(tl.float32) * inv_sqrt
                tl.store(
                    DV + (s * R + r[:, None]) * D + d[None, :],
                    dv,
                    (r[:, None] < R) & (d[None, :] < D),
                )
        sym_p = -(p_acc + tl.trans(p_acc))
        if USE_PDL:
            tlc.gdc_launch_dependents()
        tl.store(
            SYM_P + (s * N + n[:, None]) * N + j[None, :],
            sym_p,
            (n[:, None] < N) & (j[None, :] < N),
        )
        tl.store(DE_OUT + s, de_acc)

    @jit
    def short_bwd_da(
        X, SYM_P, E_ROT, V, DG_RANK, DQ_RANK, COUNT, COS, SIN, DX,
        N: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        HAS_COUNT: tl.constexpr, HAS_ROTARY: tl.constexpr, ROTARY_BATCHED: tl.constexpr,
        MODE: tl.constexpr, BP: tl.constexpr, BR: tl.constexpr, BR_OUT: tl.constexpr,
        BD: tl.constexpr, USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        r_tile = tl.program_id(1)
        b = s // H
        h = s % H
        n = tl.arange(0, BP)
        j = tl.arange(0, BP)
        r_out = r_tile * BR_OUT + tl.arange(0, BR_OUT)
        inv_sqrt = (
            _short_rsqrt(tl.load(COUNT + b, eviction_policy="evict_last"))
            if HAS_COUNT
            else tl.math.div_rn(1.0, tl.math.sqrt_rn(float(N)))
        )
        a_out = tl.load(
            X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r_out[None, :],
            (n[:, None] < N) & (r_out[None, :] < R),
            0,
        ).to(tl.float32) * inv_sqrt
        sym_p = tl.load(
            SYM_P + (s * N + n[:, None]) * N + j[None, :],
            (n[:, None] < N) & (j[None, :] < N),
            0.0,
        )
        da = _short_wide_dot(sym_p, a_out)
        if MODE == 1:
            rk = tl.arange(0, BR)
            a_full = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + h * R + rk[None, :],
                (n[:, None] < N) & (rk[None, :] < R),
                0,
            ).to(tl.float32) * inv_sqrt
            dg_row = tl.load(
                DG_RANK + (s * R + rk[:, None]) * R + r_out[None, :],
                (rk[:, None] < R) & (r_out[None, :] < R),
                0.0,
                eviction_policy="evict_last",
            )
            dg_col = tl.load(
                DG_RANK + (s * R + r_out[None, :]) * R + rk[:, None],
                (rk[:, None] < R) & (r_out[None, :] < R),
                0.0,
                eviction_policy="evict_last",
            )
            da += _short_wide_dot(a_full, dg_row + dg_col)
            for channel_tile in range(tl.cdiv(D, BD)):
                d = channel_tile * BD + tl.arange(0, BD)
                e_rot = tl.load(
                    E_ROT + (s * N + n[:, None]) * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0.0,
                )
                v = tl.load(
                    V + (s * R + r_out[:, None]) * D + d[None, :],
                    (r_out[:, None] < R) & (d[None, :] < D),
                    0.0,
                    eviction_policy="evict_last",
                )
                da += _short_wide_dot(e_rot, tl.trans(v))
                dq = tl.load(
                    DQ_RANK + (s * R + r_out[:, None]) * D + d[None, :],
                    (r_out[:, None] < R) & (d[None, :] < D),
                    0.0,
                    eviction_policy="evict_last",
                )
                c_raw = tl.load(
                    X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D),
                    0,
                ).to(tl.float32)
                if HAS_ROTARY:
                    c_rot = _short_rotate(
                        c_raw, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED, False,
                    )
                else:
                    c_rot = c_raw
                da += _short_wide_dot(c_rot, tl.trans(dq))
        if USE_PDL:
            tlc.gdc_launch_dependents()
        tl.store(
            DX + (b * N + n[:, None]) * (H * (R + D)) + h * R + r_out[None, :],
            da * inv_sqrt,
            (n[:, None] < N) & (r_out[None, :] < R),
        )

    @jit
    def short_bwd_dc(
        X, U_TOK, DQ_RANK, COUNT, COS, SIN, DX,
        N: tl.constexpr, H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        HAS_COUNT: tl.constexpr, HAS_ROTARY: tl.constexpr, ROTARY_BATCHED: tl.constexpr,
        MODE: tl.constexpr, BP: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
        USE_PDL: tl.constexpr = False,
    ):
        if USE_PDL:
            tlc.gdc_wait()
        s = tl.program_id(0)
        channel_tile = tl.program_id(1)
        b = s // H
        h = s % H
        n = tl.arange(0, BP)
        d = channel_tile * BD + tl.arange(0, BD)
        dc_rot = tl.load(
            U_TOK + (s * N + n[:, None]) * D + d[None, :],
            (n[:, None] < N) & (d[None, :] < D),
            0.0,
        )
        if MODE == 1:
            r = tl.arange(0, BR)
            inv_sqrt = (
                _short_rsqrt(tl.load(COUNT + b, eviction_policy="evict_last"))
                if HAS_COUNT
                else tl.math.div_rn(1.0, tl.math.sqrt_rn(float(N)))
            )
            a_bf16 = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
                (n[:, None] < N) & (r[None, :] < R),
                0,
            )
            dq = tl.load(
                DQ_RANK + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0.0,
                eviction_policy="evict_last",
            )
            dc_rot = tl.math.fma(_short_mixed_dot(a_bf16, dq), inv_sqrt, dc_rot)
        if HAS_ROTARY:
            dc = _short_rotate(
                dc_rot, COS, SIN, b, n, channel_tile * BD, N, D, ROTARY_BATCHED, True,
            )
        else:
            dc = dc_rot
        if USE_PDL:
            tlc.gdc_launch_dependents()
        tl.store(
            DX + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
            dc,
            (n[:, None] < N) & (d[None, :] < D),
        )

    return short_stats, short_readout, short_bwd_prep, short_bwd_da, short_bwd_dc


def _short_padded_tokens(length: int) -> int:
    return max(16, ((length + 15) // 16) * 16)


def _short_no_frame_step(
    projected, base, raw, counts, rotary_cos=None, rotary_sin=None,
    *, record: bool = False, res_out: torch.Tensor | None = None,
):
    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    head_dim = (width - heads * rank) // heads
    mode = 2 if base.numel() == 0 else 1
    padded = _short_padded_tokens(length)
    bp = _no_frame_block_size(padded)
    br = _no_frame_block_size(rank)
    bd = min(16 if (bp >= 64 or br >= 64) else 32, _no_frame_block_size(head_dim))
    warps = 8
    is_hopper = _is_hopper_or_newer(projected.device)
    launch_kwargs = {"launch_pdl": True} if is_hopper else {}

    options = dict(device=projected.device, dtype=torch.float32)
    g_tok = torch.empty((batch, heads, padded, padded), **options)
    c_rot = torch.empty((batch, heads, length, head_dim), **options)
    if mode == 1:
        g_rank = torch.empty((batch, heads, rank, rank), **options)
        q_rank = torch.empty((batch, heads, rank, head_dim), **options)
    else:
        g_rank = g_tok
        q_rank = c_rot

    short_stats, short_readout, _, _, _ = _short_kernels()
    short_stats[(batch * heads,)](
        projected, counts, rotary_cos, rotary_sin,
        g_tok, g_rank, q_rank, c_rot,
        length, padded, heads, rank, head_dim,
        counts is not None, rotary_cos is not None,
        rotary_cos is not None and rotary_cos.shape[0] > 1,
        mode, bp, br, bd, is_hopper,
        num_warps=warps, num_stages=1, **launch_kwargs,
    )

    with torch.no_grad():
        t_tok = torch.ops.memsolve_equilibrium.compact_inverse(g_tok)
        del g_tok
        if mode == 1:
            core_state = _launch_compact_core_forward(
                base, raw, record=record, allow_overlap=False,
            )
            coefficient, eta, rank_tape = _no_frame_compact(
                g_rank, q_rank, base, raw, mode=1, record=record,
                core_state=core_state, subtract_eta_in_chain=False,
            )
            core_state = None
        else:
            eta, eta_derivative = _compact_complement(raw, record=record)
            coefficient = None
            rank_tape = (eta_derivative,) if record else ()

    if record:
        residual = (
            torch.empty((batch, heads, length, head_dim), **options)
            if res_out is None
            else res_out
        )
        res_rows = residual.shape[2]
    else:
        residual = c_rot
        res_rows = length

    output = torch.empty(
        (batch, length, heads * head_dim),
        device=projected.device,
        dtype=projected.dtype,
    )
    v_ptr = c_rot if coefficient is None else coefficient
    short_readout[(batch * heads, (head_dim + bd - 1) // bd)](
        projected, c_rot, t_tok, v_ptr, eta, counts, rotary_cos, rotary_sin,
        residual, output,
        length, heads, rank, head_dim, padded, res_rows,
        counts is not None, rotary_cos is not None,
        rotary_cos is not None and rotary_cos.shape[0] > 1,
        mode, record, False, bp, br, bd, is_hopper,
        num_warps=warps, num_stages=1, **launch_kwargs,
    )
    del c_rot
    if not record:
        return output, ()
    return output, (t_tok, residual, coefficient, eta, *rank_tape)


def _run_short_readout_from_saved(
    projected, residual, coefficient, eta, counts, rotary_cos, rotary_sin,
    *, heads: int, rank: int, head_dim: int, mode: int,
) -> torch.Tensor:
    batch, length, _width = projected.shape
    padded = _short_padded_tokens(length)
    bp = _no_frame_block_size(padded)
    br = _no_frame_block_size(rank)
    bd = min(16 if (bp >= 64 or br >= 64) else 32, _no_frame_block_size(head_dim))
    warps = 8
    is_hopper = _is_hopper_or_newer(projected.device)
    launch_kwargs = {"launch_pdl": True} if is_hopper else {}
    output = torch.empty(
        (batch, length, heads * head_dim),
        device=projected.device,
        dtype=projected.dtype,
    )
    v_ptr = residual if coefficient is None else coefficient
    _short_kernels()[1][(batch * heads, (head_dim + bd - 1) // bd)](
        projected, residual, residual, v_ptr, eta, counts, rotary_cos, rotary_sin,
        residual, output,
        length, heads, rank, head_dim, padded, residual.shape[2],
        counts is not None, rotary_cos is not None,
        rotary_cos is not None and rotary_cos.shape[0] > 1,
        mode, False, True, bp, br, bd, is_hopper,
        num_warps=warps, num_stages=1, **launch_kwargs,
    )
    return output


def _short_no_frame_forward(projected, base, raw, counts, rotary_cos=None, rotary_sin=None):
    """Use the smaller token system for the cancellation-sensitive base term.

    When N <= r, C-A(I+A^T A)^-1 A^T C = (I+A A^T)^-1 C.
    The latter avoids subtracting nearly equal values for a strong relation.
    Only at most r-by-r systems are formed; no explicit soft frame is needed.
    """
    return _short_no_frame_step(
        projected, base, raw, counts, rotary_cos, rotary_sin, record=False,
    )[0]


def _short_no_frame_backward(
    projected, counts, rotary_cos, rotary_sin,
    t_tok, residual, coefficient, eta, rank_tape, gradient,
    *, heads: int, rank: int, mode: int,
):
    batch, length, width = projected.shape
    head_dim = (width - heads * rank) // heads
    padded = _short_padded_tokens(length)
    bp = _no_frame_block_size(padded)
    br = _no_frame_block_size(rank)
    bd = min(16 if (bp >= 64 or br >= 64) else 32, _no_frame_block_size(head_dim))
    warps = 8
    is_hopper = _is_hopper_or_newer(projected.device)
    launch_kwargs = {"launch_pdl": True} if is_hopper else {}

    gradient = gradient.contiguous()
    options = dict(device=projected.device, dtype=torch.float32)
    u_tok = torch.empty((batch, heads, length, head_dim), **options)
    sym_p = torch.empty((batch, heads, length, length), **options)
    de_partial = torch.empty((batch, heads), **options)
    if mode == 1:
        e_rot = torch.empty((batch, heads, length, head_dim), **options)
        dv = torch.empty((batch, heads, rank, head_dim), **options)
    else:
        e_rot = u_tok
        dv = u_tok

    _, _, short_bwd_prep, short_bwd_da, short_bwd_dc = _short_kernels()
    short_bwd_prep[(batch * heads,)](
        projected, gradient, t_tok, residual, eta, counts, rotary_cos, rotary_sin,
        u_tok, e_rot, dv, sym_p, de_partial,
        length, heads, rank, head_dim, t_tok.shape[2], residual.shape[2],
        counts is not None, rotary_cos is not None,
        rotary_cos is not None and rotary_cos.shape[0] > 1,
        mode, bp, br, bd, is_hopper,
        num_warps=warps, num_stages=1, **launch_kwargs,
    )
    eta_grad = de_partial.sum(0)
    del de_partial

    with torch.no_grad():
        if mode == 1:
            dg_rank, dq_rank, base_gradient, raw_gradient = _no_frame_compact_backward(
                None, eta, list(rank_tape), dv, eta_grad, mode=1,
                allow_overlap=False, subtract_eta_in_chain=False,
            )
            del dv
        else:
            (eta_derivative,) = rank_tape
            raw_gradient = eta_grad * eta_derivative
            base_gradient = None
            dg_rank = sym_p
            dq_rank = u_tok

    projected_gradient = torch.empty_like(projected)
    v_ptr = u_tok if coefficient is None else coefficient
    rotary_batched = rotary_cos is not None and rotary_cos.shape[0] > 1
    br_out = min(32, br)
    short_bwd_da[(batch * heads, (rank + br_out - 1) // br_out)](
        projected, sym_p, e_rot, v_ptr, dg_rank, dq_rank,
        counts, rotary_cos, rotary_sin, projected_gradient,
        length, heads, rank, head_dim,
        counts is not None, rotary_cos is not None, rotary_batched,
        mode, bp, br, br_out, bd, is_hopper,
        num_warps=warps, num_stages=1, **launch_kwargs,
    )
    short_bwd_dc[(batch * heads, (head_dim + bd - 1) // bd)](
        projected, u_tok, dq_rank,
        counts, rotary_cos, rotary_sin, projected_gradient,
        length, heads, rank, head_dim,
        counts is not None, rotary_cos is not None, rotary_batched,
        mode, bp, br, bd, is_hopper,
        num_warps=warps, num_stages=1, **launch_kwargs,
    )
    del sym_p, u_tok, e_rot, dg_rank, dq_rank
    return projected_gradient, base_gradient, raw_gradient


class _ShortNoFrameMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, base, raw, counts, rotary_cos, rotary_sin):
        output, saved = _short_no_frame_step(
            projected, base, raw, counts, rotary_cos, rotary_sin, record=True,
        )
        heads, rank = base.shape[:2]
        ctx.heads = heads
        ctx.rank = rank
        ctx.mode = 2 if base.numel() == 0 else 1
        t_tok, residual, coefficient, eta, *rank_tape = saved
        ctx.save_for_backward(
            projected, counts, rotary_cos, rotary_sin,
            t_tok, residual, coefficient, eta, *rank_tape,
        )
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        (
            projected, counts, rotary_cos, rotary_sin,
            t_tok, residual, coefficient, eta, *rank_tape,
        ) = ctx.saved_tensors
        ctx.maybe_clear_saved_tensors()
        d_projected, d_base, d_raw = _short_no_frame_backward(
            projected, counts, rotary_cos, rotary_sin,
            t_tok, residual, coefficient, eta, rank_tape, gradient,
            heads=ctx.heads, rank=ctx.rank, mode=ctx.mode,
        )
        return (
            d_projected if ctx.needs_input_grad[0] else None,
            d_base if (ctx.mode == 1 and ctx.needs_input_grad[1]) else None,
            d_raw if ctx.needs_input_grad[2] else None,
            None,
            None,
            None,
        )


def _pack_fast_mix_fwd(
    projected: torch.Tensor,
    base: torch.Tensor,
    raw: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    require_available()
    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    head_dim = (width - heads * rank) // heads
    mode = 2 if base.numel() == 0 else 1
    options = dict(device=projected.device, dtype=torch.float32)

    if length <= rank:
        res_buf = torch.empty((batch, heads, rank, head_dim), **options)
        output, saved = _short_no_frame_step(
            projected, base, raw, counts, rotary_cos, rotary_sin,
            record=True, res_out=res_buf,
        )
        t_tok, _res, coefficient, eta, *rank_tape = saved
        p = t_tok.shape[2]
        if p == rank:
            tok_inv_buf = _ensure_contig(t_tok)
        else:
            tok_inv_buf = torch.empty((batch, heads, rank, rank), **options)
            tok_inv_buf[:, :, :p, :p].copy_(t_tok)
        if mode == 1:
            assert coefficient is not None
            inverse, state, _coords, lu, pivots, mapping, eta_deriv = rank_tape
            coef_buf = _ensure_contig(coefficient)
            inv_buf = _ensure_contig(inverse)
            state_buf = _ensure_contig(state)
            lu_out = _ensure_contig(lu)
            pivots_out = _ensure_contig(pivots)
            mapping_out = _ensure_contig(mapping)
            eta_deriv_out = _ensure_contig(eta_deriv)
        else:
            (eta_deriv,) = rank_tape
            coef_buf = projected.new_empty((0,), dtype=torch.float32)
            inv_buf = projected.new_empty((0,), dtype=torch.float32)
            state_buf = projected.new_empty((0,), dtype=torch.float32)
            lu_out = projected.new_empty((0, 0, 0), dtype=torch.float32)
            pivots_out = projected.new_empty((0, 0), dtype=torch.int32)
            mapping_out = projected.new_empty((0, 0, 0), dtype=torch.float32)
            eta_deriv_out = _ensure_contig(eta_deriv)
        return (
            _ensure_contig(output),
            coef_buf,
            _ensure_contig(eta),
            inv_buf,
            state_buf,
            tok_inv_buf,
            res_buf,
            lu_out,
            pivots_out,
            mapping_out,
            eta_deriv_out,
        )

    output, saved = _no_frame_forward(
        projected, base, raw, counts, record=True, sequence_parallel=None,
        rotary_cos=rotary_cos, rotary_sin=rotary_sin,
    )
    coefficient, eta, inverse, state, _coords, lu, pivots, mapping, eta_deriv = saved
    tok_inv_buf = projected.new_empty((0,), dtype=torch.float32)
    res_buf = projected.new_empty((0,), dtype=torch.float32)
    if mode == 1:
        lu_out = _ensure_contig(lu)
        pivots_out = _ensure_contig(pivots)
        mapping_out = _ensure_contig(mapping)
    else:
        lu_out = projected.new_empty((0, 0, 0), dtype=torch.float32)
        pivots_out = projected.new_empty((0, 0), dtype=torch.int32)
        mapping_out = projected.new_empty((0, 0, 0), dtype=torch.float32)
    return (
        _ensure_contig(output),
        _ensure_contig(coefficient),
        _ensure_contig(eta),
        _ensure_contig(inverse),
        _ensure_contig(state),
        tok_inv_buf,
        res_buf,
        lu_out,
        pivots_out,
        mapping_out,
        _ensure_contig(eta_deriv),
    )


def _fake_fast_mix_tape(
    projected: torch.Tensor,
    base: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    content_width = width - heads * rank
    head_dim = content_width // heads
    mode = 2 if base.shape[2] == 0 else 1
    eta = projected.new_empty((heads,), dtype=torch.float32)
    if length <= rank:
        tok_inv_buf = projected.new_empty((batch, heads, rank, rank), dtype=torch.float32)
        res_buf = projected.new_empty((batch, heads, rank, head_dim), dtype=torch.float32)
        if mode == 1:
            coef_buf = projected.new_empty((batch, heads, rank, head_dim), dtype=torch.float32)
            inv_buf = projected.new_empty((batch, heads, rank, rank), dtype=torch.float32)
            state_buf = projected.new_empty((batch, heads, rank, head_dim), dtype=torch.float32)
        else:
            coef_buf = projected.new_empty((0,), dtype=torch.float32)
            inv_buf = projected.new_empty((0,), dtype=torch.float32)
            state_buf = projected.new_empty((0,), dtype=torch.float32)
    else:
        coef_buf = projected.new_empty((batch, heads, rank, head_dim), dtype=torch.float32)
        inv_buf = projected.new_empty((batch, heads, rank, rank), dtype=torch.float32)
        state_buf = projected.new_empty((batch, heads, rank, head_dim), dtype=torch.float32)
        tok_inv_buf = projected.new_empty((0,), dtype=torch.float32)
        res_buf = projected.new_empty((0,), dtype=torch.float32)
    if mode == 1:
        lu = projected.new_empty((heads, rank, rank), dtype=torch.float32)
        pivots = projected.new_empty((heads, rank), dtype=torch.int32)
        mapping = projected.new_empty((heads, rank, rank), dtype=torch.float32)
    else:
        lu = projected.new_empty((0, 0, 0), dtype=torch.float32)
        pivots = projected.new_empty((0, 0), dtype=torch.int32)
        mapping = projected.new_empty((0, 0, 0), dtype=torch.float32)
    eta_deriv = projected.new_empty((heads,), dtype=torch.float32)
    return (
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    )


def _unpack_fast_mix_bwd(
    grad_mixed: torch.Tensor,
    projected: torch.Tensor,
    base: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    coef_buf: torch.Tensor,
    eta: torch.Tensor,
    inv_buf: torch.Tensor,
    state_buf: torch.Tensor,
    tok_inv_buf: torch.Tensor,
    res_buf: torch.Tensor,
    lu: torch.Tensor,
    pivots: torch.Tensor,
    mapping: torch.Tensor,
    eta_deriv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    head_dim = (width - heads * rank) // heads
    mode = 2 if base.numel() == 0 else 1

    if length <= rank:
        p = _short_padded_tokens(length)
        t_tok = _ensure_contig(tok_inv_buf[:, :, :p, :p])
        if mode == 1:
            rank_tape = [inv_buf, state_buf, base, lu, pivots, mapping, eta_deriv]
            coef = coef_buf
        else:
            rank_tape = [eta_deriv]
            coef = None
        return _short_no_frame_backward(
            projected, counts, rotary_cos, rotary_sin,
            t_tok, res_buf, coef, eta, rank_tape, grad_mixed,
            heads=heads, rank=rank, mode=mode,
        )

    gradient = _ensure_contig(grad_mixed)
    cross_g, eta_g = _no_frame_statistics(
        projected, heads, rank, gradient, counts, rotary_cos, rotary_sin,
    )
    tape = [
        inv_buf,
        state_buf,
        base if mode == 1 else None,
        lu if mode == 1 else None,
        pivots if mode == 1 else None,
        mapping if mode == 1 else None,
        eta_deriv,
    ]
    with torch.no_grad():
        gram_g, cross_grad, base_grad, raw_grad, pending_join = (
            _no_frame_compact_backward(
                None, eta, tape, cross_g, eta_g, mode,
                defer_join=True,
                allow_overlap=_should_overlap_core_stream(batch * length),
            )
        )
    d_proj = _run_token_backward(
        projected, gradient, coef_buf, gram_g, cross_grad, eta,
        counts, rotary_cos, rotary_sin,
        heads=heads, rank=rank, head_dim=head_dim, mode=mode,
    )
    if pending_join is not None:
        torch.cuda.current_stream(_resolved_device_index(d_proj.device)).wait_event(
            pending_join
        )
    return d_proj, base_grad, raw_grad


@torch.library.custom_op("memsolve::fast_mix_infer", mutates_args=())
def _fast_mix_infer_op(
    projected: torch.Tensor,
    base: torch.Tensor,
    raw: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
) -> torch.Tensor:
    require_available()
    if projected.shape[1] <= base.shape[1]:
        return _ensure_contig(
            _short_no_frame_forward(
                projected, base, raw, counts, rotary_cos, rotary_sin,
            )
        )
    return _ensure_contig(
        _no_frame_forward(
            projected, base, raw, counts, record=False, sequence_parallel=None,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
        )[0]
    )


@_fast_mix_infer_op.register_fake
def _(
    projected: torch.Tensor,
    base: torch.Tensor,
    raw: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
) -> torch.Tensor:
    del raw, counts, rotary_cos, rotary_sin
    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    content_width = width - heads * rank
    return projected.new_empty((batch, length, content_width), dtype=projected.dtype)


@torch.library.custom_op("memsolve::fast_mix_fwd", mutates_args=())
def _fast_mix_fwd_op(
    projected: torch.Tensor,
    base: torch.Tensor,
    raw: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _pack_fast_mix_fwd(projected, base, raw, counts, rotary_cos, rotary_sin)


@_fast_mix_fwd_op.register_fake
def _(
    projected: torch.Tensor,
    base: torch.Tensor,
    raw: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del raw, counts, rotary_cos, rotary_sin
    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    content_width = width - heads * rank
    out = projected.new_empty((batch, length, content_width), dtype=projected.dtype)
    return (out, *_fake_fast_mix_tape(projected, base))


@torch.library.custom_op("memsolve::fast_mix_bwd", mutates_args=())
def _fast_mix_bwd_op(
    grad_output: torch.Tensor,
    projected: torch.Tensor,
    base: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    coef_buf: torch.Tensor,
    eta: torch.Tensor,
    inv_buf: torch.Tensor,
    state_buf: torch.Tensor,
    tok_inv_buf: torch.Tensor,
    res_buf: torch.Tensor,
    lu: torch.Tensor,
    pivots: torch.Tensor,
    mapping: torch.Tensor,
    eta_deriv: torch.Tensor,
    need_dproj: bool,
    need_dbase: bool,
    need_draw: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_proj, d_base, d_raw = _unpack_fast_mix_bwd(
        grad_output, projected, base, counts, rotary_cos, rotary_sin,
        coef_buf, eta, inv_buf, state_buf, tok_inv_buf, res_buf,
        lu, pivots, mapping, eta_deriv,
    )
    dp_out = _ensure_contig(d_proj) if need_dproj else projected.new_empty((0,), dtype=projected.dtype)
    db_out = (
        _ensure_contig(d_base)
        if (need_dbase and d_base is not None)
        else base.new_empty((0,), dtype=torch.float32)
    )
    dr_out = _ensure_contig(d_raw) if need_draw else base.new_empty((0,), dtype=torch.float32)
    return dp_out, db_out, dr_out


@_fast_mix_bwd_op.register_fake
def _(
    grad_output: torch.Tensor,
    projected: torch.Tensor,
    base: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    coef_buf: torch.Tensor,
    eta: torch.Tensor,
    inv_buf: torch.Tensor,
    state_buf: torch.Tensor,
    tok_inv_buf: torch.Tensor,
    res_buf: torch.Tensor,
    lu: torch.Tensor,
    pivots: torch.Tensor,
    mapping: torch.Tensor,
    eta_deriv: torch.Tensor,
    need_dproj: bool,
    need_dbase: bool,
    need_draw: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_output, counts, rotary_cos, rotary_sin
    del coef_buf, eta, inv_buf, state_buf, tok_inv_buf, res_buf, lu, pivots, mapping, eta_deriv
    dp = torch.empty_like(projected) if need_dproj else projected.new_empty((0,), dtype=projected.dtype)
    db = (
        torch.empty_like(base)
        if (need_dbase and base.shape[2] > 0)
        else base.new_empty((0,), dtype=torch.float32)
    )
    dr = (
        base.new_empty((base.shape[0],), dtype=torch.float32)
        if need_draw
        else base.new_empty((0,), dtype=torch.float32)
    )
    return dp, db, dr


def _fast_mix_setup_ctx(ctx, inputs, output):
    projected, base, _raw, counts, rotary_cos, rotary_sin = inputs
    (
        _out,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    ) = output
    ctx.save_for_backward(
        projected,
        base,
        counts,
        rotary_cos,
        rotary_sin,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    )
    ctx.has_base = base.shape[2] > 0


def _fast_mix_bwd(ctx, grad_out, *_unused_tape_grads):
    (
        projected,
        base,
        counts,
        rotary_cos,
        rotary_sin,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    ) = ctx.saved_tensors
    need_dp = ctx.needs_input_grad[0]
    need_db = ctx.has_base and ctx.needs_input_grad[1]
    need_dr = ctx.needs_input_grad[2]
    dp, db, dr = _fast_mix_bwd_op(
        grad_out,
        projected,
        base,
        counts,
        rotary_cos,
        rotary_sin,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
        need_dp,
        need_db,
        need_dr,
    )
    return (
        dp if need_dp else None,
        db if need_db else None,
        dr if need_dr else None,
        None,
        None,
        None,
    )


_fast_mix_fwd_op.register_autograd(_fast_mix_bwd, setup_context=_fast_mix_setup_ctx)


@torch.library.custom_op("memsolve::fast_mix_output_linear_fwd", mutates_args=())
def _fast_mix_out_linear_fwd_op(
    projected: torch.Tensor,
    base: torch.Tensor,
    raw: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    flat_out_mask: torch.Tensor | None,
    out_dtype_code: int,
    fuse_weight_bias_grad: bool,
    recompute_code: int,
    save_weight_bf16: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del fuse_weight_bias_grad
    (
        mixed,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    ) = _pack_fast_mix_fwd(projected, base, raw, counts, rotary_cos, rotary_sin)

    batch, length, content_width = mixed.shape
    if recompute_code == 2:
        saved_mixed = mixed.new_empty((batch * 512, content_width))
        if length <= 512:
            tokens = batch * length
            saved_mixed[:tokens].copy_(mixed.reshape(tokens, content_width))
    else:
        saved_mixed = mixed.new_empty((0, content_width))

    output_dtype = _CODE_TO_DTYPE[out_dtype_code]
    weight_bf16 = (
        weight
        if weight.dtype == torch.bfloat16
        else _ensure_contig(weight.detach().to(dtype=torch.bfloat16))
    )
    saved_weight = (
        weight_bf16
        if save_weight_bf16
        else weight.new_empty((0, 0), dtype=torch.bfloat16)
    )
    output = _ensure_contig(
        _linear_forward_output(
            mixed, weight_bf16, bias, output_dtype, flat_out_mask, None,
        )
    )
    del mixed
    return (
        output,
        saved_mixed,
        saved_weight,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    )


@_fast_mix_out_linear_fwd_op.register_fake
def _(
    projected: torch.Tensor,
    base: torch.Tensor,
    raw: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    flat_out_mask: torch.Tensor | None,
    out_dtype_code: int,
    fuse_weight_bias_grad: bool,
    recompute_code: int,
    save_weight_bf16: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del raw, counts, rotary_cos, rotary_sin, bias, flat_out_mask, fuse_weight_bias_grad
    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    content_width = width - heads * rank
    output_dtype = _CODE_TO_DTYPE[out_dtype_code]
    res_dtype = torch.float32 if output_dtype is None else output_dtype
    output = projected.new_empty((batch, length, weight.shape[0]), dtype=res_dtype)
    saved_mixed = projected.new_empty(
        (batch * 512 if recompute_code == 2 else 0, content_width),
        dtype=projected.dtype,
    )
    saved_weight = weight.new_empty(
        (weight.shape[0], weight.shape[1]) if save_weight_bf16 else (0, 0),
        dtype=torch.bfloat16,
    )
    return (
        output,
        saved_mixed,
        saved_weight,
        *_fake_fast_mix_tape(projected, base),
    )


@torch.library.custom_op("memsolve::fast_mix_output_linear_bwd", mutates_args=())
def _fast_mix_out_linear_bwd_op(
    grad_output: torch.Tensor,
    projected: torch.Tensor,
    base: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    weight_bf16: torch.Tensor | None,
    flat_out_mask: torch.Tensor | None,
    saved_mixed: torch.Tensor,
    coef_buf: torch.Tensor,
    eta: torch.Tensor,
    inv_buf: torch.Tensor,
    state_buf: torch.Tensor,
    tok_inv_buf: torch.Tensor,
    res_buf: torch.Tensor,
    lu: torch.Tensor,
    pivots: torch.Tensor,
    mapping: torch.Tensor,
    eta_deriv: torch.Tensor,
    weight_dtype_code: int,
    bias_dtype_code: int,
    fuse_weight_bias_grad: bool,
    recompute_code: int,
    need_dproj: bool,
    need_dbase: bool,
    need_draw: bool,
    need_dw: bool,
    need_db: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, width = projected.shape
    heads, rank = base.shape[:2]
    content_width = width - heads * rank
    head_dim = content_width // heads
    mode = 2 if base.numel() == 0 else 1
    need_dmixed = need_dproj or need_dbase or need_draw

    mixed = None
    if need_dw:
        if recompute_code == 2 and length <= 512:
            tokens = batch * length
            mixed = saved_mixed[:tokens].view(batch, length, content_width)
        elif length <= rank:
            with torch.no_grad():
                mixed = _run_short_readout_from_saved(
                    projected, res_buf, coef_buf if mode == 1 else None, eta,
                    counts, rotary_cos, rotary_sin,
                    heads=heads, rank=rank, head_dim=head_dim, mode=mode,
                )
        else:
            with torch.no_grad():
                mixed = _run_readout(
                    projected, coef_buf, eta, counts, rotary_cos, rotary_sin,
                    heads=heads, rank=rank, head_dim=head_dim, mode=mode,
                )

    proxy = _LinearCtxProxy()
    proxy.value_shape = (batch, length, content_width)
    proxy.value_dtype = projected.dtype
    proxy.weight_dtype = _CODE_TO_DTYPE[weight_dtype_code]
    proxy.out_features = grad_output.shape[2]
    proxy.in_features = content_width
    proxy.bias_dtype = torch.float32 if bias_dtype_code == 0 else _CODE_TO_DTYPE[bias_dtype_code]
    proxy.bias_device = grad_output.device
    proxy.has_bias = bias_dtype_code != 0
    proxy.fuse_weight_bias_grad = fuse_weight_bias_grad
    proxy.needs_input_grad = (need_dmixed, need_dw, need_db)

    grad_mixed, dw, db = _linear_backward_common(
        proxy,
        grad_output,
        mixed,
        weight_bf16 if need_dmixed else None,
        flat_out_mask,
        None,
    )
    del mixed, weight_bf16

    if need_dmixed:
        assert grad_mixed is not None
        d_proj, d_base, d_raw = _unpack_fast_mix_bwd(
            grad_mixed, projected, base, counts, rotary_cos, rotary_sin,
            coef_buf, eta, inv_buf, state_buf, tok_inv_buf, res_buf,
            lu, pivots, mapping, eta_deriv,
        )
    else:
        d_proj = d_base = d_raw = None

    dp_out = (
        _ensure_contig(d_proj)
        if (need_dproj and d_proj is not None)
        else projected.new_empty((0,), dtype=projected.dtype)
    )
    dbase_out = (
        _ensure_contig(d_base)
        if (need_dbase and d_base is not None)
        else base.new_empty((0,), dtype=torch.float32)
    )
    draw_out = (
        _ensure_contig(d_raw)
        if (need_draw and d_raw is not None)
        else base.new_empty((0,), dtype=torch.float32)
    )
    dw_out = (
        _ensure_contig(dw)
        if (need_dw and dw is not None)
        else grad_output.new_empty((0,), dtype=proxy.weight_dtype)
    )
    dbias_out = (
        _ensure_contig(db)
        if (need_db and db is not None)
        else grad_output.new_empty((0,), dtype=proxy.bias_dtype)
    )
    return dp_out, dbase_out, draw_out, dw_out, dbias_out


@_fast_mix_out_linear_bwd_op.register_fake
def _(
    grad_output: torch.Tensor,
    projected: torch.Tensor,
    base: torch.Tensor,
    counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    weight_bf16: torch.Tensor | None,
    flat_out_mask: torch.Tensor | None,
    saved_mixed: torch.Tensor,
    coef_buf: torch.Tensor,
    eta: torch.Tensor,
    inv_buf: torch.Tensor,
    state_buf: torch.Tensor,
    tok_inv_buf: torch.Tensor,
    res_buf: torch.Tensor,
    lu: torch.Tensor,
    pivots: torch.Tensor,
    mapping: torch.Tensor,
    eta_deriv: torch.Tensor,
    weight_dtype_code: int,
    bias_dtype_code: int,
    fuse_weight_bias_grad: bool,
    recompute_code: int,
    need_dproj: bool,
    need_dbase: bool,
    need_draw: bool,
    need_dw: bool,
    need_db: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del counts, rotary_cos, rotary_sin, weight_bf16, flat_out_mask, saved_mixed
    del coef_buf, eta, inv_buf, state_buf, tok_inv_buf, res_buf, lu, pivots, mapping, eta_deriv
    del fuse_weight_bias_grad, recompute_code
    content_width = projected.shape[2] - base.shape[0] * base.shape[1]
    out_features = grad_output.shape[2]
    w_dtype = _CODE_TO_DTYPE[weight_dtype_code]
    b_dtype = torch.float32 if bias_dtype_code == 0 else _CODE_TO_DTYPE[bias_dtype_code]
    dp = torch.empty_like(projected) if need_dproj else projected.new_empty((0,), dtype=projected.dtype)
    dbase = (
        torch.empty_like(base)
        if (need_dbase and base.shape[2] > 0)
        else base.new_empty((0,), dtype=torch.float32)
    )
    draw = (
        base.new_empty((base.shape[0],), dtype=torch.float32)
        if need_draw
        else base.new_empty((0,), dtype=torch.float32)
    )
    dw = (
        grad_output.new_empty((out_features, content_width), dtype=w_dtype)
        if need_dw
        else grad_output.new_empty((0,), dtype=w_dtype)
    )
    dbias = (
        grad_output.new_empty((out_features,), dtype=b_dtype)
        if need_db
        else grad_output.new_empty((0,), dtype=b_dtype)
    )
    return dp, dbase, draw, dw, dbias


def _fast_mix_out_linear_setup_ctx(ctx, inputs, output):
    (
        projected,
        base,
        raw,
        counts,
        rotary_cos,
        rotary_sin,
        weight,
        bias,
        flat_out_mask,
        _out_dtype_code,
        fuse_weight_bias_grad,
        recompute_code,
        save_weight_bf16,
    ) = inputs
    (
        _out,
        saved_mixed,
        saved_weight,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    ) = output
    need_dmixed = projected.requires_grad or base.requires_grad or raw.requires_grad
    saved_w_for_bwd = (
        (weight if weight.dtype == torch.bfloat16 else saved_weight)
        if (need_dmixed and (save_weight_bf16 or weight.dtype == torch.bfloat16))
        else None
    )
    ctx.save_for_backward(
        projected,
        base,
        counts,
        rotary_cos,
        rotary_sin,
        saved_w_for_bwd,
        flat_out_mask,
        saved_mixed,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    )
    ctx.has_base = base.shape[2] > 0
    ctx.weight_dtype_code = _DTYPE_TO_CODE[weight.dtype]
    ctx.bias_dtype_code = 0 if bias is None else _DTYPE_TO_CODE[bias.dtype]
    ctx.bias_device = None if bias is None else bias.device
    ctx.fuse_weight_bias_grad = fuse_weight_bias_grad
    ctx.recompute_code = recompute_code


def _fast_mix_out_linear_bwd(ctx, grad_output, *_unused):
    (
        projected,
        base,
        counts,
        rotary_cos,
        rotary_sin,
        weight_bf16,
        flat_out_mask,
        saved_mixed,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
    ) = ctx.saved_tensors
    need_dp = ctx.needs_input_grad[0]
    need_dbase = ctx.has_base and ctx.needs_input_grad[1]
    need_draw = ctx.needs_input_grad[2]
    need_dw = ctx.needs_input_grad[6]
    need_dbias = (ctx.bias_dtype_code != 0) and ctx.needs_input_grad[7]
    dp, dbase, draw, dw, dbias = _fast_mix_out_linear_bwd_op(
        grad_output,
        projected,
        base,
        counts,
        rotary_cos,
        rotary_sin,
        weight_bf16,
        flat_out_mask,
        saved_mixed,
        coef_buf,
        eta,
        inv_buf,
        state_buf,
        tok_inv_buf,
        res_buf,
        lu,
        pivots,
        mapping,
        eta_deriv,
        ctx.weight_dtype_code,
        ctx.bias_dtype_code,
        ctx.fuse_weight_bias_grad,
        ctx.recompute_code,
        need_dp,
        need_dbase,
        need_draw,
        need_dw,
        need_dbias,
    )
    if need_dbias and ctx.bias_device is not None and dbias.device != ctx.bias_device:
        dbias = dbias.to(device=ctx.bias_device)
    return (
        dp if need_dp else None,
        dbase if need_dbase else None,
        draw if need_draw else None,
        None,
        None,
        None,
        dw if need_dw else None,
        dbias if need_dbias else None,
        None,
        None,
        None,
        None,
        None,
    )


_fast_mix_out_linear_fwd_op.register_autograd(
    _fast_mix_out_linear_bwd, setup_context=_fast_mix_out_linear_setup_ctx
)


def _compiled_fast_mix(
    projected: torch.Tensor,
    core_base_raw: torch.Tensor,
    eta_raw: torch.Tensor,
    valid_counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    *,
    record: bool,
) -> torch.Tensor:
    if not record:
        return _fast_mix_infer_op(
            projected, core_base_raw, eta_raw, valid_counts, rotary_cos, rotary_sin,
        )
    return _fast_mix_fwd_op(
        projected, core_base_raw, eta_raw, valid_counts, rotary_cos, rotary_sin,
    )[0]


def _compiled_fast_mix_output_linear(
    projected: torch.Tensor,
    core_base_raw: torch.Tensor,
    eta_raw: torch.Tensor,
    valid_counts: torch.Tensor | None,
    rotary_cos: torch.Tensor | None,
    rotary_sin: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype | None = None,
    fuse_weight_bias_grad: bool = False,
    output_mask: torch.Tensor | None = None,
    recompute_policy: Literal["auto"] | bool = "auto",
) -> torch.Tensor:
    recompute_policy = _validate_recompute_readout(recompute_policy)
    length = projected.shape[1]
    if recompute_policy == "auto" and has_static_value(length):
        recompute_policy = length > 512
    if recompute_policy is False:
        mixed = _compiled_fast_mix(
            projected,
            core_base_raw,
            eta_raw,
            valid_counts,
            rotary_cos,
            rotary_sin,
            record=torch.is_grad_enabled()
            and (
                projected.requires_grad
                or core_base_raw.requires_grad
                or eta_raw.requires_grad
            ),
        )
        return _TensorCoreLinear.apply(
            mixed,
            weight,
            bias,
            output_dtype,
            fuse_weight_bias_grad,
            output_mask,
            None,
        )
    recompute_code = 1 if recompute_policy is True else 2
    flat_out_mask = None if output_mask is None else output_mask.reshape(-1)
    save_weight_bf16 = bool(
        torch.is_grad_enabled()
        and weight.dtype != torch.bfloat16
        and (
            projected.requires_grad
            or core_base_raw.requires_grad
            or eta_raw.requires_grad
        )
    )
    return _fast_mix_out_linear_fwd_op(
        projected,
        core_base_raw,
        eta_raw,
        valid_counts,
        rotary_cos,
        rotary_sin,
        weight,
        bias,
        flat_out_mask,
        _DTYPE_TO_CODE[output_dtype],
        fuse_weight_bias_grad,
        recompute_code,
        save_weight_bf16,
    )[0]


@torch.library.custom_op("memsolve::compact_cholesky_inverse", mutates_args=())
def _compact_cholesky_inverse_op(gram: torch.Tensor) -> torch.Tensor:
    require_available()
    rank = gram.shape[-1]
    eye = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    shifted = (gram - eye).contiguous()
    with torch.no_grad():
        return torch.ops.memsolve_equilibrium.compact_inverse(shifted.detach()).contiguous()


@_compact_cholesky_inverse_op.register_fake
def _(gram: torch.Tensor) -> torch.Tensor:
    return gram.new_empty(gram.shape, dtype=gram.dtype)


@torch.library.custom_op("memsolve::compact_lu_solve", mutates_args=())
def _compact_lu_solve_op(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    require_available()
    with torch.no_grad():
        lu, pivots = _compact_lu(system.detach().contiguous())
        return _compact_getrs(lu, pivots, rhs.detach().contiguous()).contiguous()


@_compact_lu_solve_op.register_fake
def _(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    del system
    return rhs.new_empty(rhs.shape, dtype=rhs.dtype)


def compile_hopper_kernels_offline(target=None) -> dict[str, object]:
    """Compile all SM90a Hopper-specialized Triton kernels offline and return PTX/cubins."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if target is None:
        target = GPUTarget("cuda", 90, 32)
    gemm_k = _biased_gemm_hopper_kernel()
    stats_k, readout_k, token_bwd_k, chunk_stats_k = _no_frame_kernels()

    specs = {
        "biased_gemm": (
            ASTSource(
                fn=gemm_k,
                signature={
                    "X": "*bf16", "W": "*bf16", "B": "*fp32", "Y": "*bf16",
                    "MASK": "*i1", "INPUT_MASK": "*i1",
                },
                constexprs={
                    "M": 512, "N": 512, "K": 512, "XM": 512, "XK": 1,
                    "WK": 1, "WN": 512, "BS": 1,
                    "BM": 128, "BN": 128, "BK": 64, "GROUP_M": 4,
                    "HAS_BIAS": True, "HAS_MASK": False, "HAS_INPUT_MASK": False,
                    "USE_TMA": True, "W_TRANSPOSED": True, "USE_PDL": True,
                },
            ),
            {"num_warps": 8, "num_stages": 3, "launch_pdl": True},
        ),
        "statistics_wgmma": (
            ASTSource(
                fn=stats_k,
                signature={
                    "X": "*bf16", "E": "*bf16", "COUNT": "*fp32",
                    "G": "*fp32", "Q": "*fp32", "DE": "*fp32",
                    "COS": "*fp32", "SIN": "*fp32",
                    "N_rt": "i32", "R_rt": "i32", "D_rt": "i32",
                    "W_rt": "i32", "HD_rt": "i32",
                },
                constexprs={
                    "N": 196, "H": 8, "R": 32, "D": 64, "T": 2,
                    "GRAD": False, "HAS_COUNT": False, "HAS_ROTARY": True, "ROTARY_BATCHED": False,
                    "BR": 32, "BD": 64, "BT": 128,
                    "SCALE_OUTPUT": False, "USE_TMA": True, "TRANSPOSE_Q": True, "USE_PDL": True,
                },
            ),
            {"num_warps": 4, "num_stages": 3, "launch_pdl": True},
        ),
        "chunk_statistics_wgmma": (
            ASTSource(
                fn=chunk_stats_k,
                signature={
                    "X": "*bf16", "E": "*bf16", "COUNT": "*fp32",
                    "G": "*fp32", "Q": "*fp32", "DE": "*fp32",
                    "COS": "*fp32", "SIN": "*fp32",
                    "N_rt": "i32", "R_rt": "i32", "D_rt": "i32",
                    "W_rt": "i32", "HD_rt": "i32",
                },
                constexprs={
                    "N": 4096, "H": 8, "R": 32, "D": 64, "T": 8,
                    "GRAD": False, "HAS_COUNT": False, "HAS_ROTARY": True, "ROTARY_BATCHED": False,
                    "BR": 32, "BD": 64, "BT": 512,
                    "SCALE_OUTPUT": False, "USE_TMA": True, "TRANSPOSE_Q": True, "USE_PDL": True,
                },
            ),
            {"num_warps": 4, "num_stages": 3, "launch_pdl": True},
        ),
        "readout": (
            ASTSource(
                fn=readout_k,
                signature={
                    "X": "*bf16", "V": "*fp32", "ETA": "*fp32", "COUNT": "*fp32",
                    "Y": "*bf16", "COS": "*fp32", "SIN": "*fp32",
                    "N_rt": "i32", "R_rt": "i32", "D_rt": "i32",
                    "W_rt": "i32", "HD_rt": "i32",
                },
                constexprs={
                    "N": 4096, "H": 8, "R": 32, "D": 64,
                    "HAS_COUNT": False, "HAS_ROTARY": True, "ROTARY_BATCHED": False,
                    "BR": 32, "BD": 64, "BT": 128,
                    "UNSCALED_COMPLEMENT": False, "USE_TMA": True, "USE_PDL": True,
                },
            ),
            {"num_warps": 4, "num_stages": 3, "launch_pdl": True},
        ),
        "token_backward": (
            ASTSource(
                fn=token_bwd_k,
                signature={
                    "X": "*bf16", "E": "*bf16", "V": "*fp32", "DG": "*fp32",
                    "DQ": "*fp32", "ETA": "*fp32", "COUNT": "*fp32",
                    "DX": "*bf16", "COS": "*fp32", "SIN": "*fp32",
                    "N_rt": "i32", "R_rt": "i32", "D_rt": "i32",
                    "W_rt": "i32", "HD_rt": "i32",
                },
                constexprs={
                    "N": 4096, "H": 8, "R": 32, "D": 64,
                    "HAS_COUNT": False, "HAS_ROTARY": True, "ROTARY_BATCHED": False,
                    "BR": 32, "BD": 16, "BT": 128,
                    "UNSCALED_COMPLEMENT": False, "USE_TMA": True, "USE_PDL": True,
                },
            ),
            {"num_warps": 4, "num_stages": 3, "launch_pdl": True},
        ),
        "token_backward_r64": (
            ASTSource(
                fn=token_bwd_k,
                signature={
                    "X": "*bf16", "E": "*bf16", "V": "*fp32", "DG": "*fp32",
                    "DQ": "*fp32", "ETA": "*fp32", "COUNT": "*fp32",
                    "DX": "*bf16", "COS": "*fp32", "SIN": "*fp32",
                    "N_rt": "i32", "R_rt": "i32", "D_rt": "i32",
                    "W_rt": "i32", "HD_rt": "i32",
                },
                constexprs={
                    "N": 4096, "H": 8, "R": 64, "D": 64,
                    "HAS_COUNT": False, "HAS_ROTARY": True, "ROTARY_BATCHED": False,
                    "BR": 64, "BD": 16, "BT": 128,
                    "UNSCALED_COMPLEMENT": False, "USE_TMA": True, "USE_PDL": True,
                },
            ),
            {"num_warps": 8, "num_stages": 3, "launch_pdl": True},
        ),
    }

    compiled: dict[str, object] = {}
    for name, (src, opts) in specs.items():
        compiled[name] = triton.compiler.compile(src, target=target, options=opts)
    return compiled


def _offline_compile_hopper_check(target=None) -> dict[str, object]:
    """Verify SM90a TMA, native WGMMA, PDL, IEEE rounding, and zero cubin spills offline."""
    import shutil
    import subprocess
    import tempfile

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        for candidate in ("/usr/local/cuda-13.2/bin/cuobjdump", "/usr/local/cuda/bin/cuobjdump"):
            if Path(candidate).is_file():
                cuobjdump = candidate
                break

    compiled = compile_hopper_kernels_offline(target)
    for name, cc in compiled.items():
        ptx = cc.asm["ptx"]
        if "cp.async.bulk.tensor" not in ptx:
            raise AssertionError(f"SM90a kernel {name} did not emit TMA cp.async.bulk.tensor")
        if "wgmma.mma_async" not in ptx:
            raise AssertionError(f"SM90a kernel {name} did not emit Hopper wgmma.mma_async")
        if "griddepcontrol.wait" not in ptx or "griddepcontrol.launch_dependents" not in ptx:
            raise AssertionError(f"SM90a kernel {name} did not emit PDL griddepcontrol")
        if "sqrt.approx.f32" in ptx or "div.approx.f32" in ptx:
            raise AssertionError(f"SM90a kernel {name} emitted approximate sqrt/div")
        if ".local" in ptx:
            raise AssertionError(f"SM90a kernel {name} spilled to local memory")
        if cuobjdump is not None and "cubin" in cc.asm:
            with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
                cubin_file.write(cc.asm["cubin"])
                cubin_file.flush()
                usage = subprocess.check_output(
                    [cuobjdump, "--dump-resource-usage", cubin_file.name], text=True
                )
            if "STACK:0" not in usage or "LOCAL:0" not in usage:
                raise AssertionError(f"SM90a kernel {name} spilled in cubin: {usage.strip()}")
    return compiled


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
    require(raw.ndim == 1 and raw.shape[0] == heads, "eta_raw must have shape [H]")
    if counts is not None:
        require(
            counts.device == projected.device
            and counts.dtype == torch.float32
            and counts.is_contiguous(),
            "valid_counts must be contiguous float32 on the same CUDA device",
        )
        require(counts.ndim == 1 and counts.shape[0] == batch, "valid_counts must have shape [B]")


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
    if not torch.compiler.is_compiling() and type(projected) is torch.Tensor:
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
            if (table.ndim != 3 or table.shape[1] != length
                    or table.shape[2] != head_dim // 2
                    or table.shape[0] not in (1, batch)):
                raise ValueError(f"{name} must have shape [1 or B,N,Dhead/2]")
        if rotary_cos.shape != rotary_sin.shape:
            raise ValueError("rotary_cos and rotary_sin must share their shape")
    record = torch.is_grad_enabled() and any(
        value.requires_grad
        for value in (projected, core_base_raw, eta_raw)
    )
    if sequence_parallel is not None:
        sequence_parallel.validate(projected.shape[1], core_base_raw.shape[1])
        if valid_counts is None:
            raise ValueError("sequence parallelism requires global valid_counts")
    if (torch.compiler.is_compiling() or type(projected) is not torch.Tensor) and sequence_parallel is None:
        return _compiled_fast_mix(
            projected, core_base_raw, eta_raw, valid_counts,
            rotary_cos, rotary_sin, record=record,
        )
    arguments = (projected, core_base_raw, eta_raw, valid_counts)
    if sequence_parallel is None and projected.shape[1] <= core_base_raw.shape[1]:
        return (_ShortNoFrameMix.apply(*arguments, rotary_cos, rotary_sin) if record
                else _short_no_frame_forward(*arguments, rotary_cos, rotary_sin))
    if record:
        return _NoFrameMix.apply(*arguments, sequence_parallel, rotary_cos, rotary_sin)
    return _no_frame_forward(*arguments, record=False, sequence_parallel=sequence_parallel,
                             rotary_cos=rotary_cos, rotary_sin=rotary_sin)[0]


_ORIGINAL_FAST_MIX = fast_mix


def _is_default_fast_mix() -> bool:
    return fast_mix is _ORIGINAL_FAST_MIX
