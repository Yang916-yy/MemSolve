from __future__ import annotations

from functools import lru_cache
import importlib
import os
from pathlib import Path
from threading import Lock

import torch

_LOAD_LOCK = Lock()
_SUPPORTED_ARCHITECTURES = frozenset((80, 86, 87, 89, 90, 100, 120))
_NATIVE_CONTRACT_VERSION = 11
_RUNTIME_PACKAGE = "lsso_cuda_runtime"
_LOADED_ARCHITECTURE: int | None = None


def _native_operator_abi_is_registered() -> bool:
    namespace = getattr(torch.ops, "lsso_equilibrium", None)
    return (
        namespace is not None
        and hasattr(namespace, "forward_inference")
        and hasattr(namespace, "forward_train")
        and hasattr(namespace, "backward")
    )


def _native_contract_version() -> int | None:
    """Return the registered native contract version, if the query exists."""

    namespace = getattr(torch.ops, "lsso_equilibrium", None)
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
            "the loaded LSSO CUDA extension does not expose the required native "
            f"contract version {_NATIVE_CONTRACT_VERSION}; rebuild it with "
            "tools/build_cuda.sh and restart the process"
        )
    if version != _NATIVE_CONTRACT_VERSION:
        raise RuntimeError(
            "the loaded LSSO CUDA extension has native contract version "
            f"{version}, but this package requires {_NATIVE_CONTRACT_VERSION}; "
            "rebuild it with tools/build_cuda.sh and restart the process"
        )


def is_available() -> bool:
    """Return whether the native Dynamic/Static/Zero operator is registered."""

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
                "the LSSO accretive-equilibrium CUDA extension is not loaded; "
                "install the matching precompiled runtime wheel or build it with "
                "tools/build_cuda.sh, then call lsso.ball.cuda.load(); "
                f"native contract version {_NATIVE_CONTRACT_VERSION} is required"
            )


def _device_architecture(device: torch.device | int | None = None) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the LSSO CUDA fast path requires an available CUDA device")

    if device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        resolved_device = torch.device("cuda", device)
    else:
        resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError(
            "the LSSO CUDA fast path requires a CUDA device, " f"got {resolved_device}"
        )
    if resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())

    major, minor = torch.cuda.get_device_capability(resolved_device)
    architecture = major * 10 + minor
    if architecture == 121:
        architecture = 120
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            "the LSSO CUDA fast path supports SM80, SM86, SM87, SM89, "
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
            "the installed LSSO CUDA runtime wheel could not be imported"
        ) from error

    from lsso import __version__ as package_version

    expected = {
        "LSSO_VERSION": package_version,
        "NATIVE_CONTRACT_VERSION": _NATIVE_CONTRACT_VERSION,
        "TORCH_VERSION": torch.__version__,
        "CUDA_VERSION": torch.version.cuda or "",
        "CXX11_ABI": int(torch.compiled_with_cxx11_abi()),
    }
    for name, value in expected.items():
        if getattr(runtime, name, None) != value:
            raise RuntimeError(
                "the installed LSSO CUDA runtime wheel is incompatible: "
                f"{name} is {getattr(runtime, name, None)!r}, expected {value!r}"
            )
    architectures = getattr(runtime, "ARCHITECTURES", ())
    if architecture not in architectures:
        raise RuntimeError(
            "the installed LSSO CUDA runtime wheel does not contain "
            f"lsso_equilibrium_sm{architecture}.so"
        )
    library_path = getattr(runtime, "library_path", None)
    if not callable(library_path):
        raise RuntimeError(
            "the installed LSSO CUDA runtime wheel does not expose library_path()"
        )
    return Path(library_path(architecture))


def _development_library_path(architecture: int) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    return (
        repository_root
        / "build"
        / "cuda"
        / "lib"
        / f"lsso_equilibrium_sm{architecture}.so"
    )


def _default_library_path(device: torch.device | int | None = None) -> Path:
    override = os.environ.get("LSSO_CUDA_LIBRARY")
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
    """Load one explicitly built strict CUDA extension."""

    global _LOADED_ARCHITECTURE
    requested_architecture = _device_architecture(device)
    with _LOAD_LOCK:
        if is_available():
            if (
                _LOADED_ARCHITECTURE is not None
                and _LOADED_ARCHITECTURE != requested_architecture
            ):
                raise RuntimeError(
                    "the loaded LSSO CUDA extension targets "
                    f"SM{_LOADED_ARCHITECTURE}, not requested SM{requested_architecture}"
                )
            return
        if _native_operator_abi_is_registered():
            _check_native_contract()

        library = Path(path) if path is not None else _default_library_path(device)
        library = library.expanduser().resolve()
        expected_name = f"lsso_equilibrium_sm{requested_architecture}.so"
        if library.name != expected_name:
            raise RuntimeError(
                "the strict LSSO CUDA fast path requires "
                f"{expected_name} for SM{requested_architecture}, got {library.name}"
            )
        if not library.is_file():
            raise RuntimeError(
                "the LSSO CUDA extension was not found at "
                f"{library}; build it with tools/build_cuda.sh or pass its path to load()"
            )

        torch.ops.load_library(str(library))
        _check_native_contract()
        _LOADED_ARCHITECTURE = requested_architecture


@lru_cache(maxsize=1)
def _no_frame_kernels():
    # Lazily import the Triton supplied by CUDA PyTorch, as for projections.
    from triton import jit
    import triton.language as tl

    @jit
    def _mixed_dot(a, b):
        # The token operand is already BF16. Preserve the compact FP32
        # coefficient with two BF16 words (Henry et al., arXiv:1904.06376).
        # One BF16 rounding fails the scale-64 forward/gradient envelope.
        hi = b.to(tl.bfloat16)
        lo = (b - hi.to(tl.float32)).to(tl.bfloat16)
        return tl.dot(a, hi) + tl.dot(a, lo)

    @jit
    def _statistics(
        X,
        E,
        COUNT,
        G,
        Q,
        DE,
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
                q = tl.dot(tl.trans(a), e).to(tl.float32) / tl.sqrt(count)
                de += tl.sum(tl.sum(e.to(tl.float32) * c.to(tl.float32), 0), 0)
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
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        HAS_COUNT: tl.constexpr,
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
            y = eta * c + _mixed_dot(a, v) / tl.sqrt(count)
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
        N: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        HAS_COUNT: tl.constexpr,
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
        g = tl.load(
            DG + (s * R + r[:, None]) * R + r[None, :],
            (r[:, None] < R) & (r[None, :] < R),
            0,
        )
        g = (g + tl.trans(g)).to(tl.float32)
        inv = 1.0 / tl.sqrt(count)
        eta = tl.load(ETA + h)
        da = _mixed_dot(a, g) * (inv * inv)
        for channel_tile in range(tl.cdiv(D, BD)):
            d = channel_tile * BD + tl.arange(0, BD)
            c = tl.load(
                X + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D),
                0,
            )
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
            dq = tl.load(
                DQ + (s * R + r[:, None]) * D + d[None, :],
                (r[:, None] < R) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            da += (_mixed_dot(e, tl.trans(v)) + _mixed_dot(c, tl.trans(dq))) * inv
            dc = eta * e.to(tl.float32) + _mixed_dot(a, dq) * inv
            tl.store(
                DX + (b * N + n[:, None]) * (H * (R + D)) + H * R + h * D + d[None, :],
                dc, (n[:, None] < N) & (d[None, :] < D),
            )
        tl.store(
            DX + (b * N + n[:, None]) * (H * (R + D)) + h * R + r[None, :],
            da, (n[:, None] < N) & (r[None, :] < R),
        )

    return _statistics, _readout, _token_backward


def _no_frame_block_size(size: int) -> int:
    return max(16, 1 << (size - 1).bit_length())


def _no_frame_statistics(projected, heads, rank, gradient=None, counts=None):
    batch, length, width = projected.shape
    head_dim = (width - heads * rank) // heads
    tiles = (length + 127) // 128
    options = dict(device=projected.device, dtype=torch.float32)
    cross = torch.empty((batch * heads, tiles, rank, head_dim), **options)
    if gradient is None:
        gram = torch.empty((batch * heads, tiles, rank, rank), **options)
        eta_gradient = cross
    else:
        gram = cross
        eta_gradient = torch.empty((batch * heads, tiles), **options)
    _no_frame_kernels()[0][(batch * heads, tiles)](
        projected,
        gradient,
        counts,
        gram,
        cross,
        eta_gradient,
        length,
        heads,
        rank,
        head_dim,
        tiles,
        gradient is not None,
        counts is not None,
        _no_frame_block_size(rank),
        min(128, _no_frame_block_size(head_dim)),
        128,
        num_warps=4,
        num_stages=1,
    )
    cross = cross.sum(1).view(batch, heads, rank, head_dim)
    if gradient is None:
        return gram.sum(1).view(batch, heads, rank, rank), cross
    return cross, eta_gradient.sum(1).view(batch, heads).sum(0)


@lru_cache(maxsize=1)
def _compact_kernels():
    from triton import jit
    import triton.language as tl

    @jit
    def prepare(BASE, UPDATE, COUNT, COORD, FACTOR, SIZE: tl.constexpr,
                R: tl.constexpr, H: tl.constexpr, DYNAMIC: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        local = i % (R * R)
        row, col = local // R, local % R
        h = (i // (R * R)) % H
        value = tl.load(BASE + h * R * R + local, i < SIZE, 0)
        if DYNAMIC:
            count = tl.load(COUNT + i // (H * R * R), i < SIZE, 1)
            value += tl.load(UPDATE + i, i < SIZE, 0) / tl.sqrt(count)
        shifted = value + 0.5413248546129181
        diagonal = tl.where(shifted > 20.0, shifted, tl.log(1.0 + tl.exp(shifted)))
        f = tl.where(row > col, value, tl.where(row == col, diagonal, 0.0))
        tl.store(COORD + i, value, i < SIZE)
        tl.store(FACTOR + i, f, i < SIZE)

    @jit
    def matrix(A, B, COORD, COUNT, OUT, AUX, SIZE: tl.constexpr,
               R: tl.constexpr, H: tl.constexpr, OP: tl.constexpr,
               DYNAMIC: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        local = i % (R * R)
        row, col = local // R, local % R
        offset = i - local
        transposed = offset + col * R + row
        a = tl.load(A + i, i < SIZE, 0)
        if OP == 0:  # K=F F^T+Omega and both I +/- K in one pass.
            t = tl.load(COORD + i, i < SIZE, 0)
            tt = tl.load(COORD + transposed, i < SIZE, 0)
            k = a + tl.where(row < col, t, tl.where(row > col, -tt, 0.0))
            tl.store(OUT + i, (row == col).to(tl.float32) + k, i < SIZE)
            tl.store(AUX + i, (row == col).to(tl.float32) - k, i < SIZE)
        elif OP == 1:  # Cholesky VJP: half the symmetric copy of the lower triangle.
            at = tl.load(A + transposed, i < SIZE, 0)
            tl.store(OUT + i, 0.5 * tl.where(row >= col, a, at), i < SIZE)
        else:  # Generator VJP plus sample normalization.
            g = tl.load(B + i, i < SIZE, 0)
            gt = tl.load(B + transposed, i < SIZE, 0)
            coordinate = tl.load(COORD + i, i < SIZE, 0)
            shifted = coordinate + 0.5413248546129181
            sigmoid = 1.0 / (1.0 + tl.exp(-shifted))
            d = tl.where(row > col, a, tl.where(row == col, a * sigmoid, g - gt))
            tl.store(OUT + i, d, i < SIZE)
            if DYNAMIC:
                count = tl.load(COUNT + i // (H * R * R), i < SIZE, 1)
                tl.store(AUX + i, d / tl.sqrt(count), i < SIZE)

    @jit
    def complement(RAW, ETA, DERIVATIVE, H: tl.constexpr, BLOCK: tl.constexpr):
        h = tl.arange(0, BLOCK)
        raw = tl.load(RAW + h, h < H, 0)
        exponent = tl.exp(-2.0 * tl.abs(raw))
        scale: tl.constexpr = 1.0 - 1.1920928955078125e-7
        value = scale * tl.where(raw >= 0.0, 1.0 - exponent, exponent - 1.0) / (1.0 + exponent)
        derivative = scale * 4.0 * exponent / ((1.0 + exponent) * (1.0 + exponent))
        tl.store(ETA + h, value, h < H)
        tl.store(DERIVATIVE + h, derivative, h < H)

    @jit
    def combine(DELTA, STATE, ETA, OUT, SIZE: tl.constexpr, H: tl.constexpr,
                RD: tl.constexpr, ZERO: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        z = tl.load(STATE + i, i < SIZE, 0)
        eta = tl.load(ETA + (i // RD) % H, i < SIZE, 0)
        delta = 0.0 if ZERO else tl.load(DELTA + i, i < SIZE, 0)
        tl.store(OUT + i, delta - eta * z, i < SIZE)

    return prepare, matrix, complement, combine


def _compact_prepare(base, update, counts):
    coordinates = torch.empty_like(base if update is None else update)
    core_factor = torch.empty_like(coordinates)
    size = coordinates.numel()
    _compact_kernels()[0][((size + 255) // 256,)](
        base, update, counts, coordinates, core_factor, size, base.shape[-1],
        base.shape[0], update is not None, 256, num_warps=4,
    )
    return coordinates, core_factor


def _compact_matrix(a, coordinates=None, b=None, counts=None, *, heads=1, op=0):
    output = torch.empty_like(a, memory_format=torch.contiguous_format)
    auxiliary = torch.empty_like(output) if op == 0 or counts is not None else output
    # GEMM outputs are contiguous; the Cholesky product is normalized here too.
    a = a.contiguous()
    _compact_kernels()[1][((a.numel() + 255) // 256,)](
        a, b, coordinates, counts, output, auxiliary, a.numel(), a.shape[-1],
        heads, op, counts is not None, 256, num_warps=4,
    )
    return output, auxiliary


def _compact_complement(raw):
    eta, derivative = torch.empty_like(raw), torch.empty_like(raw)
    _compact_kernels()[2][(1,)](raw, eta, derivative, raw.numel(), _no_frame_block_size(raw.numel()), num_warps=4)
    return eta, derivative


def _compact_combine(correction, state, eta, mode):
    output = torch.empty_like(state, memory_format=torch.contiguous_format)
    state = state.contiguous()
    if correction is not None:
        correction = correction.contiguous()
    _compact_kernels()[3][((state.numel() + 255) // 256,)](
        correction, state, eta, output, state.numel(), state.shape[1],
        state.shape[-2] * state.shape[-1], mode == 2, 256, num_warps=4,
    )
    return output


def _compact_lu(system):
    return torch.ops.lsso_equilibrium.compact_lu(system.contiguous())[:2]


def _compact_getrs(lu, pivots, rhs, *, transpose=False):
    return torch.ops.lsso_equilibrium.compact_getrs(
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


def _no_frame_compact(gram, cross, base, drive, raw, counts, mode):
    # reference.py owns these equations. This boundary records values, not an
    # autograd graph: R^T R=I+A^T A, Z=R^-T A^T C, V=R^-1(Delta-eta Z).
    rank = gram.shape[-1]
    identity = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    factor, _info = torch.linalg.cholesky_ex(gram + identity, check_errors=False)
    state = torch.linalg.solve_triangular(factor, cross, upper=False)
    eta, eta_derivative = _compact_complement(raw)
    coordinates = core_factor = lu = pivots = mapping = None
    if mode == 2:
        correction = None
    else:
        coordinates, core_factor = _compact_prepare(base, None if mode == 1 else state @ drive, counts)
        system, difference = _compact_matrix(core_factor @ core_factor.mT, coordinates)
        lu, pivots = _compact_lu(system)
        if mode == 1:
            mapping = _compact_getrs(lu, pivots, difference)
            correction = mapping @ state
        else:
            correction = _compact_getrs(lu, pivots, difference @ state)
    coefficient = torch.linalg.solve_triangular(
        factor.mT, _compact_combine(correction, state, eta, mode), upper=True,
    ).contiguous()
    tape = (factor, state, coordinates, lu, pivots, mapping if mode == 1 else correction, drive, eta_derivative, counts, core_factor)
    return coefficient, eta, tape


def _no_frame_compact_backward(coefficient, eta, tape, gradient, eta_gradient, mode):
    factor, state, coordinates, lu, pivots, correction, drive, eta_derivative, counts, core_factor = tape
    # V=L^-T(Delta-eta Z): E_delta=L^-1 E_v, E_L=-V E_delta^T.
    delta_gradient = torch.linalg.solve_triangular(factor, gradient, upper=False)
    factor_gradient = -(coefficient @ delta_gradient.mT)
    state_gradient = -eta[None, :, None, None] * delta_gradient
    eta_gradient = eta_gradient - (delta_gradient * state).sum((0, 2, 3))
    base_gradient = drive_gradient = None
    if mode != 2:
        if mode == 1:
            # Accumulate a single per-head map adjoint before solving.
            map_gradient = (delta_gradient @ state.mT).sum(0)
            adjoint = _compact_getrs(lu, pivots, map_gradient, transpose=True)
            identity = torch.eye(state.shape[-2], device=state.device, dtype=state.dtype)
            generator_gradient = -(adjoint @ (correction + identity).mT)
            del map_gradient, identity
            state_gradient = state_gradient + correction.mT @ delta_gradient
        else:
            # E_K=-U(Delta+Z)^T and E_Z=2U-E_delta. Keep the K(Z) chain below.
            adjoint = _compact_getrs(lu, pivots, delta_gradient, transpose=True)
            generator_gradient = -(adjoint @ (correction + state).mT)
            state_gradient = state_gradient + 2 * adjoint - delta_gradient
        core_gradient = (generator_gradient + generator_gradient.mT) @ core_factor
        coordinate_gradient, scaled = _compact_matrix(
            core_gradient, coordinates, generator_gradient,
            counts if mode == 0 else None, heads=state.shape[1], op=2,
        )
        del core_gradient, generator_gradient, adjoint
        if mode == 1:
            base_gradient = coordinate_gradient
        else:
            base_gradient = coordinate_gradient.sum(0)
            drive_gradient = (state.mT @ scaled).sum(0)
            state_gradient = state_gradient + scaled @ drive.mT
        del coordinate_gradient, scaled
    # Z=L^-1 Q, followed by the Cholesky adjoint for G+I=L L^T.
    cross_gradient = torch.linalg.solve_triangular(factor.mT, state_gradient, upper=True)
    factor_gradient.sub_(cross_gradient @ state.mT)
    product = factor.mT @ factor_gradient
    del factor_gradient, state_gradient, delta_gradient
    symmetric, _ = _compact_matrix(product, op=1)
    del product
    intermediate = torch.linalg.solve_triangular(factor.mT, symmetric, upper=True)
    gram_gradient = torch.linalg.solve_triangular(factor.mT, intermediate.mT, upper=True).mT
    raw_gradient = eta_gradient * eta_derivative
    return gram_gradient, cross_gradient, base_gradient, drive_gradient, raw_gradient


def _no_frame_forward(projected, base, drive, raw, counts, *, record):
    from .reference import _ieee_fp32_matmul

    heads, rank = base.shape[:2]
    batch, length, width = projected.shape
    head_dim = (width - heads * rank) // heads
    mode = 2 if base.numel() == 0 else (1 if drive.numel() == 0 else 0)
    gram, cross = _no_frame_statistics(projected, heads, rank, counts=counts)
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
    # All compact operations remain FP32 regardless of the caller's AMP/TF32
    # policy. No process-wide precision setting is left changed.
    with (
        torch.no_grad(),
        torch.autocast(device_type="cuda", enabled=False),
        _ieee_fp32_matmul(projected.device),
    ):
        coefficient, eta, tape = _no_frame_compact(
            gram, cross, base, drive, raw, normalization, mode
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
        length,
        heads,
        rank,
        head_dim,
        counts is not None,
        _no_frame_block_size(rank),
        min(128, _no_frame_block_size(head_dim)),
        128,
        num_warps=4,
        num_stages=1,
    )
    return output, (coefficient, eta, *tape)


class _NoFrameMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, base, drive, raw, counts):
        output, saved = _no_frame_forward(
            projected, base, drive, raw, counts, record=True
        )
        ctx.save_for_backward(projected, counts, *saved)
        ctx.heads, ctx.rank = base.shape[:2]
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        from .reference import _ieee_fp32_matmul

        projected, counts, coefficient, eta, *tape = ctx.saved_tensors
        heads, rank = ctx.heads, ctx.rank
        batch, length, width = projected.shape
        head_dim = (width - heads * rank) // heads
        gradient = gradient.contiguous()
        coefficient_gradient, eta_gradient = _no_frame_statistics(
            projected,
            heads,
            rank,
            gradient,
            counts,
        )
        mode = 2 if tape[2] is None else (1 if tape[6].numel() == 0 else 0)
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", enabled=False),
            _ieee_fp32_matmul(projected.device),
        ):
            gram_gradient, cross_gradient, base_gradient, drive_gradient, raw_gradient = _no_frame_compact_backward(
                coefficient, eta, tape, coefficient_gradient, eta_gradient, mode
            )
        projected_gradient = torch.empty_like(projected)
        # Bound live token adjoints independently of the statistics tile.
        # Larger fused tiles can spill registers at rank 48/64 and head_dim 64.
        _no_frame_kernels()[2][(batch * heads, (length + 31) // 32)](
            projected,
            gradient,
            coefficient,
            gram_gradient.contiguous(),
            cross_gradient.contiguous(),
            eta,
            counts,
            projected_gradient,
            length,
            heads,
            rank,
            head_dim,
            counts is not None,
            _no_frame_block_size(rank),
            min(128, _no_frame_block_size(head_dim)),
            32,
            num_warps=4,
            num_stages=1,
        )
        return projected_gradient, base_gradient, drive_gradient, raw_gradient, None


def _short_no_frame_forward(projected, base, drive, raw, counts):
    """Use the smaller token system for the cancellation-sensitive base term.

    When N <= r, C-A(I+A^T A)^-1 A^T C = (I+A A^T)^-1 C.
    The latter avoids subtracting nearly equal values for a strong relation.
    Only at most r-by-r systems are formed; no explicit soft frame is needed.
    """
    from .reference import accretive_generator, bounded_complement, _ieee_fp32_matmul

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
            coordinates = (
                base
                if not drive.numel()
                else base + state @ drive / normalization.sqrt().view(batch, 1, 1, 1)
            )
            generator = accretive_generator(coordinates)
            if drive.numel():
                correction = _NoFrameSolve.apply(
                    identity + generator, (identity - generator) @ state
                )
            else:
                correction = (
                    _NoFrameSolve.apply(identity + generator, identity - generator)
                    @ state
                )
            coefficient = torch.linalg.solve_triangular(
                factor.mT, correction, upper=True
            )
            result = result + relation @ coefficient
        return (
            result.transpose(1, 2)
            .reshape(batch, length, heads * head_dim)
            .to(torch.bfloat16)
        )


class _ShortNoFrameMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, base, drive, raw, counts):
        with torch.enable_grad():
            leaves = [
                value.detach().requires_grad_(True)
                for value in (projected, base, drive, raw)
            ]
            result = _short_no_frame_forward(*leaves, counts)
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
        return (*gradients, None)


def _validate_no_frame_inputs(projected, base, drive, raw, counts):
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
        ("core_drive_weight", drive),
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
    require(base.numel() != 0 or drive.numel() == 0, "Zero cannot have a dynamic drive")
    require(
        tuple(drive.shape)
        == (heads, content_dim // heads, rank if drive.numel() else 0),
        "core_drive_weight has an incompatible shape",
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


class _NativeFrameMix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        projected: torch.Tensor,
        core_base_raw: torch.Tensor,
        core_drive_weight: torch.Tensor,
        eta_raw: torch.Tensor,
        valid_counts: torch.Tensor | None,
    ) -> torch.Tensor:
        output, tape, pivots = torch.ops.lsso_equilibrium.forward_train(
            projected,
            core_base_raw,
            core_drive_weight,
            eta_raw,
            valid_counts,
        )
        saved = [
            projected,
            core_base_raw,
            core_drive_weight,
            eta_raw,
            tape,
            pivots,
        ]
        if valid_counts is not None:
            saved.append(valid_counts)
        ctx.save_for_backward(*saved)
        ctx.has_valid_counts = valid_counts is not None
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        saved = ctx.saved_tensors
        projected, core_base_raw, core_drive_weight, eta_raw, tape, pivots = saved[:6]
        index = 6
        valid_counts = saved[index] if ctx.has_valid_counts else None

        gradients = torch.ops.lsso_equilibrium.backward(
            grad_output.contiguous(),
            projected,
            core_base_raw,
            core_drive_weight,
            eta_raw,
            tape,
            pivots,
            valid_counts,
        )
        return (*gradients, None)


def fast_mix(
    projected: torch.Tensor,
    core_base_raw: torch.Tensor,
    core_drive_weight: torch.Tensor,
    eta_raw: torch.Tensor,
    valid_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the common no-frame operator; never dispatch to a materialized P.

    The only dimension choice is the primal/dual identity for the base term:
    solve an N-by-N token system when N <= r, otherwise an r-by-r system.
    Rank, core mode and batch size do not select a different CUDA algorithm.
    """
    if valid_counts is not None and valid_counts.requires_grad:
        raise ValueError("the LSSO CUDA fast path does not support gradients for valid_counts")
    require_available()
    _validate_no_frame_inputs(
        projected, core_base_raw, core_drive_weight, eta_raw, valid_counts
    )
    record = torch.is_grad_enabled() and any(
        value.requires_grad
        for value in (projected, core_base_raw, core_drive_weight, eta_raw)
    )
    arguments = (projected, core_base_raw, core_drive_weight, eta_raw, valid_counts)
    if projected.shape[1] <= core_base_raw.shape[1]:
        return _ShortNoFrameMix.apply(*arguments) if record else _short_no_frame_forward(*arguments)
    if record:
        return _NoFrameMix.apply(*arguments)
    return _no_frame_forward(*arguments, record=False)[0]
