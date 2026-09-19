from __future__ import annotations

import importlib
import os
from pathlib import Path
from threading import Lock

import torch


_LOAD_LOCK = Lock()
_SUPPORTED_ARCHITECTURES = frozenset((80, 86, 87, 89, 90, 100, 120))
_NATIVE_CONTRACT_VERSION = 9
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
            "the LSSO CUDA fast path requires a CUDA device, "
            f"got {resolved_device}"
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


class _FastMix(torch.autograd.Function):
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
    """Run the strict native mixer with first-order autograd support."""

    if valid_counts is not None and valid_counts.requires_grad:
        raise ValueError(
            "the LSSO CUDA fast path does not support gradients for valid_counts"
        )

    require_available()
    if not torch.is_grad_enabled() or not any(
        value.requires_grad
        for value in (projected, core_base_raw, core_drive_weight, eta_raw)
    ):
        return torch.ops.lsso_equilibrium.forward_inference(
            projected,
            core_base_raw,
            core_drive_weight,
            eta_raw,
            valid_counts,
        )
    return _FastMix.apply(
        projected,
        core_base_raw,
        core_drive_weight,
        eta_raw,
        valid_counts,
    )


__all__ = ["fast_mix", "is_available", "load", "require_available"]
