#!/usr/bin/env python3
"""Package the current CUDA artifacts as one installable runtime wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DIST_NAME = "memsolve_cuda_runtime"
PACKAGE_NAME = "memsolve_cuda_runtime"
SUPPORTED_ARCHITECTURES = (80, 86, 87, 89, 90, 100, 120)


def _default_platform_tag() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "linux_x86_64"
    if machine in ("aarch64", "arm64"):
        return "linux_aarch64"
    raise RuntimeError(f"unsupported Linux CPU architecture for wheel packaging: {machine!r}")


PLATFORM_TAG = _default_platform_tag()


def _normalized(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value)


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_glibc_version(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", value)
    if match is None:
        raise ValueError(f"invalid GLIBC version {value!r}; expected major.minor")
    return int(match.group(1)), int(match.group(2))


def _parse_architectures(value: str) -> tuple[int, ...]:
    items = [part.strip().rstrip("af") for part in re.split(r"[;,]", value) if part.strip()]
    if not items:
        raise ValueError("architectures list must not be empty")
    parsed: list[int] = []
    for item in items:
        arch = int(item)
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"unsupported architecture {arch}; expected one of {SUPPORTED_ARCHITECTURES}")
        if arch not in parsed:
            parsed.append(arch)
    return tuple(parsed)


def _run(*command: str) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def _strip_and_validate_library(
    library: Path,
    *,
    max_glibc: tuple[int, int],
) -> dict[str, str]:
    if not library.is_file():
        raise FileNotFoundError(f"missing CUDA artifact {library}")
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        raise RuntimeError("patchelf is required to package a CUDA runtime wheel")
    _run(patchelf, "--remove-rpath", str(library))

    dynamic = _run("readelf", "-d", str(library))
    if "RPATH" in dynamic or "RUNPATH" in dynamic:
        raise RuntimeError(f"{library} retains a build-host RPATH or RUNPATH")
    version_info = _run("readelf", "--version-info", str(library))
    glibc_versions = {
        (int(major), int(minor))
        for major, minor in re.findall(r"GLIBC_([0-9]+)\.([0-9]+)", version_info)
    }
    required_glibc = max(glibc_versions, default=(0, 0))
    if required_glibc > max_glibc:
        raise RuntimeError(
            f"{library} requires GLIBC_{required_glibc[0]}.{required_glibc[1]}, "
            f"above the release limit GLIBC_{max_glibc[0]}.{max_glibc[1]}"
        )
    return {
        "filename": library.name,
        "sha256": _sha256(library),
        "glibc_max": f"{required_glibc[0]}.{required_glibc[1]}",
    }


def _torch_contract(python: Path) -> dict[str, Any]:
    probe = (
        "import json, torch; "
        "print(json.dumps({'version': torch.__version__, "
        "'cuda_version': torch.version.cuda or '', "
        "'cxx11_abi': int(torch.compiled_with_cxx11_abi())}))"
    )
    # Torch may emit non-fatal initialization warnings on stderr. Keep the
    # machine-readable contract probe isolated on stdout.
    output = subprocess.check_output((str(python), "-c", probe), text=True)
    return json.loads(output)


def _runtime_init(
    *,
    memsolve_version: str,
    native_contract_version: int,
    torch_version: str,
    cuda_version: str,
    cxx11_abi: int,
    architectures: tuple[int, ...],
) -> bytes:
    return f'''"""Generated CUDA runtime for MemSolve."""

from pathlib import Path

MEMSOLVE_VERSION = {memsolve_version!r}
NATIVE_CONTRACT_VERSION = {native_contract_version}
TORCH_VERSION = {torch_version!r}
CUDA_VERSION = {cuda_version!r}
CXX11_ABI = {cxx11_abi}
ARCHITECTURES = {architectures!r}


def library_path(architecture: int) -> Path:
    lib_dir = Path(__file__).with_name("lib")
    unified = lib_dir / "memsolve_equilibrium.so"
    if unified.is_file():
        return unified
    return lib_dir / f"memsolve_equilibrium_sm{{architecture}}.so"
'''.encode("utf-8")


def _write_wheel(
    wheel_path: Path,
    *,
    files: dict[str, bytes],
) -> None:
    records: list[list[str]] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, data in files.items():
            wheel.writestr(name, data)
            records.append([name, _record_digest(data), str(len(data))])
        record_name = next(name for name in files if name.endswith(".dist-info/WHEEL"))
        record_name = record_name.removesuffix("WHEEL") + "RECORD"
        records.append([record_name, "", ""])
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(records)
        wheel.writestr(record_name, stream.getvalue().encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libraries-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--native-contract-version", type=int, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--mathdx-version", default="none")
    parser.add_argument("--mathdx-license", type=Path, default=None)
    parser.add_argument("--project-license", type=Path, default=ROOT / "LICENSE")
    parser.add_argument("--notice", type=Path, default=ROOT / "NOTICE")
    parser.add_argument("--max-glibc", default="2.31")
    parser.add_argument("--platform-tag", default=PLATFORM_TAG)
    parser.add_argument(
        "--architectures",
        default=",".join(str(a) for a in SUPPORTED_ARCHITECTURES),
        help="Comma/semicolon-separated SM architectures packaged in the wheel",
    )
    args = parser.parse_args()

    required_paths = [args.python, args.project_license, args.notice]
    if args.mathdx_license is not None:
        required_paths.append(args.mathdx_license)
    for path in required_paths:
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    try:
        max_glibc = _parse_glibc_version(args.max_glibc)
        architectures = _parse_architectures(args.architectures)
    except ValueError as error:
        parser.error(str(error))

    torch_contract = _torch_contract(args.python)
    if torch_contract["version"] != args.torch_version:
        parser.error(
            f"selected Python has torch {torch_contract['version']}, "
            f"expected {args.torch_version}"
        )
    if torch_contract["cuda_version"] != args.cuda_version:
        parser.error(
            f"selected Python has torch CUDA {torch_contract['cuda_version']}, "
            f"expected {args.cuda_version}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unified_library = args.libraries_dir / "memsolve_equilibrium.so"
    manifest_artifacts: dict[str, dict[str, str]] = {}
    packaged_binaries: dict[str, bytes] = {}

    if unified_library.is_file():
        validated = _strip_and_validate_library(unified_library, max_glibc=max_glibc)
        unified_bytes = unified_library.read_bytes()
        packaged_binaries[f"{PACKAGE_NAME}/lib/memsolve_equilibrium.so"] = unified_bytes
        for architecture in architectures:
            manifest_artifacts[str(architecture)] = dict(validated)
    else:
        artifacts: dict[int, Path] = {
            architecture: args.libraries_dir / f"memsolve_equilibrium_sm{architecture}.so"
            for architecture in architectures
        }
        manifest_artifacts = {
            str(architecture): _strip_and_validate_library(
                library,
                max_glibc=max_glibc,
            )
            for architecture, library in artifacts.items()
        }
        packaged_binaries = {
            f"{PACKAGE_NAME}/lib/{library.name}": library.read_bytes()
            for library in artifacts.values()
        }

    torch_public_version = args.torch_version.split("+", maxsplit=1)[0]
    local = (
        f"torch{torch_public_version.replace('.', '')}"
        f"cu{args.cuda_version.replace('.', '')}"
    )
    wheel_version = f"{args.version}+{local}"
    normalized_dist = _normalized(DIST_NAME)
    dist_info = f"{normalized_dist}-{wheel_version}.dist-info"
    wheel_path = args.output_dir / (
        f"{normalized_dist}-{wheel_version}-py3-none-{args.platform_tag}.whl"
    )
    metadata = {
        "schema_version": 1,
        "memsolve_version": args.version,
        "native_contract_version": args.native_contract_version,
        "torch_version": args.torch_version,
        "cuda_version": args.cuda_version,
        "cxx11_abi": torch_contract["cxx11_abi"],
        "mathdx_version": args.mathdx_version,
        "platform_tag": args.platform_tag,
        "architectures": list(architectures),
        "artifacts": manifest_artifacts,
    }
    license_lines = (
        "License-File: licenses/LICENSE\n"
        "License-File: licenses/NOTICE\n"
    )
    if args.mathdx_license is not None:
        license_lines += "License-File: licenses/LICENSE.NVIDIA-MATHDX\n"
    core_metadata = (
        "Metadata-Version: 2.3\n"
        "Name: memsolve-cuda-runtime\n"
        f"Version: {wheel_version}\n"
        "Summary: Precompiled CUDA runtime for MemSolve\n"
        "Requires-Python: >=3.10,<3.13\n"
        f"{license_lines}"
    ).encode("utf-8")
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: MemSolve tools/package_cuda_runtime.py\n"
        "Root-Is-Purelib: false\n"
        f"Tag: py3-none-{args.platform_tag}\n"
    ).encode("utf-8")
    files = {
        f"{PACKAGE_NAME}/__init__.py": _runtime_init(
            memsolve_version=args.version,
            native_contract_version=args.native_contract_version,
            torch_version=args.torch_version,
            cuda_version=args.cuda_version,
            cxx11_abi=int(torch_contract["cxx11_abi"]),
            architectures=architectures,
        ),
        f"{PACKAGE_NAME}/metadata.json": (
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        f"{dist_info}/METADATA": core_metadata,
        f"{dist_info}/WHEEL": wheel_metadata,
        f"{dist_info}/licenses/LICENSE": args.project_license.read_bytes(),
        f"{dist_info}/licenses/NOTICE": args.notice.read_bytes(),
    }
    if args.mathdx_license is not None:
        files[f"{dist_info}/licenses/LICENSE.NVIDIA-MATHDX"] = args.mathdx_license.read_bytes()
    files.update(packaged_binaries)
    _write_wheel(wheel_path, files=files)
    print(wheel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
