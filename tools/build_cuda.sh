#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_root="${CUDA_HOME:-/usr/local/cuda-13.2}"
mathdx_root="${MATHDX_ROOT:-/opt/nvidia/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06}"
python_bin="${PYTHON:-python3}"
architectures="${LSSO_CUDA_ARCHITECTURES:-80;86;87;89;90;100;120}"
build_dir="${repo_root}/build/cuda"

if [[ ! -x "${cuda_root}/bin/nvcc" ]]; then
    printf 'CUDA 13.2 nvcc not found at %s/bin/nvcc\n' "${cuda_root}" >&2
    exit 1
fi

cuda_version="$("${cuda_root}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if [[ "${cuda_version}" != "13.2" ]]; then
    printf 'LSSO requires CUDA 13.2 exactly; found %s at %s\n' "${cuda_version:-unknown}" "${cuda_root}" >&2
    exit 1
fi

if [[ ! -f "${mathdx_root}/lib/cmake/mathdx/mathdx-config.cmake" ]]; then
    printf 'MathDx 26.06 package not found under %s\n' "${mathdx_root}" >&2
    exit 1
fi

torch_prefix="$("${python_bin}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
torch_cuda="$("${python_bin}" -c 'import torch; print(torch.version.cuda or "")')"
torch_version="$("${python_bin}" -c 'import torch; print(torch.__version__)')"
if [[ "${torch_version}" != "2.14.0+cu132" ]]; then
    printf 'LSSO native builds require torch 2.14.0+cu132; found %s\n' "${torch_version}" >&2
    exit 1
fi
site_packages="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1])')"
if [[ "${torch_cuda}" != "13.2" ]]; then
    printf 'The selected PyTorch must be built for CUDA 13.2; found %s\n' "${torch_cuda:-cpu}" >&2
    exit 1
fi

nvrtc_library="${cuda_root}/targets/x86_64-linux/lib/libnvrtc.so"
if [[ ! -f "${nvrtc_library}" ]]; then
    nvrtc_library="${site_packages}/nvidia/cuda_nvrtc/lib/libnvrtc.so.13"
fi
if [[ ! -f "${nvrtc_library}" ]]; then
    printf 'CUDA 13.2 NVRTC library was not found under %s or the selected PyTorch environment\n' "${cuda_root}" >&2
    exit 1
fi

export CUDA_HOME="${cuda_root}"
export PATH="${cuda_root}/bin:${PATH}"

IFS=';' read -r -a architecture_list <<< "${architectures}"
if [[ ${#architecture_list[@]} -eq 0 ]]; then
    printf 'LSSO_CUDA_ARCHITECTURES must contain at least one architecture\n' >&2
    exit 1
fi

for architecture in "${architecture_list[@]}"; do
    case "${architecture}" in
        80|86|87|89|90|100|120) ;;
        *)
            printf 'LSSO_CUDA_ARCHITECTURES contains unsupported architecture %s\n' "${architecture}" >&2
            exit 1
            ;;
    esac

    artifact_build_dir="${build_dir}/sm${architecture}"
    artifact_library="${build_dir}/lib/lsso_equilibrium_sm${architecture}.so"

    # Torch_DIR and the imported libtorch targets are cached by CMake. Each
    # strict artifact must therefore start from a fresh build directory when
    # PYTHON selects a different PyTorch installation.
    cmake -E rm -rf "${artifact_build_dir}"
    cmake -E rm -f "${artifact_library}"

    cmake -S "${repo_root}/csrc/ball" -B "${artifact_build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc" \
        -DCUDAToolkit_ROOT="${cuda_root}" \
        -DCUDA_nvrtc_LIBRARY="${nvrtc_library}" \
        -DCMAKE_PREFIX_PATH="${torch_prefix}" \
        -Dmathdx_DIR="${mathdx_root}/lib/cmake/mathdx" \
        -DLSSO_CUDA_ARCHITECTURE="${architecture}" \
        -DLSSO_CUDA_OUTPUT_DIR="${build_dir}/lib"

    build_args=(--build "${artifact_build_dir}" --config Release)
    if [[ -n "${LSSO_CUDA_JOBS:-}" ]]; then
        build_args+=(--parallel "${LSSO_CUDA_JOBS}")
    fi
    cmake "${build_args[@]}"
    printf 'Built %s\n' "${build_dir}/lib/lsso_equilibrium_sm${architecture}.so"
done
