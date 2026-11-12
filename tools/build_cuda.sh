#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_root="${CUDA_HOME:-/usr/local/cuda-13.2}"
use_mathdx="${MEMSOLVE_USE_MATHDX:-OFF}"
mathdx_root="${MATHDX_ROOT:-/opt/nvidia/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06}"
python_bin="${PYTHON:-python3}"
build_dir="${repo_root}/build/cuda"

case "${use_mathdx^^}" in
    ON|1|TRUE|YES)
        use_mathdx="ON"
        architectures="${MEMSOLVE_CUDA_ARCHITECTURES:-80;86;87;89;90;100;120}"
        ;;
    OFF|0|FALSE|NO)
        use_mathdx="OFF"
        architectures="${MEMSOLVE_CUDA_ARCHITECTURES:-80;90a}"
        ;;
    *)
        printf 'Invalid MEMSOLVE_USE_MATHDX=%s (expected ON or OFF)\n' "${use_mathdx}" >&2
        exit 1
        ;;
esac

if [[ ! -x "${cuda_root}/bin/nvcc" ]]; then
    printf 'CUDA nvcc not found at %s/bin/nvcc\n' "${cuda_root}" >&2
    exit 1
fi

cuda_version="$("${cuda_root}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if [[ -z "${cuda_version}" ]]; then
    printf 'Could not determine CUDA version from %s/bin/nvcc\n' "${cuda_root}" >&2
    exit 1
fi

if [[ "${use_mathdx}" == "ON" ]]; then
    if [[ "${cuda_version}" != "13.2" ]]; then
        printf 'MemSolve MathDx builds require CUDA 13.2 exactly; found %s at %s\n' "${cuda_version}" "${cuda_root}" >&2
        exit 1
    fi
    if [[ ! -f "${mathdx_root}/lib/cmake/mathdx/mathdx-config.cmake" ]]; then
        printf 'MathDx 26.06 package not found under %s\n' "${mathdx_root}" >&2
        exit 1
    fi
fi

torch_prefix="$("${python_bin}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
torch_cuda="$("${python_bin}" -c 'import torch; print(torch.version.cuda or "")')"
site_packages="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1])')"
if [[ "${torch_cuda}" != "${cuda_version}" ]]; then
    printf 'The selected PyTorch CUDA (%s) must match nvcc CUDA (%s)\n' "${torch_cuda:-cpu}" "${cuda_version}" >&2
    exit 1
fi

host_arch="$(uname -m)"
case "${host_arch}" in
    x86_64|amd64) cuda_target_dir="x86_64-linux" ;;
    aarch64|arm64) cuda_target_dir="sbsa-linux" ;;
    *)
        printf 'Unsupported host CPU architecture: %s\n' "${host_arch}" >&2
        exit 1
        ;;
esac

nvrtc_library=""
for candidate in \
    "${cuda_root}/targets/${cuda_target_dir}/lib/libnvrtc.so" \
    "${cuda_root}/lib64/libnvrtc.so" \
    "${site_packages}/nvidia/cuda_nvrtc/lib/libnvrtc.so.13" \
    "${site_packages}/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"; do
    if [[ -f "${candidate}" ]]; then
        nvrtc_library="${candidate}"
        break
    fi
done
if [[ -z "${nvrtc_library}" ]]; then
    printf 'CUDA NVRTC library was not found under %s (target %s) or the selected PyTorch environment\n' "${cuda_root}" "${cuda_target_dir}" >&2
    exit 1
fi

export CUDA_HOME="${cuda_root}"
export PATH="${cuda_root}/bin:${PATH}"

IFS=';' read -r -a architecture_list <<< "${architectures}"
if [[ ${#architecture_list[@]} -eq 0 ]]; then
    printf 'MEMSOLVE_CUDA_ARCHITECTURES must contain at least one architecture\n' >&2
    exit 1
fi

if [[ "${use_mathdx}" == "OFF" ]]; then
    for architecture in "${architecture_list[@]}"; do
        case "${architecture}" in
            80|86|87|89|90|90a|100|100a|103|103a|110|120|120a) ;;
            *)
                printf 'MEMSOLVE_CUDA_ARCHITECTURES contains unsupported architecture %s\n' "${architecture}" >&2
                exit 1
                ;;
        esac
    done

    fatbin_build_dir="${build_dir}/fatbin"
    cmake -E rm -rf "${fatbin_build_dir}"
    cmake -E make_directory "${build_dir}/lib"
    rm -f "${build_dir}"/lib/memsolve_equilibrium*.so

    cmake -S "${repo_root}/csrc/ball" -B "${fatbin_build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc" \
        -DCUDAToolkit_ROOT="${cuda_root}" \
        -DCUDA_nvrtc_LIBRARY="${nvrtc_library}" \
        -DCMAKE_PREFIX_PATH="${torch_prefix}" \
        -DMEMSOLVE_USE_MATHDX=OFF \
        -DMEMSOLVE_CUDA_ARCHITECTURES="${architectures}" \
        -DMEMSOLVE_CUDA_OUTPUT_DIR="${build_dir}/lib"

    build_args=(--build "${fatbin_build_dir}" --config Release)
    if [[ -n "${MEMSOLVE_CUDA_JOBS:-}" ]]; then
        build_args+=(--parallel "${MEMSOLVE_CUDA_JOBS}")
    fi
    cmake "${build_args[@]}"
    printf 'Built multi-arch fatbinary %s (architectures: %s)\n' "${build_dir}/lib/memsolve_equilibrium.so" "${architectures}"
else
    cmake -E rm -f "${build_dir}/lib/memsolve_equilibrium.so"
    for architecture in "${architecture_list[@]}"; do
        case "${architecture}" in
            80|86|87|89|90|90a|100|100a|120|120a) ;;
            *)
                printf 'MEMSOLVE_CUDA_ARCHITECTURES contains unsupported MathDx architecture %s\n' "${architecture}" >&2
                exit 1
                ;;
        esac

        numeric_arch="${architecture%a}"
        artifact_build_dir="${build_dir}/sm${numeric_arch}"
        artifact_library="${build_dir}/lib/memsolve_equilibrium_sm${numeric_arch}.so"

        # Clear cached Torch_DIR/libtorch targets when switching Python environments.
        cmake -E rm -rf "${artifact_build_dir}"
        cmake -E rm -f "${artifact_library}"

        cmake -S "${repo_root}/csrc/ball" -B "${artifact_build_dir}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc" \
            -DCUDAToolkit_ROOT="${cuda_root}" \
            -DCUDA_nvrtc_LIBRARY="${nvrtc_library}" \
            -DCMAKE_PREFIX_PATH="${torch_prefix}" \
            -Dmathdx_DIR="${mathdx_root}/lib/cmake/mathdx" \
            -DMEMSOLVE_USE_MATHDX=ON \
            -DMEMSOLVE_CUDA_ARCHITECTURE="${architecture}" \
            -DMEMSOLVE_CUDA_OUTPUT_DIR="${build_dir}/lib"

        build_args=(--build "${artifact_build_dir}" --config Release)
        if [[ -n "${MEMSOLVE_CUDA_JOBS:-}" ]]; then
            build_args+=(--parallel "${MEMSOLVE_CUDA_JOBS}")
        fi
        cmake "${build_args[@]}"
        printf 'Built %s\n' "${artifact_library}"
    done
fi
