#!/usr/bin/env bash
# Build the ViT-Adapter/OpenMMLab native operators against the active Torch ABI.
set -euo pipefail
PYTHON="${PYTHON:-python}"
MMCV_BUILD_ROOT="${MMCV_BUILD_ROOT:-/tmp/memsolve-mmcv-2.1.0}"
MMCV_REVISION=57c4e25e06e2d4f8a9357c84bcd24089a284dc88
if [[ ! -d "$MMCV_BUILD_ROOT/.git" ]]; then
    git clone --depth 1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git "$MMCV_BUILD_ROOT"
fi
if [[ "$(git -C "$MMCV_BUILD_ROOT" rev-parse HEAD)" != "$MMCV_REVISION" ]]; then
    echo "MMCV_BUILD_ROOT must contain MMCV 2.1.0 ($MMCV_REVISION)" >&2
    exit 1
fi
"$PYTHON" - "$MMCV_BUILD_ROOT/setup.py" <<'PY'
import sys
from pathlib import Path
import torch
setup = Path(sys.argv[1])
header = Path(torch.__file__).parent / 'include/ATen/ATen.h'
if 'C++20' in header.read_text():
    setup.write_text(setup.read_text().replace('c++17', 'c++20'))
PY
export MMCV_WITH_OPS=1 FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
"$PYTHON" -m pip wheel --no-build-isolation --no-deps "$MMCV_BUILD_ROOT" -w "$MMCV_BUILD_ROOT/wheels"
"$PYTHON" -m pip uninstall -y mmcv-lite
"$PYTHON" -m pip install --force-reinstall --no-deps "$MMCV_BUILD_ROOT"/wheels/mmcv-2.1.0-*.whl
"$PYTHON" -c 'from mmcv.ops import MultiScaleDeformableAttention, RoIAlign, nms; print("MMCV native operators loaded")'
