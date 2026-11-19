#!/usr/bin/env bash
set -euo pipefail
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Conda's CUDA toolkit keeps the device compiler under nvvm/bin instead of
# beside nvcc. nvcc launches cicc by name, so this directory must be on PATH.
if [[ -x "${CONDA_PREFIX:-}/nvvm/bin/cicc" ]]; then
  export PATH="${CONDA_PREFIX}/nvvm/bin:${PATH}"
fi

# Some conda CUDA packages provide nvcc without the development headers that
# torch extensions need. Prefer a complete toolkit when CUDA_HOME is incomplete.
if [[ -z "${CUDA_HOME:-}" || ! -f "${CUDA_HOME}/include/cuda_runtime_api.h" ]]; then
  for cuda_root in \
    "${CONDA_PREFIX:-}/targets/x86_64-linux" \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-12.6 \
    /usr/local/cuda; do
    if [[ -x "${cuda_root}/bin/nvcc" && -f "${cuda_root}/include/cuda_runtime_api.h" ]]; then
      export CUDA_HOME="${cuda_root}"
      break
    fi
  done
fi

if [[ ! -f "${CUDA_HOME:-}/include/cuda_runtime_api.h" ]]; then
  echo "A complete CUDA toolkit with cuda_runtime_api.h is required." >&2
  exit 1
fi

python - <<'PY'
import os
import sys

import torch
from torch.utils.cpp_extension import CUDA_HOME

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"CUDA_HOME: {CUDA_HOME}")
print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', '<auto>')}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(0)}")
PY

# Native PyTorch extensions must be rebuilt after changing PyTorch/CUDA.
python -m pip uninstall -y MultiScaleDeformableAttention || true
FORCE_CUDA=1 python setup.py clean --all
FORCE_CUDA=1 python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  --no-deps \
  --force-reinstall \
  .

python - <<'PY'
import torch
import MultiScaleDeformableAttention as MSDA

print(f"Loaded extension: {MSDA.__file__}")
if torch.cuda.is_available():
    print(f"Runtime GPU capability: {torch.cuda.get_device_capability(0)}")
else:
    print("Extension import passed; CUDA runtime test skipped (no visible GPU).")
PY
