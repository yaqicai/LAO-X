#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-standard}"
# Unmodified upstream dependencies, pinned to the commits used for LAO-X.
DETECTRON2="${DETECTRON2:-git+https://github.com/MaureenZOU/detectron2-xyz.git@42121d75e10d9f858f3a91b6a39f5722c02868f0}"
DINOV3_COMMIT="54694f7627fd815f62a5dcc82944ffa6153bbb76"
SAM2_SOURCE="${ROOT}/sam2"

case "${PROFILE}" in
  standard)
    ENV_NAME="${ENV_NAME:-UnSAMv2}"
    YAML="${ROOT}/environment/UnSAMv2.rebuild.yml"
    ;;
  5090)
    ENV_NAME="${ENV_NAME:-UnSAMv2_5090}"
    YAML="${ROOT}/environment/UnSAMv2_5090.rebuild.yml"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
    ;;
  *)
    echo "Usage: $0 [standard|5090]" >&2
    exit 2
    ;;
esac

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${YAML}" --prune
else
  conda env create -n "${ENV_NAME}" -f "${YAML}"
fi

if [[ "${PROFILE}" == "5090" ]]; then
  # The 5090 environment ships its own CUDA toolkit; point CUDA_HOME at it.
  ENV_PREFIX="$(conda run -n "${ENV_NAME}" printenv CONDA_PREFIX)"
  conda env config vars set -n "${ENV_NAME}" CUDA_HOME="${ENV_PREFIX}" >/dev/null
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install --no-deps --no-build-isolation "${DETECTRON2}"
conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install --no-deps --no-build-isolation -e "${SAM2_SOURCE}"

DINOV3_DIR="${ROOT}/granularity_divide_and_conquer/dinov3"
if [[ ! -d "${DINOV3_DIR}" ]]; then
  git clone https://github.com/facebookresearch/dinov3.git "${DINOV3_DIR}"
  git -C "${DINOV3_DIR}" checkout --quiet "${DINOV3_COMMIT}"
fi

OPS="${ROOT}/granularity_divide_and_conquer/mask2former/modeling/pixel_decoder/ops"
if [[ -d "${OPS}" && "${BUILD_CUDA_OPS:-1}" == "1" ]]; then
  (
    cd "${OPS}"
    conda run --no-capture-output -n "${ENV_NAME}" bash make.sh
  )
fi

conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import torch
import detectron2
import sam2
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("detectron2", detectron2.__file__)
print("sam2", sam2.__file__)
PY

echo "Environment ready: ${ENV_NAME}"
