#!/usr/bin/env bash
# Download the SAM 2.1 Hiera-S checkpoint that LAO-X training starts from.
#
# Usage: bash sam2/checkpoints/download_ckpts.sh
# Destination: ${LAOX_CKPT_ROOT} (default <repo>/checkpoints).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CKPT_ROOT="${LAOX_CKPT_ROOT:-${REPO_ROOT}/checkpoints}"

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
NAME="sam2.1_hiera_small.pt"

mkdir -p "${CKPT_ROOT}"
if [[ -s "${CKPT_ROOT}/${NAME}" ]]; then
  echo "${CKPT_ROOT}/${NAME} already exists."
  exit 0
fi

echo "Downloading ${NAME} to ${CKPT_ROOT}"
if command -v wget >/dev/null 2>&1; then
  wget -O "${CKPT_ROOT}/${NAME}.part" "${BASE_URL}/${NAME}"
elif command -v curl >/dev/null 2>&1; then
  curl -L -o "${CKPT_ROOT}/${NAME}.part" "${BASE_URL}/${NAME}"
else
  echo "Please install wget or curl to download the checkpoint." >&2
  exit 1
fi
mv "${CKPT_ROOT}/${NAME}.part" "${CKPT_ROOT}/${NAME}"
echo "Saved ${CKPT_ROOT}/${NAME}"
