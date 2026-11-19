#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COMMENT="${1:-}"
if [[ -n "${COMMENT}" ]]; then
  COMMENT_SUFFIX="${COMMENT}"
else
  COMMENT_SUFFIX=""
fi

GPUS="${GPUS:-0}"
DATASET="${DATASET:-det_compass_xp}"
SAM2_CKPT="${SAM2_CKPT:-/path/to/checkpoints/sam2.1_s.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/LAO-X/output_eval}"
CONF="${CONF:-0.25}"

START_ID="${START_ID:-0}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs specified in GPUS." >&2
  exit 1
fi

TOTAL_IMAGES="$(DATASET="${DATASET}" python - <<PY
from config import XrayRoot
import json, os

dataset = os.environ.get("DATASET", "det_compass_xp")
root = XrayRoot.get(dataset)
if not root:
    raise SystemExit(1)

candidates = [
    *(
        [os.path.join(root, "annotations", "instances_test2017_agnostic.json")]
        if dataset.lower() == "pidray"
        else []
    ),
    os.path.join(root, "val.json"),
    os.path.join(root, "default_coco_xray_v2.json"),
    os.path.join(root, "default_coco_xray_v1.json"),
    os.path.join(root, "coco_agnostic.json"),
    os.path.join(root, "annotations", "coco_agnostic.json"),
    os.path.join(root, "default_coco_xray.json"),
]
path = next((p for p in candidates if os.path.exists(p)), None)
if not path:
    raise SystemExit(1)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
images = data.get("images", [])
print(len(images))
PY
)"

if [[ -z "${END_ID}" ]]; then
  END_ID="${TOTAL_IMAGES}"
fi

RANGE_SIZE=$((END_ID - START_ID))
if [[ "${RANGE_SIZE}" -le 0 ]]; then
  echo "Invalid range: START_ID=${START_ID} END_ID=${END_ID}" >&2
  exit 1
fi

CHUNK_SIZE=$(( (RANGE_SIZE + NUM_GPUS - 1) / NUM_GPUS ))

run_shard() {
  local gpu="$1"
  local start_id="$2"
  local end_id="$3"
  echo "[sam2] dataset=${DATASET} gpu=${gpu} start=${start_id} end=${end_id}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python test_sam2.py \
      --dataset "${DATASET}" \
      --sam2-ckpt "${SAM2_CKPT}" \
      --output-dir "${OUTPUT_DIR}" \
      --start-id "${start_id}" \
      --end-id "${end_id}" \
      --conf "${CONF}" \
      --comment "${COMMENT_SUFFIX}" &
}

for i in "${!GPU_LIST[@]}"; do
  shard_start=$((START_ID + i * CHUNK_SIZE))
  shard_end=$((START_ID + (i + 1) * CHUNK_SIZE))
  if [[ "${shard_end}" -gt "${END_ID}" ]]; then
    shard_end="${END_ID}"
  fi
  if [[ "${shard_start}" -ge "${shard_end}" ]]; then
    continue
  fi
  run_shard "${GPU_LIST[$i]}" "${shard_start}" "${shard_end}"
done

wait
