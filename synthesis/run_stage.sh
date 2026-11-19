#!/usr/bin/env bash
# Generate one LAO-X synthesis stage: split the images over parallel
# generator processes, then merge their COCO shards into coco.json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENERATOR="${SCRIPT_DIR}/generate.py"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${LAOX_DATA_ROOT:-${REPO_ROOT}/datasets}"

OBJECT_BANK_JSON=""
STAGE=""
NUM_OBJECTS=""
GAMMA=""
DATASET_GEN=""
DATASET_ROOT="${DATA_ROOT}"
WORKERS=16
NUM_GENERATED=8000
IO_WORKERS=2
SEED=40
OFFLINE=1
TEST_COCO="${DATA_ROOT}/det_compass_xp/annotations/coco_agnostic.json"
MATCHES_JSON="${DATA_ROOT}/det_compass_xp/annotations/huggingface_matches.json"
COMPOSITION_MODE="xray"

usage() {
  cat <<'EOF'
Usage:
  run_stage.sh --object-bank-json FILE --stage STAGE --num-objects MIN,MAX \
               --gamma MIN,MAX --dataset-gen NAME [options]

Writes DATASET_ROOT/NAME/STAGE/{xray/*.png, coco_<start>_<end>.json, coco_train.json, coco.json}.

Required:
  --object-bank-json FILE  Object-bank COCO mask JSON (e.g. coco_cutler_musc.json).
  --stage STAGE            Stage name (easy, hard, extreme); part of every image seed.
  --num-objects MIN,MAX    Objects per synthetic image.
  --gamma MIN,MAX          Range of the maximum box IoU between placed objects.
  --dataset-gen NAME       Output dataset directory name.

Options:
  --dataset-root DIR       Output parent (default: ${LAOX_DATA_ROOT:-<repo>/datasets}).
  --workers N              Generator processes (default: 16).
  --num-generated N        Images in this stage (default: 8000).
  --io-workers N           PNG writer threads per process (default: 2).
  --seed N                 Base seed (default: 40); output does not depend on --workers.
  --online                 Allow HuggingFace downloads (default: local cache only).
  --test-coco FILE         Held-out labelled COCO excluded from the object bank
                           (default: ${LAOX_DATA_ROOT}/det_compass_xp/annotations/coco_agnostic.json).
  --matches-json FILE      Output of 'generate.py match'
                           (default: ${LAOX_DATA_ROOT}/det_compass_xp/annotations/huggingface_matches.json).
  --composition-mode MODE  xray (Beer-Lambert, default) or rgb (copy-paste ablation).

Environment:
  PYTHON         Python interpreter (default: python).
  LAOX_DATA_ROOT Default data root (default: <repo>/datasets).
  LAOX_LUT_DIR   Directory with the three lookup-table pickles (default: synthesis/luts).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --object-bank-json) OBJECT_BANK_JSON="$2"; shift 2 ;;
    --stage)            STAGE="$2"; shift 2 ;;
    --num-objects)      NUM_OBJECTS="$2"; shift 2 ;;
    --gamma)            GAMMA="$2"; shift 2 ;;
    --dataset-gen)      DATASET_GEN="$2"; shift 2 ;;
    --dataset-root)     DATASET_ROOT="$2"; shift 2 ;;
    --workers)          WORKERS="$2"; shift 2 ;;
    --num-generated)    NUM_GENERATED="$2"; shift 2 ;;
    --io-workers)       IO_WORKERS="$2"; shift 2 ;;
    --seed)             SEED="$2"; shift 2 ;;
    --online)           OFFLINE=0; shift ;;
    --test-coco)        TEST_COCO="$2"; shift 2 ;;
    --matches-json)     MATCHES_JSON="$2"; shift 2 ;;
    --composition-mode) COMPOSITION_MODE="$2"; shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for value_name in OBJECT_BANK_JSON STAGE NUM_OBJECTS GAMMA DATASET_GEN; do
  if [[ -z "${!value_name}" ]]; then
    echo "Missing required argument: ${value_name}" >&2
    usage >&2
    exit 2
  fi
done

for required_file in "${GENERATOR}" "${OBJECT_BANK_JSON}" "${TEST_COCO}" "${MATCHES_JSON}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "File not found: ${required_file}" >&2
    exit 1
  fi
done

for value_name in WORKERS NUM_GENERATED IO_WORKERS; do
  if ! [[ "${!value_name}" =~ ^[0-9]+$ ]] || (( ${!value_name} <= 0 )); then
    echo "${value_name} must be a positive integer." >&2
    exit 2
  fi
done

if ! [[ "${SEED}" =~ ^[0-9]+$ ]] || (( SEED > 4294967295 )); then
  echo "seed must be an unsigned 32-bit integer." >&2
  exit 2
fi

case "${COMPOSITION_MODE}" in
  xray|rgb) ;;
  *) echo "composition-mode must be xray or rgb." >&2; exit 2 ;;
esac

stage_dir="${DATASET_ROOT%/}/${DATASET_GEN}/${STAGE}"
if [[ -d "${stage_dir}" ]] && [[ -n "$(find "${stage_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to write into a non-empty stage directory: ${stage_dir}" >&2
  exit 1
fi

export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if (( OFFLINE == 1 )); then
  hf_mode=--offline
else
  hf_mode=--online
fi

per_worker=$((NUM_GENERATED / WORKERS))
remainder=$((NUM_GENERATED % WORKERS))
pids=()
start_id=0

for ((worker_id=0; worker_id<WORKERS; worker_id++)); do
  count="${per_worker}"
  if (( worker_id < remainder )); then
    count=$((count + 1))
  fi
  if (( count == 0 )); then
    continue
  fi

  echo "[gen] worker=${worker_id} start_id=${start_id} count=${count} object_bank=${OBJECT_BANK_JSON}"
  "${PYTHON}" "${GENERATOR}" augment \
    --object-bank-json "${OBJECT_BANK_JSON}" \
    --test-coco "${TEST_COCO}" \
    --matches-json "${MATCHES_JSON}" \
    --output-dir "${stage_dir}" \
    --stage "${STAGE}" \
    --num-generated "${count}" \
    --start-id "${start_id}" \
    --num-objects "${NUM_OBJECTS}" \
    --gamma "${GAMMA}" \
    --composition-mode "${COMPOSITION_MODE}" \
    --seed "${SEED}" \
    --io-workers "${IO_WORKERS}" \
    "${hf_mode}" &
  pids+=("$!")
  start_id=$((start_id + count))
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "At least one generation worker failed; merge was not run." >&2
  exit 1
fi

"${PYTHON}" "${GENERATOR}" merge \
  --output-dir "${stage_dir}" \
  --pattern 'coco_[0-9]*_[0-9]*.json' \
  --output coco.json

echo "Synthetic stage ready: ${stage_dir}"
