#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR_ROOT="${GENERATOR_ROOT:-${REPO_ROOT}/dependencies/det_compass_crops}"
GENERATOR="${GENERATOR_ROOT}/create_synthetic_dataset_musc.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

WORKERS=16
NUM_GENERATED=8000
STAGE=""
NUM_OBJECTS=""
GAMMA=""
DATASET_ROOT="/path/to/datasets"
DATASET_GEN=""
OBJECT_BANK_JSON=""
IO_WORKERS=2
SEED="40"
OFFLINE=1
COMPOSITION_MODE="xray"
COLOR_MAP_MODE="legacy"
COLOR_MAP_PALETTE="all"
COLOR_MAP_JITTER="0.0"
CONTAINER_COUNT_PROBS="1,0,0"
TEST_COCO="/path/to/datasets/det_compass_xp/annotations/coco_agnostic.json"
MATCHES_JSON="/path/to/datasets/det_compass_xp/annotations/huggingface_matches.json"

usage() {
  cat <<'EOF'
Usage:
  task_create_synthetic_dataset_object_bank.sh \
    --object-bank-json FILE --stage STAGE --num-objects MIN,MAX \
    --gamma MIN,MAX --dataset-gen NAME [options]

Required:
  --object-bank-json FILE  Exact COCO mask JSON used to build the object bank.
  --stage STAGE            easy, hard, or extreme.
  --num-objects MIN,MAX    Object-count range for each synthetic image.
  --gamma MIN,MAX          Overlap range.
  --dataset-gen NAME       Output dataset directory name.

Options:
  --workers N              Generation processes (default: 16).
  --num-generated N        Images generated for this stage (default: 8000).
  --dataset-root DIR       Output dataset parent (default: /path/to/datasets).
  --io-workers N           Image writer threads per process (default: 2).
  --seed N                 Fixed base seed (default: 40); image streams do not depend on worker count.
  --online                 Allow HuggingFace downloads instead of cache-only mode.
  --test-coco FILE         Labeled test COCO excluded from the object bank.
  --matches-json FILE      Local-file to HuggingFace split/index mapping.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --object-bank-json)
      OBJECT_BANK_JSON="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --num-generated)
      NUM_GENERATED="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --num-objects)
      NUM_OBJECTS="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --dataset-gen)
      DATASET_GEN="$2"
      shift 2
      ;;
    --io-workers)
      IO_WORKERS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --online)
      OFFLINE=0
      shift
      ;;
    --test-coco)
      TEST_COCO="$2"
      shift 2
      ;;
    --matches-json)
      MATCHES_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
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

if (( WORKERS <= 0 || NUM_GENERATED <= 0 || IO_WORKERS <= 0 )); then
  echo "workers, num-generated, and io-workers must be positive." >&2
  exit 2
fi

# The generator forms the path as "${coco_dir}_${crop_ablation}.json".
object_bank_stem="${OBJECT_BANK_JSON%.json}"
if [[ "${object_bank_stem}" == "${OBJECT_BANK_JSON}" || "${object_bank_stem}" != *_* ]]; then
  echo "Object-bank filename must be a .json name containing an underscore." >&2
  exit 2
fi
COCO_DIR="${object_bank_stem%_*}"
CROP_ABLATION="${object_bank_stem##*_}"

stage_dir="${DATASET_ROOT%/}/${DATASET_GEN}/${STAGE}"
if [[ -d "${stage_dir}" ]] && [[ -n "$(find "${stage_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to mix with an existing stage: ${stage_dir}/coco.json" >&2
  exit 1
fi

if ! [[ "${SEED}" =~ ^[0-9]+$ ]] || (( SEED > 4294967295 )); then
  echo "seed must be an unsigned 32-bit integer." >&2
  exit 2
fi
export PYTHONHASHSEED="${SEED}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

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

  common_args=(
    --mode augment
    --stage "${STAGE}"
    --num_generated "${count}"
    --start_id "${start_id}"
    --num_objects "${NUM_OBJECTS}"
    --gamma "${GAMMA}"
    --dataset_root "${DATASET_ROOT}"
    --dataset_gen "${DATASET_GEN}"
    --io_workers "${IO_WORKERS}"
    --composition_mode "${COMPOSITION_MODE}"
    --coco_dir "${COCO_DIR}"
    --crop_ablation "${CROP_ABLATION}"
    --xray_test_anno_dir "${TEST_COCO}"
    --matches_path "${MATCHES_JSON}"
    --color_map_mode "${COLOR_MAP_MODE}"
    --color_map_palette "${COLOR_MAP_PALETTE}"
    --color_map_jitter "${COLOR_MAP_JITTER}"
    --container_count_probs "${CONTAINER_COUNT_PROBS}"
  )
  if [[ -n "${SEED}" ]]; then
    common_args+=(--seed "${SEED}")
  fi
  if (( OFFLINE == 1 )); then
    common_args+=(--offline)
  else
    common_args+=(--online)
  fi

  echo "[gen] worker=${worker_id} start_id=${start_id} count=${count} object_bank=${OBJECT_BANK_JSON}"
  "${PYTHON_BIN}" "${GENERATOR}" "${common_args[@]}" &
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

"${PYTHON_BIN}" "${GENERATOR}" \
  --mode merge \
  --stage "${STAGE}" \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_gen "${DATASET_GEN}" \
  --merge_pattern 'coco_[0-9]*_[0-9]*.json' \
  --merge_output coco.json

echo "Synthetic stage ready: ${stage_dir}"
