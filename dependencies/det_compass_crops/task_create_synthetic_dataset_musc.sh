#!/usr/bin/env bash
set -e
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKERS="${WORKERS:-4}"
NUM_GENERATED="${NUM_GENERATED:-8000}"
STAGE="${STAGE:-easy}"
NUM_OBJECTS="${NUM_OBJECTS:-2,4}"
GAMMA="${GAMMA:-0.0,0.05}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/datasets/}"
DATASET_GEN="${DATASET_GEN:-crop_aug}"
IO_WORKERS="${IO_WORKERS:-4}"
OFFLINE="${OFFLINE:-1}"
MERGE_PATTERN="${MERGE_PATTERN:-coco_*.json}"
MERGE_OUTPUT="${MERGE_OUTPUT:-coco.json}"
COMPOSITION_MODE="${COMPOSITION_MODE:-xray}"
CROP_ABLATION="${CROP_ABLATION:-musc}"
RUN_MODE="${RUN_MODE:-augment}"
PREVIEW_NUM_SAMPLES="${PREVIEW_NUM_SAMPLES:-6}"
PREVIEW_DIR="${PREVIEW_DIR:-preview_colormap}"
# LAO-X synthesis: one container per image, rendered with the fixed
# high/low-energy -> colour/grey/density lookup tables built from DET-COMPASS.
COLOR_MAP_MODE="${COLOR_MAP_MODE:-legacy}"
COLOR_MAP_PALETTE="${COLOR_MAP_PALETTE:-all}"
COLOR_MAP_JITTER="${COLOR_MAP_JITTER:-0.0}"
COLOR_MAP_SEED="${COLOR_MAP_SEED:-}"
CONTAINER_COUNT_PROBS="${CONTAINER_COUNT_PROBS:-1,0,0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --offline)
      OFFLINE="$2"
      shift 1
      ;;
    --merge-pattern)
      MERGE_PATTERN="$2"
      shift 2
      ;;
    --merge-output)
      MERGE_OUTPUT="$2"
      shift 2
      ;;
    --composition-mode|--composition_mode)
      COMPOSITION_MODE="$2"
      shift 2
      ;;
    --crop-ablation|--crop_ablation)
      CROP_ABLATION="$2"
      shift 2
      ;;
    --mode)
      RUN_MODE="$2"
      shift 2
      ;;
    --preview-num-samples|--preview_num_samples)
      PREVIEW_NUM_SAMPLES="$2"
      shift 2
      ;;
    --preview-dir|--preview_dir)
      PREVIEW_DIR="$2"
      shift 2
      ;;
    --color-map-mode|--color_map_mode)
      COLOR_MAP_MODE="$2"
      shift 2
      ;;
    --color-map-palette|--color_map_palette)
      COLOR_MAP_PALETTE="$2"
      shift 2
      ;;
    --color-map-jitter|--color_map_jitter)
      COLOR_MAP_JITTER="$2"
      shift 2
      ;;
    --color-map-seed|--color_map_seed)
      COLOR_MAP_SEED="$2"
      shift 2
      ;;
    --container-count-probs|--container_count_probs)
      CONTAINER_COUNT_PROBS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${WORKERS}" -le 0 ]]; then
  echo "WORKERS must be > 0." >&2
  exit 1
fi

PER_WORKER=$((NUM_GENERATED / WORKERS))
REMAINDER=$((NUM_GENERATED % WORKERS))

run_shard() {
  local worker_id="$1"
  local start_id="$2"
  local count="$3"
  local color_map_args=(
    --color_map_mode "${COLOR_MAP_MODE}"
    --color_map_palette "${COLOR_MAP_PALETTE}"
    --color_map_jitter "${COLOR_MAP_JITTER}"
    --container_count_probs "${CONTAINER_COUNT_PROBS}"
  )
  if [[ -n "${COLOR_MAP_SEED}" ]]; then
    color_map_args+=(--color_map_seed "${COLOR_MAP_SEED}")
  fi

  echo "[gen] worker=${worker_id} start_id=${start_id} num_generated=${count}"
  python "${SCRIPT_DIR}/create_synthetic_dataset_musc.py" \
    --mode augment \
    --stage "${STAGE}" \
    --num_generated "${count}" \
    --start_id "${start_id}" \
    --num_objects "${NUM_OBJECTS}" \
    --gamma "${GAMMA}" \
    --dataset_root "${DATASET_ROOT}" \
    --dataset_gen "${DATASET_GEN}" \
    --io_workers "${IO_WORKERS}" \
    --composition_mode "${COMPOSITION_MODE}" \
    --crop_ablation "${CROP_ABLATION}" \
    "${color_map_args[@]}" \
    $( [[ "${OFFLINE}" -eq 1 ]] && echo "--offline" )
}

if [[ "${RUN_MODE}" == "preview" ]]; then
  echo "[preview] stage=${STAGE} num_samples=${PREVIEW_NUM_SAMPLES}"
  preview_args=(
    --mode preview
    --stage "${STAGE}"
    --num_objects "${NUM_OBJECTS}"
    --gamma "${GAMMA}"
    --dataset_root "${DATASET_ROOT}"
    --dataset_gen "${DATASET_GEN}"
    --io_workers "${IO_WORKERS}"
    --composition_mode "${COMPOSITION_MODE}"
    --crop_ablation "${CROP_ABLATION}"
    --preview_num_samples "${PREVIEW_NUM_SAMPLES}"
    --preview_dir "${PREVIEW_DIR}"
    --color_map_mode "${COLOR_MAP_MODE}"
    --color_map_palette "${COLOR_MAP_PALETTE}"
    --color_map_jitter "${COLOR_MAP_JITTER}"
    --container_count_probs "${CONTAINER_COUNT_PROBS}"
  )
  if [[ -n "${COLOR_MAP_SEED}" ]]; then
    preview_args+=(--color_map_seed "${COLOR_MAP_SEED}")
  fi
  python "${SCRIPT_DIR}/create_synthetic_dataset_musc.py" \
    "${preview_args[@]}" \
    $( [[ "${OFFLINE}" -eq 1 ]] && echo "--offline" )
  exit 0
fi

start=0
for ((i=0; i<WORKERS; i++)); do
  count="${PER_WORKER}"
  if (( i < REMAINDER )); then
    count=$((count + 1))
  fi
  run_shard "${i}" "${start}" "${count}" &
  start=$((start + count))
done
wait

echo "[merge] stage=${STAGE}"
python "${SCRIPT_DIR}/create_synthetic_dataset_musc.py" \
  --mode merge \
  --stage "${STAGE}" \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_gen "${DATASET_GEN}" \
  --merge_pattern "${MERGE_PATTERN}" \
  --merge_output "${MERGE_OUTPUT}"
