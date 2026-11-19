#!/usr/bin/env bash
# Quick CLCXray check of object-bank ablation runs: freezes the latest
# checkpoint of every ${TRAIN_ROOT}/augmentation_restart_crop_objbank_*_fusion_bs16_seed1
# experiment (plus an optional reference run), evaluates images
# [START_ID, END_ID) with test_det_v2.py and summarizes them with
# scripts/summarize_quick_object_bank_eval.py.
#
# Usage: [VAR=value ...] bash scripts/quick_eval_object_bank_clcxray.sh
# Environment: TRAIN_ROOT [${LAOX_RESULTS_ROOT:-<repo>/results}/train],
# OUTPUT_ROOT [<repo>/output_eval/quick_object_bank_clcxray], START_ID [0], END_ID [32],
# GPU [0], GRA [1.0], MASK_THRESHOLD [0.5], REFERENCE_EXPERIMENT, PYTHON [python].
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
TRAIN_ROOT="${TRAIN_ROOT:-${LAOX_RESULTS_ROOT:-${REPO_ROOT}/results}/train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output_eval/quick_object_bank_clcxray}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-32}"
GPU="${GPU:-0}"
GRA="${GRA:-1.0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EVAL_SEED="${EVAL_SEED:-1}"
INCLUDE_REFERENCE="${INCLUDE_REFERENCE:-1}"
REFERENCE_EXPERIMENT="${REFERENCE_EXPERIMENT:-${TRAIN_ROOT}/augmentation_restart_crop_musc_aug_fusion_bs16_xp}"
REFERENCE_STAGE="${REFERENCE_STAGE:-extreme}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
PREDICTION_ROOT="${RUN_DIR}/predictions"
SNAPSHOT_ROOT="${RUN_DIR}/checkpoints"
LOG_ROOT="${RUN_DIR}/logs"
MANIFEST="${RUN_DIR}/manifest.tsv"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if (( END_ID <= START_ID )); then
  echo "END_ID must be greater than START_ID." >&2
  exit 1
fi

mkdir -p "${PREDICTION_ROOT}" "${SNAPSHOT_ROOT}" "${LOG_ROOT}"
printf 'method\texperiment\tstage\tepoch\tseed\tsource_checkpoint\tsnapshot\tprediction\tstatus\n' > "${MANIFEST}"

mapfile -t EXPERIMENT_DIRS < <(
  find "${TRAIN_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name 'augmentation_restart_crop_objbank_*_fusion_bs16_seed1' | sort
)
if [[ "${#EXPERIMENT_DIRS[@]}" -eq 0 ]]; then
  echo "No object-bank training directories found under ${TRAIN_ROOT}" >&2
  exit 1
fi
if [[ "${INCLUDE_REFERENCE}" == "1" && -d "${REFERENCE_EXPERIMENT}" ]]; then
  EXPERIMENT_DIRS+=("${REFERENCE_EXPERIMENT}")
fi

failures=0
for experiment_dir in "${EXPERIMENT_DIRS[@]}"; do
  experiment="$(basename "${experiment_dir}")"
  if [[ "${experiment_dir}" == "${REFERENCE_EXPERIMENT}" ]]; then
    method="previous_8gpu_xp"
    source_checkpoint="${experiment_dir}/${REFERENCE_STAGE}/log/checkpoints/checkpoint.pt"
    latest_record=$'reference\t'"${source_checkpoint}"
  else
    method="${experiment#augmentation_restart_crop_objbank_}"
    method="${method%_fusion_bs16_seed1}"
    latest_record="$(
      find "${experiment_dir}" -mindepth 4 -maxdepth 4 -type f \
        -path '*/log/checkpoints/checkpoint.pt' -printf '%T@\t%p\n' \
        | sort -nr | head -n 1
    )"
    source_checkpoint="${latest_record#*$'\t'}"
  fi
  if [[ -z "${latest_record}" || ! -f "${source_checkpoint}" ]]; then
    echo "[skip] ${method}: no checkpoint.pt found"
    continue
  fi
  stage="$(basename "$(dirname "$(dirname "$(dirname "${source_checkpoint}")")")")"

  snapshot_tmp="${SNAPSHOT_ROOT}/${method}_${stage}.pt.tmp"
  metadata_file="${SNAPSHOT_ROOT}/${method}_${stage}.metadata.tmp"
  snapshot_ok=0
  for attempt in 1 2 3; do
    if "${PYTHON_BIN}" -c \
      'import sys, torch; d=torch.load(sys.argv[1], map_location="cpu", weights_only=False); torch.save({"model": d["model"]}, sys.argv[2]); print(int(d.get("epoch", -1)), int(d.get("seed_value", -1)))' \
      "${source_checkpoint}" "${snapshot_tmp}" > "${metadata_file}"; then
      snapshot_ok=1
      break
    fi
    echo "[retry] ${method}: checkpoint changed while being read (${attempt}/3)" >&2
    sleep 2
  done
  if [[ "${snapshot_ok}" != "1" ]]; then
    echo "[skip] ${method}: could not freeze ${source_checkpoint}" >&2
    failures=$((failures + 1))
    continue
  fi
  read -r epoch seed < "${metadata_file}"
  if [[ "${seed}" == "-1" ]]; then
    resolved_config="${experiment_dir}/${stage}/log/config_resolved.yaml"
    if [[ -f "${resolved_config}" ]]; then
      seed="$(awk '/^  seed_value:/ {print $2; exit}' "${resolved_config}")"
      seed="${seed:-1}"
    fi
  fi
  snapshot="${SNAPSHOT_ROOT}/${method}_${stage}_epoch${epoch}.pt"
  mv -f "${snapshot_tmp}" "${snapshot}"
  # The inference loader uses weights_only=True, so validate that exact path.
  "${PYTHON_BIN}" -c \
    'import sys, torch; d=torch.load(sys.argv[1], map_location="cpu", weights_only=True); assert "model" in d' \
    "${snapshot}"

  comment="_quick_objbank_${method}_${stage}_e${epoch}_m${MASK_THRESHOLD}"
  ckpt_suffix="epoch${epoch}"
  output_subdir="clcxray_img${START_ID}_${END_ID}_unsamv2_${GRA}${comment}_${ckpt_suffix}"
  prediction="${PREDICTION_ROOT}/${output_subdir}/coco_instances_results.json"
  log_file="${LOG_ROOT}/${method}_${stage}_epoch${epoch}.log"

  echo "[eval] method=${method} stage=${stage} epoch=${epoch} seed=${seed}"
  echo "[eval] snapshot=${snapshot}"
  if CUDA_VISIBLE_DEVICES="${GPU}" \
      CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
      PYTHONHASHSEED="${EVAL_SEED}" \
      "${PYTHON_BIN}" test_det_v2.py \
        --method unsamv2 \
        --dataset CLCXray \
        --sam2-ckpt "${snapshot}" \
        --ckpt-suffix "${ckpt_suffix}" \
        --gra "${GRA}" \
        --eval-mode bbox \
        --start_id "${START_ID}" \
        --end_id "${END_ID}" \
        --comment "${comment}" \
        --mask-threshold "${MASK_THRESHOLD}" \
        --test_batch 1 \
        --num_workers "${NUM_WORKERS}" \
        --seed "${EVAL_SEED}" \
        --model-seed "${seed}" \
        --deterministic \
        --no-wandb \
        --no-resume-from-results \
        --save-every 0 \
        --device-id 0 \
        --output-dir "${PREDICTION_ROOT}" \
        2>&1 | tee "${log_file}"; then
    status="ok"
    if [[ ! -f "${prediction}" ]]; then
      status="missing_prediction"
      failures=$((failures + 1))
    fi
  else
    status="inference_failed"
    failures=$((failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${method}" "${experiment}" "${stage}" "${epoch}" "${seed}" \
    "${source_checkpoint}" "${snapshot}" "${prediction}" "${status}" \
    >> "${MANIFEST}"
done

"${PYTHON_BIN}" scripts/summarize_quick_object_bank_eval.py \
  --manifest "${MANIFEST}" \
  --start-id "${START_ID}" \
  --end-id "${END_ID}" \
  --protocol s3_r1p5_center \
  --output-json "${RUN_DIR}/summary.json" \
  --output-markdown "${RUN_DIR}/summary.md"

echo "Results: ${RUN_DIR}/summary.md"
if (( failures > 0 )); then
  exit 1
fi
