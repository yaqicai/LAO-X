# Helpers shared by the ablation scripts. Source it; do not run it.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pipeline/env.sh"

# build_stage DATASET STAGE OBJECTS GAMMA [extra synthesis args...]
# Synthesis -> oversegmentation -> class-agnostic labels -> label fusion for one stage.
build_stage() {
  local dataset="$1" stage="$2" objects="$3" gamma="$4"
  shift 4
  bash "${LAOX_REPO}/synthesis/run_stage.sh" \
    --test-coco "${HELDOUT_COCO}" --matches-json "${MATCHES_JSON}" \
    --dataset-root "${LAOX_DATA_ROOT}" --dataset-gen "${dataset}" --stage "${stage}" \
    --num-objects "${objects}" --gamma "${gamma}" \
    --num-generated "${NUM_GENERATED:-8000}" --workers "${WORKERS:-16}" --io-workers 2 \
    "$@"
  bash "${LAOX_REPO}/granularity_divide_and_conquer/run_gra_dico.sh" "${dataset}" "${stage}"
  (
    cd "${LAOX_REPO}"
    "${PYTHON}" pseudo_label_generate.py --to-agnostic --dataset "${dataset}" --stage "${stage}"
    "${PYTHON}" pseudo_label_generate.py --dataset "${dataset}" --coco-name coco_agnostic.json --stage "${stage}"
    "${PYTHON}" pseudo_label_generate.py --fusion --fusion-name anno_fusion --dataset "${dataset}" \
      --stage "${stage}" --fusion-overlap-thresh 0.95
  )
}

# build_curriculum DATASET [extra synthesis args...]: the three standard stages.
build_curriculum() {
  local dataset="$1" stage
  shift
  for stage in easy hard extreme; do
    build_stage "${dataset}" "${stage}" "$(stage_objects "${stage}")" "$(stage_gamma "${stage}")" "$@"
  done
}

# train_curriculum EXPERIMENT DATASET ANNO_COMMENT [STAGE3] [BATCH_PER_GPU]
# Paper schedule; STAGE3 replaces "extreme" for the difficulty ablations.
train_curriculum() {
  local experiment="$1" dataset="$2" anno="$3" s3="${4:-extreme}" batch="${5:-2}"
  local num_gpus
  num_gpus="$(awk -F, '{print NF}' <<< "${GPUS}")"
  CUDA_VISIBLE_DEVICES="${GPUS}" bash "${LAOX_REPO}/sam2/training/train_curriculum.sh" "${experiment}" \
    --data-root-name "${dataset}" \
    --anno-comment "${anno}" \
    --stages "easy,hard,${s3}" \
    --stage-ranges "easy:0-4000,hard:0-6000,${s3}:0-8000" \
    --stage-base-lrs "easy:1.6e-4,hard:1.2e-4,${s3}:1e-4" \
    --stage-vision-lrs "easy:4.8e-6,hard:3.2e-6,${s3}:3e-6" \
    --stage-epochs "easy:80,hard:64,${s3}:48" \
    --batch-size "${batch}" \
    --num-gpus "${num_gpus}"
}
