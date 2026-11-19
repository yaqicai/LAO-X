SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
COMMENT="${1:-}"


DATASET="${DATASET:-det_compass_xp}"
START_ID="${START_ID:-0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0}"
# END_ID="${END_ID:-1928}"
if [[ -z "${END_ID}" ]]; then
  case "${DATASET}" in
    det_compass_xp) END_ID=307 ;;
    CLCXray) END_ID=956 ;;
    DvXray) END_ID=9155 ;;
    PIXray) END_ID=5047 ;;
    HiXray) END_ID=9069 ;;
    pidray) END_ID=18220 ;;
    *) END_ID=1928 ;;
  esac
fi
SAM2_PROFILE="${SAM2_PROFILE:-0}"
GPUS="${GPUS:-0}"
# GRAS="${GRAS:-0.9,1.0}"
GRAS="${GRAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
IFS=',' read -ra GRA_LIST <<< "${GRAS}"

NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs specified in GPUS." >&2
  exit 1
fi

if [[ -n "${COMMENT}" ]]; then
  CKPT_ROOT="results/train/det_compass_xp_${COMMENT}"
  COMMENT_SUFFIX="_det_compass_xp_${COMMENT}_m${MASK_THRESHOLD}"
else
  CKPT_ROOT="results/train/det_compass_xp"
  COMMENT_SUFFIX="_det_compass_xp"
fi

run_eval() {
  local gra="$1"
  local gpu="$2"
  local ckpt_path="${CKPT_ROOT}/log/checkpoints/checkpoint.pt"
  echo "[eval] dataset=${DATASET} gra=${gra} gpu=${gpu} ckpt=${ckpt_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    SAM2_PROFILE="${SAM2_PROFILE}" \
    python test_det_v2.py \
      --method unsamv2 \
      --dataset "${DATASET}" \
      --sam2-ckpt "${ckpt_path}" \
      --gra "${gra}" \
      --eval-mode bbox \
      --start_id "${START_ID}" \
      --end_id "${END_ID}" \
      --comment "${COMMENT_SUFFIX}" \
      --mask-threshold "${MASK_THRESHOLD}" \
      --device-id 0 &
}

job_count=0
idx=0
for gra in "${GRA_LIST[@]}"; do
  gpu="${GPU_LIST[$((idx % NUM_GPUS))]}"
  run_eval "${gra}" "${gpu}"
  idx=$((idx + 1))
  job_count=$((job_count + 1))
  if [[ "${job_count}" -ge "${NUM_GPUS}" ]]; then
    wait -n
    job_count=$((job_count - 1))
  fi
done
wait

# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/easy/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_easy 

# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard 
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/hard/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_hard

# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
# python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt ${CKPT_ROOT}/extreme/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment ${COMMENT_SUFFIX}_extreme
