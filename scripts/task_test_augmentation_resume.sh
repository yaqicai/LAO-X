SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/easy/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_easy
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/easy/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_easy 
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/easy/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_easy 
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/easy/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_easy 
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/easy/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_easy 

python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/hard/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_hard
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/hard/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_hard
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/hard/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_hard 
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/hard/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_hard
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/hard/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_hard

python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/extreme/log/checkpoints/checkpoint.pt --gra 0.15 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_extreme
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/extreme/log/checkpoints/checkpoint.pt --gra 0.3 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_extreme
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/extreme/log/checkpoints/checkpoint.pt --gra 0.5 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_extreme
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/extreme/log/checkpoints/checkpoint.pt --gra 0.7 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_extreme
python test_det_v2.py --method unsamv2 --dataset det_compass_syn_100 --sam2-ckpt results/train/augmentation_resume/extreme/log/checkpoints/checkpoint.pt --gra 1.0 --eval-mode bbox --start_id 0 --end_id 100 --comment augmentation_resume_extreme
