#!/bin/bash
#SBATCH --job-name=test_fusion
#SBATCH --account=bckk-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/bckk/flare/projects/FLARE/logs/test_fusion_%j.log

module purge
module load miniforge3-python
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/bckk/flare/envs/flare_ct
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/bckk/flare/projects/FLARE
export DATA_ROOT=/scratch/bckk/flare/ct_brain
export FLARE_MRI_CLS_CHECKPOINT="/scratch/bckk/flare/mri_brain/classification/outputs/output/best_model.pth"
export FLARE_MRI_SEG_CHECKPOINT="/scratch/bckk/flare/mri_brain/segmentation_brisc/outputs/swin_unet/best_model_swin.pth"
export FLARE_MRI_BRATS_CHECKPOINT="/scratch/bckk/flare/projects/FLARE/ml/brain/mri/checkpoints/best_seg_model.pth"
cd /scratch/bckk/flare/projects/FLARE

mkdir -p logs

echo "=== TESTING CT INFERENCE DIRECTLY FROM NPZ ==="
python3 -c "
import torch
import numpy as np
from pathlib import Path
from ml.brain.ct.infer import _load_model, _load_patient_arr, _imagenet_normalize_volume, _predict_one, _thickness_from_npz

CKPT = Path('/scratch/bckk/flare/ct_brain/outputs/rsna_windowed_exp2/best.pt')
NPZ  = Path('/scratch/bckk/flare/ct_brain/cache/rsna_windowed/ID_1b44dc3cc/slices_k21.npz')
PATIENT_ID = 'ID_1b44dc3cc'

print(f'Loading checkpoint: {CKPT}')
model, model_kind, agg, k_slices_meta, ckpt_meta = _load_model(CKPT)
device = next(model.parameters()).device
print(f'Model loaded on: {device}')

arr = _load_patient_arr(NPZ, expected_k=21)
x_raw = torch.from_numpy(arr).float().to(device)
x_all = _imagenet_normalize_volume(x_raw)
x_batch = x_all.unsqueeze(0)
thick = _thickness_from_npz(NPZ, device)

probs = _predict_one(model, x_batch, use_grad=False, agg=agg, thickness=thick)
ct_prob = float(probs[1])
threshold = float(ckpt_meta.get('eval_threshold', 0.488))
pred = 'Abnormal' if ct_prob >= threshold else 'Normal'

print()
print('=== CT RESULT ===')
print(f'patient_id:   {PATIENT_ID}')
print(f'p_normal:     {float(probs[0]):.4f}')
print(f'p_abnormal:   {ct_prob:.4f}')
print(f'threshold:    {threshold}')
print(f'prediction:   {pred}')
print(f'true_label:   Abnormal (label=1)')

mri_prob = 0.82
CT_WEIGHT = 0.6
MRI_WEIGHT = 0.4
fusion_score = (ct_prob * CT_WEIGHT) + (mri_prob * MRI_WEIGHT)

print()
print('=== FUSION RESULT (mocked MRI) ===')
print(f'ct_prob:      {ct_prob:.4f}')
print(f'mri_prob:     {mri_prob:.4f} (mocked)')
print(f'fusion_score: {fusion_score:.4f}')
print(f'prediction:   {\"Abnormal\" if fusion_score >= 0.5 else \"Normal\"}')
"

echo ""
echo "=== TESTING MRI SIDE ==="
python3 -c "
import sys
sys.path.insert(0, 'backend')
from predict_mri import predict_mri

mri_result = predict_mri(
    '/scratch/bckk/flare/projects/FLARE/ml/brain/mri/processed/test/BraTS-GLI-00024-001.npz',
    'brain_brats',
    'TEST_MRI_001'
)
print('pred_label:  ', mri_result.get('pred_label'))
print('result_class:', mri_result.get('result_class'))
print('confidence:  ', mri_result.get('confidence'))
"

echo "=== DONE ==="
