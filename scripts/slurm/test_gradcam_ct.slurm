#!/bin/bash
#SBATCH --job-name=test_gradcam
#SBATCH --account=bckk-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/bckk/flare/projects/FLARE/logs/test_gradcam_%j.log

module purge
module load miniforge3-python
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/bckk/flare/envs/flare_ct
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/bckk/flare/projects/FLARE
export DATA_ROOT=/scratch/bckk/flare/ct_brain
cd /scratch/bckk/flare/projects/FLARE

mkdir -p logs
mkdir -p backend/cam

echo "=== TESTING CT GRADCAM ==="
python3 -c "
import torch
import numpy as np
from pathlib import Path
from ml.brain.ct.infer import (
    _load_model, _load_patient_arr, _imagenet_normalize_volume,
    _predict_one, _thickness_from_npz, _gradcam_layer_name,
    _save_overlay_png, GradCAM
)
from ml.brain.ct.model_ct import is_sequence_ct_model

CKPT = Path('/scratch/bckk/flare/ct_brain/outputs/rsna_windowed_exp2/best.pt')
NPZ  = Path('/scratch/bckk/flare/ct_brain/cache/rsna_windowed/ID_1b44dc3cc/slices_k21.npz')
CAM_DIR = Path('backend/cam')
PATIENT_ID = 'ID_1b44dc3cc'

print('Loading model...')
model, model_kind, agg, k_slices_meta, ckpt_meta = _load_model(CKPT)
device = next(model.parameters()).device

arr = _load_patient_arr(NPZ, expected_k=21)
x_raw = torch.from_numpy(arr).float().to(device)
x_all = _imagenet_normalize_volume(x_raw)
x_batch = x_all.unsqueeze(0)
thick = _thickness_from_npz(NPZ, device)

probs = _predict_one(model, x_batch, use_grad=True, agg=agg, thickness=thick)
ct_prob = float(probs[1])
pred = 1 if ct_prob >= 0.488 else 0

print(f'p_abnormal: {ct_prob:.4f}')
print(f'prediction: {\"Abnormal\" if pred==1 else \"Normal\"}')

# GradCAM
print('Generating GradCAM...')
gradcam = GradCAM(model, _gradcam_layer_name(model))
k = x_raw.shape[0]
center_raw = x_raw[k // 2 : k // 2 + 1]

if is_sequence_ct_model(model):
    x_cam = x_all[k // 2 : k // 2 + 1].unsqueeze(0)
else:
    x_cam = x_all[k // 2 : k // 2 + 1]

_, overlay = gradcam(x_cam, target_class=pred, input_for_overlay=center_raw)

if overlay is not None:
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    cam_path = CAM_DIR / f'{PATIENT_ID}.png'
    saved = _save_overlay_png(overlay, cam_path)
    if saved:
        print(f'GradCAM saved: {cam_path}')
        print(f'File size: {cam_path.stat().st_size} bytes')
    else:
        print('GradCAM save failed')
else:
    print('GradCAM overlay is None')

print()
print('=== DONE ===')
"
