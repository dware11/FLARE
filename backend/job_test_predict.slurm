#!/bin/bash
#SBATCH --job-name=test_predict
#SBATCH --partition=gpuA40x4
#SBATCH --account=bckk-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/bckk/hjaffara/slurm-test-predict-%j.out
#SBATCH --error=/scratch/bckk/hjaffara/slurm-test-predict-%j.err

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/bckk/hjaffara/mri_env

cd /scratch/bckk/flare/projects/FLARE/backend

python3 -c "
import json
from predict_mri import predict_mri

# Test BRISC
test_img = '/scratch/bckk/flare/mri_brain/data/raw/brisc2025/classification_task/test/pituitary/brisc2025_test_00776_pi_ax_t1.jpg'
print('[BRISC] Testing...')
result = predict_mri(test_img, modality='brain_mri', patient_id='TEST_001')
print(json.dumps(result, indent=2))

# Test BraTS
test_npz = '/scratch/bckk/flare/mri_brain/data/brats_processed/test/BraTS-SSA-00057-000.npz'
print('[BraTS] Testing...')
result = predict_mri(test_npz, modality='brain_brats', patient_id='TEST_BRATS_001')
print(json.dumps(result, indent=2))

print('DONE')
"
