#!/bin/bash
#SBATCH --job-name=ct_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=bckk-delta-gpu
#SBATCH -p gpuA40x4-interactive

set -eo pipefail

export DATA_ROOT=/scratch/bckk/flare/ct_brain
cd /scratch/bckk/flare/projects/FLARE

# Load miniforge; use env's Python directly so we don't rely on conda activate in batch
module load miniforge3-python
ENV_PYTHON=/scratch/bckk/flare/envs/flare_ct/bin/python
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "ERROR: $ENV_PYTHON not found or not executable" >&2
  exit 1
fi

# If Delta requires a module load for GPU, add it here ex.: 
# module load cuda 

$ENV_PYTHON ml/brain/ct/train_ct.py --epochs 5 --batch-size 16 --lr 1e-4

