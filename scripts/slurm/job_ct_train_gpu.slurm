#!/bin/bash
#SBATCH --job-name=ct_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bckk-delta-gpu

set -eo pipefail 

export DATA_ROOT=/scratch/bckk/flare
cd /scratch/bckk/flare/projects/FLARE
echo "FLARE_DEBUG: DATA_ROOT=$DATA_ROOT" >&2
echo "FLARE_DEBUG: PWD=$(pwd)" >&2

# Load miniforge (for any system libs); use env's Python directly so we don't rely on conda activate in batch
module load miniforge3-python
ENV_PYTHON=/scratch/bckk/flare/envs/flare_ct/bin/python
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "FLARE_DEBUG: ERROR $ENV_PYTHON not found or not executable" >&2
  exit 1
fi
echo "FLARE_DEBUG: using $ENV_PYTHON" >&2
$ENV_PYTHON --version >&2

# If Delta requires a module load for GPU, add it here ex.: 
# module load cuda 

$ENV_PYTHON ml/brain/ct/train_ct.py --epochs 5 --batch-size 16 --lr 1e-4

