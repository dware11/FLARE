#!/bin/bash
#SBATCH --job-name=ct_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=bckk-delta-gpu
# Batch GPU queue (use gpuA40x4-interactive with salloc for interactive shells).
#SBATCH -p gpuA40x4

set -eo pipefail

export DATA_ROOT=/scratch/bckk/flare/ct_brain
cd /scratch/bckk/flare/projects/FLARE

mkdir -p logs

module load miniforge3-python
ENV_PYTHON=/scratch/bckk/flare/envs/flare_ct/bin/python
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "ERROR: $ENV_PYTHON not found or not executable" >&2
  exit 1
fi

# DenseNet121 + BiGRU + max slice pooling + focal (gamma=2) + grad clip + AUC checkpointing (train_ct defaults).
# Re-run preprocess_ct.py --force after ct_transforms window changes. Then: threshold_sweep.py for operating point.
$ENV_PYTHON ml/brain/ct/train_ct.py \
  --epochs 30 \
  --batch-size 8 \
  --lr 5e-5 \
  --grad-clip 1.0 \
  --model sequence \
  --rnn-type gru \
  --sequence-pool max \
  --rnn-dropout 0.25 \
  --focal-gamma 2.0
