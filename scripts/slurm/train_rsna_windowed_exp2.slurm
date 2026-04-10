#!/bin/bash
#SBATCH --job-name=rsna_windowed_exp2
#SBATCH --account=bckk-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/bckk/flare/ct_brain/outputs/rsna_windowed_exp2_%j.log

module purge
module load miniforge3-python
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/bckk/flare/envs/flare_ct
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/bckk/flare/projects/FLARE
export DATA_ROOT=/scratch/bckk/flare/ct_brain
cd /scratch/bckk/flare/projects/FLARE

echo "=== START ==="
date
nvidia-smi

python3 scripts/train_rsna_subset.py \
  --train-manifest /scratch/bckk/flare/ct_brain/meta/ct_rsna_windowed_20k_train.json \
  --val-manifest   /scratch/bckk/flare/ct_brain/meta/ct_rsna_windowed_20k_val.json \
  --test-manifest  /scratch/bckk/flare/ct_brain/meta/ct_rsna_windowed_20k_test.json \
  --output-dir     /scratch/bckk/flare/ct_brain/outputs \
  --experiment-name rsna_windowed_exp2 \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 2 \
  --lr 3e-5 \
  --k-slices 21 \
  --focal-gamma 1.5 \
  --grad-clip 1.0 \
  --weight-decay 1e-4 \
  --imbalance-mode none \
  --auc-patience 5 \
  --tune-f1-on-val \
  --seed 42 \
  --checkpoint /scratch/bckk/flare/ct_brain/outputs/rsna_k21_balanced16k_exp1/best.pt

echo "=== DONE ==="
date
