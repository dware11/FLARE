#!/bin/bash
#SBATCH --job-name=eval_exp2
#SBATCH --account=bckk-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/bckk/flare/ct_brain/outputs/eval_exp2_%j.log

module purge
module load miniforge3-python
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/bckk/flare/envs/flare_ct
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/bckk/flare/projects/FLARE
export DATA_ROOT=/scratch/bckk/flare/ct_brain
cd /scratch/bckk/flare/projects/FLARE

EXP2=/scratch/bckk/flare/ct_brain/outputs/rsna_windowed_exp2/best.pt

echo "=== CQ500 EVAL — exp2 ==="
python3 eval_ct.py \
  --model densenet_gru \
  --checkpoint $EXP2 \
  --dataset cq500 \
  --manifest /scratch/bckk/flare/ct_brain/meta/ct_processed_manifest_k21.json \
  --output-dir results/densenet_gru_cq500_exp2 \
  --threshold 0.488

echo "=== RSNA EVAL — exp2 ==="
python3 eval_ct.py \
  --model densenet_gru \
  --checkpoint $EXP2 \
  --dataset rsna \
  --manifest /scratch/bckk/flare/ct_brain/meta/ct_rsna_windowed_20k_test.json \
  --output-dir results/densenet_gru_rsna_exp2 \
  --threshold 0.488

echo "=== DONE ==="
