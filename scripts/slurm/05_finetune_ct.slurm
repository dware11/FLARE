#!/bin/bash
#SBATCH --job-name=finetune_ct_mm
#SBATCH --account=bckk-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/bckk/flare/logs/finetune_ct_%j.out

set -euo pipefail

module purge
module load miniforge3-python
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/bckk/flare/envs/flare_ct
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/bckk/flare/projects/FLARE

mkdir -p /scratch/bckk/flare/logs
cd /scratch/bckk/flare/projects/FLARE

echo "=== FINETUNE ==="
date
nvidia-smi || true

python3 scripts/data/04_finetune_ct.py

echo "=== PLOT RESULTS ==="
date
python3 scripts/data/05_plot_results_ct.py

echo "=== DONE ==="
date
