#!/bin/bash
#SBATCH --job-name=organize_preprocess
#SBATCH --account=bckk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/bckk/flare/logs/organize_preprocess_%j.out

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

echo "=== ORGANIZE ==="
date
python3 scripts/data/02_organize_multimodal.py

echo "=== PREPROCESS CT ==="
date
python3 scripts/data/03_preprocess_ct.py

echo "=== DONE ==="
date
