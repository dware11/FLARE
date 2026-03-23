#!/bin/bash
#SBATCH --job-name=ct_preprocess_k15
#SBATCH --output=logs/ct_preprocess_%j.out
#SBATCH --error=logs/ct_preprocess_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=bckk-delta-cpu
#SBATCH -p cpu

set -eo pipefail

export DATA_ROOT=/scratch/bckk/flare/ct_brain
cd /scratch/bckk/flare/projects/FLARE

mkdir -p logs

module load miniforge3-python
ENV_PYTHON=/scratch/bckk/flare/envs/flare_ct/bin/python
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "ERROR: $ENV_PYTHON not found" >&2
  exit 1
fi

$ENV_PYTHON scripts/preprocess_ct.py \
  --k-slices 15 \
  --labels /scratch/bckk/flare/ct_brain/meta/reads.csv \
  --link-selected-series \
  --force