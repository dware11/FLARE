#!/bin/bash
#SBATCH --job-name=ct_preprocess_rsna_k21
#SBATCH --output=logs/ct_preprocess_rsna_%j.out
#SBATCH --error=logs/ct_preprocess_rsna_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=bckk-delta-cpu
#SBATCH -p cpu

set -eo pipefail

# Requires a FLARE checkout that includes RSNA in scripts/preprocess_ct.py (git pull on login if needed).
export DATA_ROOT=/scratch/bckk/flare/ct_brain
export PYTHONUNBUFFERED=1
cd /scratch/bckk/flare/projects/FLARE

mkdir -p logs /scratch/bckk/flare/ct_brain/outputs

module load miniforge3-python
ENV_PYTHON=/scratch/bckk/flare/envs/flare_ct/bin/python
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "ERROR: $ENV_PYTHON not found" >&2
  exit 1
fi

$ENV_PYTHON scripts/preprocess_rsna.py \
  --k-slices 21 \
  --rsna-manifest /scratch/bckk/flare/ct_brain/meta/ct_rsna_manifest.json
