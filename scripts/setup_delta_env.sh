#!/bin/bash
# One-time setup on Delta: create flare_ct env and install deps.
# Run this on the login node (SSH) where you can use conda interactively.
# Usage: bash scripts/setup_delta_env.sh

set -e

ENV_PREFIX=/scratch/bckk/flare/envs/flare_ct
REPO_ROOT=/scratch/bckk/flare/projects/FLARE

echo "Loading miniforge..."
module load miniforge3-python

# Initialize conda for this shell (needed for conda create/activate in script)
CONDA_BASE=$(conda info --base)
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found under $CONDA_BASE"
  exit 1
fi

echo "Creating env at $ENV_PREFIX ..."
conda create --prefix "$ENV_PREFIX" python=3.9 -y

echo "Activating and installing packages..."
conda activate "$ENV_PREFIX"
pip install torch
cd "$REPO_ROOT"
pip install -r requirements.txt

echo "Verifying..."
"$ENV_PREFIX/bin/python" -c "import torch; print('torch', torch.__version__)"

echo "Done. Use sbatch scripts/slurm/job_ct_train_gpu.slurm to run training."
