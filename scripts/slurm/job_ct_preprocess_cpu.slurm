#!/bin/bash
#SBATCH --job-name=ct_preprocess
#SBATCH --output=logs/ct_preprocess_%j.out
#SBATCH --error=logs/ct_preprocess_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=bckk-delta-gpu
#SBATCH -p cpu

set -eo pipefail

export DATA_ROOT=/scratch/bckk/flare/ct_brain
cd /scratch/bckk/flare/projects/FLARE

mkdir -p logs

# Load miniforge; use env's Python directly so we don't rely on conda activate in batch
module load miniforge3-python
ENV_PYTHON=/scratch/bckk/flare/envs/flare_ct/bin/python
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "ERROR: $ENV_PYTHON not found or not executable" >&2
  exit 1
fi

$ENV_PYTHON scripts/preprocess_ct.py --link-selected-series

#!/bin/bash 

#SBATCH --job-name=ct_preprocess
#SBATCH --output=logs/preprocess_%j.output
#SBATCH --error=logs/preprocess_%j.err 
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu 
#SBATCH --account=bckk-delta-cpu 

set -eo pipefail 

# 1. Shared root (NOT ct_brain) 
export DATA_ROOT=/scratch/bckk/flare

# 2. Go to repo root 
cd /scratch/bckk/flare/projects/FLARE

# 3. Load miniforge so conda is available, then activate env
module load miniforge3-python
conda activate /scratch/bckk/flare/envs/flare_ct

# 4. Run preprocess (writers CT_MANIFEST + CACHE_CT)
python scripts/preprocess_ct.py --limit 150