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

# 3. Activate environment 
source ~/.bashrc 
conda activate /scratch/bckk/flare/envs/flare_ct

# 4. Run preprocess (writers CT_MANIFEST + CACHE_CT)
python scripts/preprocess_ct.py --limit 150