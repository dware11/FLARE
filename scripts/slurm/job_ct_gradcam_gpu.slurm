#!/bin/bash
#SBATCH --job-name=ct_gradcam
#SBATCH --output=logs/gradcam_%j.out
#SBATCH --error=logs/gradcam_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bckk-delta-gpu

set -eo pipefail 
 
export DATA_ROOT=/scratch/bckk/flare 
cd /scratch/bckk/flare/projects/FLARE 

# Load miniforge so conda is available, then activate env
module load miniforge3-python
conda activate /scratch/bckk/flare/envs/flare_ct

python - << 'PY'
import json
import random
import sys
from pathlib import Path

# Ensure repo root is on path (Slurm script has already cd'd here)
sys.path.insert(0, str(Path.cwd()))

from src.config import CT_MANIFEST, CT_BEST_CKPT, CT_CAM_DIR
from ml.brain.ct.infer import run_ct_for_patient

CT_CAM_DIR.mkdir(parents=True, exist_ok=True)

with open(CT_MANIFEST) as f:
    manifest = json.load(f)
patient_ids = [row["patient_id"] for row in manifest]

random.seed(0)
subset = random.sample(patient_ids, k=min(25, len(patient_ids)))

for pid in subset:
    out = run_ct_for_patient(pid, checkpoint=CT_BEST_CKPT, cam_dir=CT_CAM_DIR)
    if out:
        print(pid, out.get("label"), out.get("confidence"), out.get("cam_path"))
    else:
        print(pid, "FAIL")
PY 