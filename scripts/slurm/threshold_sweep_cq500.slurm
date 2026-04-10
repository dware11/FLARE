#!/bin/bash
#SBATCH --job-name=cq500_sweep
#SBATCH --account=bckk-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/bckk/flare/ct_brain/outputs/cq500_sweep_%j.log

module purge
module load miniforge3-python
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/bckk/flare/envs/flare_ct
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/bckk/flare/projects/FLARE
export DATA_ROOT=/scratch/bckk/flare/ct_brain
cd /scratch/bckk/flare/projects/FLARE

CKPT=/scratch/bckk/flare/ct_brain/outputs/rsna_windowed_exp2/best.pt
MANIFEST=/scratch/bckk/flare/ct_brain/meta/ct_processed_manifest_k21.json

for t in 0.25 0.30 0.35 0.40 0.45 0.50; do
    echo "=== threshold $t ==="
    python3 eval_ct.py \
        --model densenet_gru \
        --checkpoint $CKPT \
        --dataset cq500 \
        --manifest $MANIFEST \
        --output-dir results/densenet_gru_cq500_exp2_t${t} \
        --threshold $t
    echo ""
done

echo "=== SWEEP DONE ==="
