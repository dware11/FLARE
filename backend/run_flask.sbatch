#!/bin/bash
#SBATCH --job-name=flare_backend
#SBATCH --account=bckk-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/bckk/flare/projects/FLARE/backend/flask_%j.log

module purge
module load miniforge3-python
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/bckk/flare/envs/flare_backend
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/bckk/flare/projects/FLARE

export FLARE_MRI_CLS_CHECKPOINT="/scratch/bckk/flare/mri_brain/classification/outputs/output/best_model.pth"
export FLARE_MRI_SEG_CHECKPOINT="/scratch/bckk/flare/mri_brain/segmentation_brisc/outputs/swin_unet/best_model_swin.pth"
export FLARE_MRI_BRATS_CHECKPOINT="/scratch/bckk/flare/projects/FLARE/ml/brain/mri/checkpoints/best_seg_model.pth"

echo "Node: $(hostname)" >> /scratch/bckk/flare/projects/FLARE/backend/flask_node.txt
echo "Started: $(date)" >> /scratch/bckk/flare/projects/FLARE/backend/flask_node.txt

cd /scratch/bckk/flare/projects/FLARE
python backend/app.py --host 0.0.0.0
