#!/bin/bash
#SBATCH --job-name=flare_ensemble
#SBATCH --partition=gpuMI100x8
#SBATCH --account=bckk-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --constraint="scratch"
#SBATCH -o /scratch/bckk/hjaffara/slurm-ensemble-%j.out
#SBATCH -e /scratch/bckk/hjaffara/slurm-ensemble-%j.err

echo "====================================="
echo "Job starting on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "====================================="


mkdir -p /scratch/bckk/flare/mri_brain/segmentation/outputs/ensemble

/scratch/bckk/hjaffara/mri_env/bin/python \
    /scratch/bckk/flare/mri_brain/classification/ensemble_segmentation.py

echo "Job finished"
echo "====================================="
