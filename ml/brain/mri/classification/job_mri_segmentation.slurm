#!/bin/bash
#SBATCH --job-name=flare_mri_seg
#SBATCH --partition=gpuA40x4
#SBATCH --account=bckk-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --constraint="scratch"
#SBATCH -o /scratch/bckk/hjaffara/slurm-seg-%j.out
#SBATCH -e /scratch/bckk/hjaffara/slurm-seg-%j.err

echo "====================================="
echo "Job starting on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "====================================="

nvidia-smi

mkdir -p /scratch/bckk/flare/mri_brain/segmentation/outputs/training

/scratch/bckk/hjaffara/mri_env/bin/python \
    /scratch/bckk/flare/mri_brain/classification/train_segmentation.py

echo "====================================="
echo "Job finished"
echo "====================================="
