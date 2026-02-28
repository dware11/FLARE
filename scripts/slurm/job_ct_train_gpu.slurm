#!/bin/bash
#SBATCH --job-name=ct_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=bckk-delta-gpu

set -eo pipefail 

export DATA_ROOT=/scratch/bckk/flare
cd /scratch/bckk/flare/projects/FLARE

source ~/.bashrc 
conda activate /scratch/bckk/flare/envs/flare_ct

# If Delta requires a module load, add it here ex.: 
# module load cuda 

python ml/brain/ct/train_ct.py --epochs 5 --batch-size 16 --lr 1e-4

