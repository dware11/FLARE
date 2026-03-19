#!/bin/bash
#SBATCH --job-name=mammo_3d_train
#SBATCH --account=bckk
#SBATCH --partition=gpuA100x1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/mammo_train_%j.out
#SBATCH --error=logs/mammo_train_%j.err

module purge
module load anaconda3_gpu
conda activate flare

cd $SLURM_SUBMIT_DIR

echo "=== 3D DBT Mammography Training ==="
echo "Node: $SLURMD_NODENAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python ml/breast/mammo/train_mammo.py \
    --epochs 30 \
    --batch-size 4 \
    --lr 1e-4
