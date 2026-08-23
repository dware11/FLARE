#!/bin/bash
# ============================================================
# FLARE Backend Startup Script
# Run this on Delta to start Flask on a GPU node
# Usage: bash start_flare.sh
# ============================================================

echo "============================================"
echo "  FLARE Backend — Starting on Delta GPU"
echo "============================================"

# ── Load environment ──────────────────────────────────────
module purge
module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate flare_backend
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/bckk/flare/projects/FLARE

# ── Checkpoint paths ──────────────────────────────────────
export FLARE_MRI_CLS_CHECKPOINT="/scratch/bckk/flare/mri_brain/classification/outputs/training_5class/best_model.pth"
export FLARE_MRI_SEG_CHECKPOINT="/scratch/bckk/flare/mri_brain/segmentation_brisc/outputs/swin_unet/best_model_swin.pth"
export FLARE_MRI_BRATS_CHECKPOINT="/scratch/bckk/flare/projects/FLARE/ml/brain/mri/checkpoints/best_seg_model.pth"
# ── Print hostname so you know what to tunnel to ──────────
echo ""
echo "  Node hostname: $(hostname)"
echo ""
echo "  On your laptop run:"
echo "  ssh -N -L 5000:$(hostname):5000 dware@login.delta.ncsa.illinois.edu"
echo ""
echo "============================================"

# ── Start Flask ───────────────────────────────────────────
cd /scratch/bckk/flare/projects/FLARE/backend
python app.py --host 0.0.0.0
