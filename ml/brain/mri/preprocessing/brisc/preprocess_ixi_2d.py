"""
IXI PREPROCESSING - Adaptive Slice Selection from 3D Volumes
Finds and extracts the slice with maximum brain tissue content
Output: /scratch/bckk/flare/mri_brain/data/ixi_processed/
"""

import os
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

IXI_DIR = Path('/scratch/bckk/flare/mri_brain/data/raw/mri_normal/ixi')
OUTPUT_DIR = Path('/scratch/bckk/flare/mri_brain/data/ixi_processed')
TARGET_SIZE = (512, 512)
NORMAL_CLASS_LABEL = 4
NORMAL_CLASS_NAME = 'normal'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_nifti(filepath):
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        return data
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None

def normalize_volume(volume):
    vol_min = volume.min()
    vol_max = volume.max()
    
    if vol_max == vol_min:
        return np.zeros_like(volume, dtype=np.uint8)
    
    normalized = (volume - vol_min) / (vol_max - vol_min) * 255.0
    return normalized.astype(np.uint8)

def find_best_axial_slice(volume):
    """
    Find axial slice with maximum brain tissue content
    Returns: slice index with highest non-zero pixel sum
    """
    # Sum non-zero pixels along axial axis
    slice_scores = np.zeros(volume.shape[0])
    
    for i in range(volume.shape[0]):
        slice_2d = volume[i, :, :]
        # Score = sum of intensity (more brain tissue = higher score)
        slice_scores[i] = np.sum(slice_2d)
    
    # Find slice with maximum brain content
    best_idx = np.argmax(slice_scores)
    return best_idx, slice_scores[best_idx]

def resize_slice(slice_2d, target_size):
    resized = cv2.resize(slice_2d, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    return resized

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def main():
    logger.info("="*70)
    logger.info("IXI PREPROCESSING - ADAPTIVE SLICE SELECTION")
    logger.info("="*70)
    logger.info(f"Input:  {IXI_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Target size: {TARGET_SIZE}")
    logger.info("Slice selection: Axial slice with maximum brain tissue")
    logger.info("="*70 + "\n")
    
    nifti_files = sorted(IXI_DIR.glob("**/T1/NIfTI/*T1.nii.gz"))
    logger.info(f"Found {len(nifti_files)} T1 NIfTI files\n")
    
    if len(nifti_files) == 0:
        logger.error("No T1 NIfTI files found!")
        return
    
    total_slices = 0
    failed = 0
    
    for nifti_path in tqdm(nifti_files, desc="Processing volumes"):
        filename = nifti_path.stem.replace('.nii', '')
        try:
            subject_id = int(filename.split('-')[0].replace('IXI', ''))
        except:
            logger.warning(f"Could not parse subject ID from {filename}")
            failed += 1
            continue
        
        # Load 3D volume
        volume = load_nifti(str(nifti_path))
        if volume is None:
            failed += 1
            continue
        
        # Normalize
        volume_norm = normalize_volume(volume)
        
        # Find best axial slice (adaptive selection)
        best_idx, score = find_best_axial_slice(volume_norm)
        slice_2d = volume_norm[best_idx, :, :]
        
        if len(slice_2d.shape) == 3:
            slice_2d = slice_2d[:, :, 0]
        
        # Resize
        slice_resized = resize_slice(slice_2d, TARGET_SIZE)
        
        # Save NPZ
        filename_npz = f"classification_train_ixi_{subject_id:05d}.npz"
        filepath_npz = OUTPUT_DIR / filename_npz
        
        np.savez_compressed(
            filepath_npz,
            image=slice_resized.astype(np.uint8),
            label=NORMAL_CLASS_LABEL,
            view='axial',
            tumor_label=NORMAL_CLASS_NAME,
            dataset='IXI',
            subject_id=subject_id,
            split='train',
            slice_index=best_idx,
            slice_score=float(score)
        )
        
        total_slices += 1
    
    logger.info("\n" + "="*70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Volumes processed: {len(nifti_files) - failed}/{len(nifti_files)}")
    logger.info(f"Total slices saved: {total_slices}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*70)
    
    npz_files = list(OUTPUT_DIR.glob("classification_*.npz"))
    logger.info(f"\nVerification: {len(npz_files)} NPZ files in output")

if __name__ == "__main__":
    main()
