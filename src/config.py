"""
Data paths for FLARE: shared DATA_ROOT works on both Windows and Delta.

- Set env DATA_ROOT to override (e.g. on Delta: /scratch/bckk/flare).
- Default (Windows): D:\\FLARE_DATA.
- CT data under DATA_ROOT/ct_brain; MRI under DATA_ROOT/mri_brain.
"""
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 1) Shared root (env or Windows default)
# -----------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"D:\FLARE_DATA"))

# -----------------------------------------------------------------------------
# 2) Modality roots
# Delta: DATA_ROOT=/scratch/bckk/flare -> CT_ROOT=/scratch/bckk/flare/ct_brain, etc.
# -----------------------------------------------------------------------------
CT_ROOT = DATA_ROOT / "ct_brain"
MRI_ROOT = DATA_ROOT / "mri_brain"

# -----------------------------------------------------------------------------
# 3) CT brain paths (explicit CT_* for future expansion)
# -----------------------------------------------------------------------------
CT_RAW = CT_ROOT / "raw"
CT_RAW_ZIPS = CT_ROOT / "raw_zips"
CT_CACHE = CT_ROOT / "cache"
CT_META = CT_ROOT / "meta"
CT_OUTPUTS = CT_ROOT / "outputs"

# CT cache subdir for NPZ files (cache/ct/<patient_id>/middle.npz)
CACHE_CT = CT_ROOT / "cache" / "ct"

# Convenience: manifest, checkpoint, Grad-CAM output dir
CT_MANIFEST = CT_META / "ct_processed_manifest.json"
CT_BEST_CKPT = CT_OUTPUTS / "ct_baseline_best.pt"
CT_CAM_DIR = CT_OUTPUTS / "cam"

# -----------------------------------------------------------------------------
# 4) MRI brain paths (explicit MRI_* for future expansion)
# -----------------------------------------------------------------------------
MRI_RAW = MRI_ROOT / "raw"
MRI_RAW_ZIPS = MRI_ROOT / "raw_zips"
MRI_CACHE = MRI_ROOT / "cache"
MRI_META = MRI_ROOT / "meta"
MRI_OUTPUTS = MRI_ROOT / "outputs"

# -----------------------------------------------------------------------------
# 5) Backward compatibility: existing names map to CT by default
#    (do not remove; other files import these)
# -----------------------------------------------------------------------------
RAW = CT_RAW
RAW_CT = CT_RAW
RAW_ZIPS = CT_RAW_ZIPS
CACHE = CT_CACHE
META = CT_META
OUTPUTS = CT_OUTPUTS
