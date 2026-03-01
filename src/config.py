"""
Data paths for FLARE: shared DATA_ROOT works on both Windows and Delta.

Canonical roots:
  Windows (default):  DATA_ROOT = D:\\FLARE_DATA
                      -> CT: D:\\FLARE_DATA\\ct_brain,  MRI: D:\\FLARE_DATA\\mri_brain
  Delta (set in Slurm/shell):  export DATA_ROOT=/scratch/bckk/flare
                      -> CT: /scratch/bckk/flare/ct_brain,  MRI: /scratch/bckk/flare/mri_brain
"""
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 1) Shared root: env DATA_ROOT if set, else Windows default
# -----------------------------------------------------------------------------
_data_root = Path(os.environ.get("DATA_ROOT", r"D:\FLARE_DATA"))
# When DATA_ROOT is set to ct_brain (e.g. /scratch/.../ct_brain), use it as CT_ROOT directly
if _data_root.name == "ct_brain":
    DATA_ROOT = _data_root
    CT_ROOT = DATA_ROOT
    MRI_ROOT = _data_root.parent / "mri_brain"
else:
    DATA_ROOT = _data_root
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
