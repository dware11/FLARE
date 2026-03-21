"""Breast MRI branch config (EA1141 / TCIA-style). Env overrides for Delta."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[3]

try:
    from src.config import DATA, OUTPUTS
except ImportError:
    DATA = ROOT / "data"
    OUTPUTS = ROOT / "outputs"

BREAST_MRI_ROOT = Path(os.environ.get("FLARE_BREAST_MRI_ROOT", str(DATA / "breast_mri")))
BREAST_MRI_RAW = Path(os.environ.get("FLARE_BREAST_MRI_RAW", str(BREAST_MRI_ROOT / "raw")))
BREAST_MRI_CACHE = Path(os.environ.get("FLARE_BREAST_MRI_CACHE", str(BREAST_MRI_ROOT / "cache")))
BREAST_MRI_META = Path(os.environ.get("FLARE_BREAST_MRI_META", str(BREAST_MRI_ROOT / "meta")))
BREAST_MRI_OUTPUTS = Path(os.environ.get("FLARE_BREAST_MRI_OUTPUTS", str(BREAST_MRI_ROOT / "outputs")))
BREAST_MRI_CKPT = Path(os.environ.get("FLARE_BREAST_MRI_CKPT", str(BREAST_MRI_OUTPUTS / "checkpoints")))
BREAST_MRI_CAM_DIR = Path(os.environ.get("FLARE_BREAST_MRI_CAM_DIR", str(BREAST_MRI_OUTPUTS / "gradcam_mri")))

BREAST_MRI_NPZ_DIR = BREAST_MRI_CACHE / "npz"
BREAST_MRI_MANIFEST_PATH = BREAST_MRI_META / "mri_ea1141_manifest.csv"
BREAST_MRI_PRED_CSV = BREAST_MRI_OUTPUTS / "mri_exam_predictions.csv"
BREAST_MRI_TRAIN_METRICS_PATH = BREAST_MRI_OUTPUTS / "mri_train_metrics.json"

MRI_TARGET_SHAPE: Tuple[int, int, int] = (32, 128, 128)
MRI_NUM_CHANNELS: int = 3
MRI_ADD_SUBTRACTION: bool = False

MRI_NORM_LO_PCT: float = 1.0
MRI_NORM_HI_PCT: float = 99.0

MRI_ROI_FOREGROUND_THRESHOLD: float = 0.08
MRI_ROI_PADDING_FRAC: float = 0.08
MRI_ROI_MIN_FRACTION: float = 0.02
MRI_ROI_REFERENCE: str = "post1"  # or "channel_max"
MRI_CENTER_CROP_FRAC: float = 0.75

MRI_PRE_KEYWORDS: List[str] = [
    "pre",
    "non-contrast",
    "noncontrast",
    "unenhanced",
    "precontrast",
    "t0",
]
MRI_POST1_KEYWORDS: List[str] = ["post1", "post", "early", "dyn1", "phase1", "enhanced", "+c"]
MRI_POST2_KEYWORDS: List[str] = ["post2", "late", "delayed", "dyn2", "phase2"]

MRI_DEFAULT_EPOCHS: int = 30
MRI_DEFAULT_BATCH_SIZE: int = 2
MRI_DEFAULT_LR: float = 1e-4
MRI_DEFAULT_WEIGHT_DECAY: float = 1e-4
MRI_DEFAULT_VAL_FRAC: float = 0.15
MRI_DEFAULT_SEED: int = 42
MRI_PREPROC_VERSION: str = "v1"


def mri_full_target_shape(
    num_channels: int | None = None, spatial: Tuple[int, int, int] | None = None
) -> Tuple[int, int, int, int]:
    c = MRI_NUM_CHANNELS if num_channels is None else num_channels
    d, h, w = MRI_TARGET_SHAPE if spatial is None else spatial
    return (c, d, h, w)


def ensure_mri_dirs() -> None:
    for p in (
        BREAST_MRI_NPZ_DIR,
        BREAST_MRI_META,
        BREAST_MRI_OUTPUTS,
        BREAST_MRI_CKPT,
        BREAST_MRI_CAM_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)
