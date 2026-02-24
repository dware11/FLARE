"""Data paths for CT pipeline: raw, cache, outputs, meta (all under DATA_ROOT)."""
from pathlib import Path

DATA_ROOT = Path(r"D:\FLARE_DATA\ct_brain")

RAW_ZIPS = DATA_ROOT / "raw_zips"
RAW = DATA_ROOT / "raw"
RAW_CT = RAW
# Demo: CACHE_CT = cache/ct/<patient_id>/middle.npz; META = manifest + reads.
CACHE = DATA_ROOT / "cache"
CACHE_CT = CACHE / "ct"
OUTPUTS = DATA_ROOT / "outputs"
META = DATA_ROOT / "meta"
