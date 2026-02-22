"""Data paths for CT pipeline: raw, cache, outputs, meta (all under DATA_ROOT)."""
from pathlib import Path

DATA_ROOT = Path(r"D:\FLARE_DATA\ct_brain")

RAW_ZIPS = DATA_ROOT / "raw_zips"   # has the study zip folders/files
RAW      = DATA_ROOT / "raw" 
RAW_CT = RAW    # has extracted patients 
CACHE    = DATA_ROOT / "cache"
CACHE_CT = CACHE / "ct" 
OUTPUTS  = DATA_ROOT / "outputs"
META     = DATA_ROOT / "meta"       # metadata lives WITH the data
