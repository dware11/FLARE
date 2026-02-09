from pathlib import Path

DATA_ROOT = Path(r"D:\FLARE_DATA\ct_brain")

RAW_ZIPS = DATA_ROOT / "raw_zips"   # has the study zip folders/files
RAW      = DATA_ROOT / "raw"        # has extracted patients 
CACHE    = DATA_ROOT / "cache"
OUTPUTS  = DATA_ROOT / "outputs"
META     = DATA_ROOT / "meta"       # metadata lives WITH the data
