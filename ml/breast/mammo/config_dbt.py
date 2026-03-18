from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]

try:
    from src.config import DATA, OUTPUTS
except ImportError:
    DATA = ROOT / "data"
    OUTPUTS = ROOT / "outputs"

RAW_DBT_ROOT = Path(
    os.environ.get("FLARE_DBT_RAW", str(DATA / "bcs_dbt_raw"))
)

DBT_CACHE_ROOT = Path(
    os.environ.get("FLARE_DBT_CACHE", str(OUTPUTS / "breast_dbt_cache"))
)

DBT_NPZ_DIR = DBT_CACHE_ROOT / "npz"

DBT_MANIFEST_PATH = DBT_CACHE_ROOT / "bcs_dbt_manifest.json"

DBT_PRED_ROOT = Path(
    os.environ.get("FLARE_DBT_PRED", str(OUTPUTS / "breast_dbt_predictions"))
)

DBT_EXAM_PRED_CSV = DBT_PRED_ROOT / "bcs_dbt_exam_predictions.csv"

DBT_VIEW_PRED_CSV = DBT_PRED_ROOT / "bcs_dbt_view_predictions.csv" 

# VOLUME SHAPES & PREPROCESSING SETTINGS 

DBT_TARGET_CHANNELS = 1
DBT_TARGET_DEPTH = 32
DBT_TARGET_HEIGHT = 256
DBT_TARGET_WIDTH = 256

DBT_TARGET_SHAPE = (
    DBT_TARGET_CHANNELS,
    DBT_TARGET_DEPTH,
    DBT_TARGET_HEIGHT,
    DBT_TARGET_WIDTH,
)

DBT_NORMALIZE_TO_UNIT = True 
DBT_APPLY_CLAHE = True
DBT_CLAHE_CLIP_LIMIT = 2.0 
DBT_CLAHE_TITLE_GRID_SIZE = (8,8)

DBT_ENABLE_ROI_CROP = True

# TRAIN/VAL/TEST SPLIT & TRAINING DEFAULTS 

DBT_TRAIN_FRACTION = 0.7 
DBT_VAL_FRACTION = 0.15 

DBT_DEFAULT_EPOCHS = 20 
DBT_DEFAULT_BATCH_SIZE = 2 
DBT_DEFAULT_LR = 1e-4
DBT_DEFAULT_WEIGHT_DECAY = 1e-4

DBT_DEFAULT_VIEW_AGG = "mean" # "max" or "mean" 



