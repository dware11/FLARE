from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]

# Delta-first defaults for DBT workflow.
DBT_DATA_ROOT = Path(os.environ.get("FLARE_DBT_DATA_ROOT", "/scratch/bckk/flare/mammo_breast"))
DBT_PROJECT_ROOT = Path(os.environ.get("FLARE_DBT_PROJECT_ROOT", "/scratch/bckk/flare/projects/FLARE"))

# Raw input roots by split/batch.
DBT_RAW_TRAIN_ROOT = Path(
    os.environ.get("FLARE_DBT_RAW_TRAIN", str(DBT_DATA_ROOT / "raw" / "train" / "batch1"))
)
DBT_RAW_VAL_ROOT = Path(
    os.environ.get("FLARE_DBT_RAW_VAL", str(DBT_DATA_ROOT / "raw" / "val" / "batch1"))
)
DBT_RAW_TEST_ROOT = Path(
    os.environ.get("FLARE_DBT_RAW_TEST", str(DBT_DATA_ROOT / "raw" / "test" / "batch1"))
)

# Backward-compatible single raw root used by preprocess/inference internals.
# Default points to train/batch1, and can be overridden per run.
RAW_DBT_ROOT = Path(os.environ.get("FLARE_DBT_RAW", str(DBT_RAW_TRAIN_ROOT)))

# Cache/manifest are still required by current preprocessing code.
DBT_CACHE_ROOT = Path(
    os.environ.get("FLARE_DBT_CACHE", str(DBT_DATA_ROOT / "cache" / "train"))
)
DBT_NPZ_DIR = DBT_CACHE_ROOT / "npz"
DBT_MANIFEST_PATH = DBT_CACHE_ROOT / "bcs_dbt_manifest.json"

# Delta outputs by split/batch.
DBT_OUTPUT_TRAIN_ROOT = Path(
    os.environ.get("FLARE_DBT_OUT_TRAIN", str(DBT_DATA_ROOT / "outputs" / "train" / "batch1"))
)
DBT_OUTPUT_VAL_ROOT = Path(
    os.environ.get("FLARE_DBT_OUT_VAL", str(DBT_DATA_ROOT / "outputs" / "val" / "batch1"))
)
DBT_OUTPUT_TEST_ROOT = Path(
    os.environ.get("FLARE_DBT_OUT_TEST", str(DBT_DATA_ROOT / "outputs" / "test" / "batch1"))
)

# Prediction CSV defaults.
DBT_PRED_ROOT = Path(os.environ.get("FLARE_DBT_PRED", str(DBT_OUTPUT_TEST_ROOT)))
DBT_EXAM_PRED_CSV = DBT_PRED_ROOT / "bcs_dbt_exam_predictions.csv"
DBT_VIEW_PRED_CSV = DBT_PRED_ROOT / "bcs_dbt_view_predictions.csv"

# Training artifact defaults.
DBT_TRAIN_ARTIFACT_ROOT = Path(
    os.environ.get("FLARE_DBT_TRAIN_ARTIFACTS", str(DBT_OUTPUT_TRAIN_ROOT))
)
DBT_BEST_CKPT_PATH = DBT_TRAIN_ARTIFACT_ROOT / "breast_dbt_best.pt"
DBT_TRAIN_METRICS_PATH = DBT_TRAIN_ARTIFACT_ROOT / "breast_dbt_train_metrics.json"

# Volume shapes and preprocessing settings.
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
DBT_CLAHE_TITLE_GRID_SIZE = (8, 8)
DBT_ENABLE_ROI_CROP = True

# Legacy random split defaults (used when training from one manifest).
DBT_TRAIN_FRACTION = 0.7
DBT_VAL_FRACTION = 0.15

# Training defaults.
DBT_DEFAULT_EPOCHS = 20
DBT_DEFAULT_BATCH_SIZE = 2
DBT_DEFAULT_LR = 1e-4
DBT_DEFAULT_WEIGHT_DECAY = 1e-4
DBT_DEFAULT_VIEW_AGG = "mean"  # "max" or "mean"
