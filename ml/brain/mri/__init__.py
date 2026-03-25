import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("FLARE_ROOT", "/scratch/bckk/flare/projects/FLARE"))
BRATS_GLI_TRAIN  = PROJECT_ROOT / "data/raw/BraTS24/GLI-Challenge-TrainingData"
BRATS_AFRICA     = PROJECT_ROOT / "data/raw/BraTS-Africa/95_Glioma"
BRATS_GLI_VAL    = PROJECT_ROOT / "data/raw/BraTS24/GLI-Challenge-ValidationData"
TRAINING_DATA_DIRS = [BRATS_GLI_TRAIN, BRATS_AFRICA]
SPLITS_DIR       = PROJECT_ROOT / "pre-processing/splits"
PREPROCESSED_DIR = PROJECT_ROOT / "data/processed"
TRAIN_RATIO = 0.70; VAL_RATIO = 0.15; TEST_RATIO = 0.15
RANDOM_SEED = 42; BATCH_SIZE = 1; NUM_WORKERS = 2
LEARNING_RATE = 1e-4; NUM_EPOCHS = 10