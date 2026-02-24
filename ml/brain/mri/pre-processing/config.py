"""
Configuration file for BraTS project
"""

from pathlib import Path
import os


# Your project root
PROJECT_ROOT = Path('/Users/huda/Desktop/Flare_brain_mri')
# Allow override via environment variable while keeping the original default
PROJECT_ROOT = Path(os.getenv("FLARE_MRI_ROOT", str(PROJECT_ROOT)))

BRATS_GLI_TRAIN = PROJECT_ROOT / 'data/raw/BraTS24/GLI-Challenge-TrainingData'
BRATS_AFRICA = PROJECT_ROOT / 'data/raw/BraTS-Africa/95_Glioma'

# Validation data 
BRATS_GLI_VAL = PROJECT_ROOT / 'data/raw/BraTS24/GLI-Challenge-ValidationData'

# Which datasets to use for creating train/val/test splits
# List of all folders we want to include
TRAINING_DATA_DIRS = [
    BRATS_GLI_TRAIN,
    BRATS_AFRICA,
]

# Splits directory
SPLITS_DIR = PROJECT_ROOT / 'pre-processing/splits'

# Preprocessed output directory
PREPROCESSED_DIR = PROJECT_ROOT / 'data/processed'

# Dataset split ratios
TRAIN_RATIO = 0.70  # 70% training
VAL_RATIO = 0.15    # 15% validation
TEST_RATIO = 0.15   # 15% testing

# Random seed for reproducibility
RANDOM_SEED = 42

# Training configuration
BATCH_SIZE = 1
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# Verify paths exist
def verify_config():
    """Check that configured paths exist"""
    print("=" * 20)
    print("VERIFYING CONFIGURATION")
    print("=" * 20)
    
    print(f"\nProject root: {PROJECT_ROOT}")
    if not PROJECT_ROOT.exists():
        print("Does not exist!")
        return False
    print("Exists")
    
    print(f"\nTraining data directories:")
    total_cases = 0
    for data_dir in TRAINING_DATA_DIRS:
        print(f"  {data_dir.name}:")
        if not data_dir.exists():
            print(f"Does not exist!")
            return False
        
        # Count cases
        cases = [d for d in data_dir.iterdir() if d.is_dir()]
        cases = [d for d in cases if list(d.glob('*-t1c.nii*'))]
        print(f"Found {len(cases)} cases")
        total_cases += len(cases)
    
    print(f"\n  Total cases for train/val/test: {total_cases}")
    
    print(f"\nSplits directory: {SPLITS_DIR}")
    if not SPLITS_DIR.exists():
        print("Does not exist yet (will be created)")
    else:
        print("Exists")
    
    print(f"\nPreprocessed output: {PREPROCESSED_DIR}")
    if not PREPROCESSED_DIR.exists():
        print("Does not exist yet (will be created)")
    else:
        print(" Exists")
    
    print("\n" + "=" * 20)
    print("CONFIGURATION VALID!")
    print("=" * 20)
    return True


if __name__ == '__main__':
    verify_config()