"""
Create Train/Val/Test Splits
==============================

then splits the combined dataset into:
- 70% training
- 15% validation  
- 15% testing

Creates 3 text files with case names.
"""

import sys
from pathlib import Path
import random

# Import config
try:
    from config import TRAINING_DATA_DIRS, SPLITS_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
except ImportError:
    print("ERROR: config.py not found!")
    print("Please make sure config.py is in the same directory")
    sys.exit(1)


def create_splits():
    """
    Create train/val/test splits from ALL training data directories
    
    Combines cases from all directories listed in TRAINING_DATA_DIRS,
    then splits them into train/val/test.
    """
    
    print("=" * 70)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    
    # Collect cases from ALL training directories
    all_cases = []
    dir_counts = {}
    
    print(f"\nScanning {len(TRAINING_DATA_DIRS)} training directories:")
    for data_dir in TRAINING_DATA_DIRS:
        data_dir = Path(data_dir)
        print(f"\n  {data_dir.name}:")
        
        if not data_dir.exists():
            print(f"    ❌ Directory not found!")
            continue
        
        # Find valid BraTS cases (have t1c file + segmentation)
        cases = [d for d in data_dir.iterdir() if d.is_dir()]
        cases = [d for d in cases if list(d.glob('*-t1c.nii*'))]
        cases = [d for d in cases if list(d.glob('*-seg.nii*'))]
        
        print(f"    Found {len(cases)} cases with segmentation")
        
        all_cases.extend(cases)
        dir_counts[data_dir.name] = len(cases)
    
    if len(all_cases) == 0:
        print("\n❌ ERROR: No valid cases found!")
        print("   Check that directories contain case folders with:")
        print("   - *-t1c.nii.gz (at least one modality)")
        print("   - *-seg.nii.gz (segmentation)")
        return
    
    print(f"\n{'─' * 70}")
    print(f"COMBINED TOTAL: {len(all_cases)} cases")
    for dir_name, count in dir_counts.items():
        print(f"  {dir_name}: {count}")
    print(f"{'─' * 70}")
    
    # Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(all_cases)
    
    # Calculate split sizes
    n_total = len(all_cases)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val  # Remaining
    
    print(f"\nSplit ratios: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")
    print(f"Split sizes:")
    print(f"  Train: {n_train} cases ({100*n_train/n_total:.1f}%)")
    print(f"  Val:   {n_val} cases ({100*n_val/n_total:.1f}%)")
    print(f"  Test:  {n_test} cases ({100*n_test/n_total:.1f}%)")
    print(f"  ───────────────────────")
    print(f"  Total: {n_train + n_val + n_test} cases")
    
    # Split
    train_cases = all_cases[:n_train]
    val_cases = all_cases[n_train:n_train+n_val]
    test_cases = all_cases[n_train+n_val:]
    
    # Create output directory
    output_dir = Path(SPLITS_DIR)
    
    # CLEAN: Remove old splits if they exist
    if output_dir.exists():
        print(f"\nSplits directory already exists: {output_dir}")
        print(f"   Cleaning old split files...")
        
        # Remove old split files
        old_files = list(output_dir.glob('*.txt'))
        for f in old_files:
            f.unlink()
            print(f"     Deleted: {f.name}")
        
        if len(old_files) == 0:
            print(f"     (No old files to clean)")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated splits directory: {output_dir}")
    
    # Save splits (just case names, NOT preprocessed data!)
    print(f"\nSaving splits to: {output_dir}")
    
    with open(output_dir / 'train_cases.txt', 'w') as f:
        f.write('\n'.join([c.name for c in train_cases]))
    print(f"train_cases.txt ({len(train_cases)} cases)")
    
    with open(output_dir / 'val_cases.txt', 'w') as f:
        f.write('\n'.join([c.name for c in val_cases]))
    print(f"=val_cases.txt ({len(val_cases)} cases)")
    
    with open(output_dir / 'test_cases.txt', 'w') as f:
        f.write('\n'.join([c.name for c in test_cases]))
    print(f"test_cases.txt ({len(test_cases)} cases)")
    
    print("\n" + "=" * 70)
    print("✅ SPLITS CREATED!")
    print("=" * 70)
    
    print(f"\nCreated splits from combined dataset:")
    for dir_name, count in dir_counts.items():
        print(f"  {dir_name}: {count} cases")
    print(f"  ───────────────────────")
    print(f"  Total: {len(all_cases)} cases")
    
    print(f"\nSplit into:")
    print(f"  train_cases.txt: {len(train_cases)} cases")
    print(f"  val_cases.txt:   {len(val_cases)} cases")
    print(f"  test_cases.txt:  {len(test_cases)} cases")
    
    print(f"\n💡 Next step:")
    print(f"   python preprocess_all.py")


if __name__ == '__main__':
    create_splits()