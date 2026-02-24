"""
preprocess_brats2020.py — Preprocess BraTS 2020 for Classification

BraTS 2020 Structure:
    BraTS2020/
    ├── Training/
    │   ├── name_mapping.csv      ← Grade labels here!
    │   ├── survival_info.csv     ← We ignore this
    │   ├── BraTS20_Training_001/
    │   └── ...
    └── Validation/
        ├── name_mapping.csv      ← Grade labels here!
        ├── survival_info.csv     ← We ignore this
        └── ...

name_mapping.csv columns:
    - Grade                    ← "LGG" or "HGG" (WE NEED THIS)
    - BraTS_2017_subject_ID    ← ignore
    - BraTS_2018_subject_ID    ← ignore
    - TCGA_TCIA_subject_ID     ← ignore
    - BraTS_2019_subject_ID    ← ignore
    - BraTS_2020_subject_ID    ← folder name (WE NEED THIS)

Output:
    brats2020_processed/
        train/           → {image, seg}
        val/             → {image}
        test/            → {image}
        val_labels.npz   → segmentation ground truth
        test_labels.npz  → segmentation ground truth
        grade_labels.npz → 0=LGG, 1=HGG


Run:
    python preprocess_brats2020.py --check
    python preprocess_brats2020.py --all

"""

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
import argparse
import sys
import time

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Input: BraTS 2020 root directory
BRATS2020_DIR = Path("/Users/huda/Desktop/Flare_brain_mri/classification_data_2020")

# Output: Preprocessed data
OUTPUT_DIR = Path("/Users/huda/Desktop/Flare_brain_mri/classification_data_2020/brats2020_processed")
# Preprocessing parameters (same as your BraTS 2023)
TARGET_SIZE = (128, 128, 128)

# Split ratios (we combine Training + Validation, then re-split)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ══════════════════════════════════════════════════════════════════════════════
#                         LOAD GRADE LABELS FROM CSV
# ══════════════════════════════════════════════════════════════════════════════

def load_grade_labels():
    """
    Read name_mapping.csv from Training/ folder ONLY.
    (Validation/ folder does NOT have grade labels - hidden for competition)
    
    Extract: Grade and BraTS_2020_subject_ID
    
    Returns:
        dict: {case_id: grade_label} where 0=LGG, 1=HGG
    """
    grade_labels = {}
    
    # ONLY use Training folder - Validation has no grade labels!
    csv_path = BRATS2020_DIR / 'Training' / 'name_mapping.csv'
    
    # Also try alternate path
    if not csv_path.exists():
        csv_path = BRATS2020_DIR / 'name_mapping.csv'
    
    if not csv_path.exists():
        print(f"  ❌ name_mapping.csv not found!")
        print(f"     Looked in: {BRATS2020_DIR / 'Training'}")
        return grade_labels
    
    print(f"  Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Show columns for debugging
    print(f"    Columns: {list(df.columns)}")
    
    # Find the right columns (case-insensitive)
    grade_col = None
    id_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'grade':
            grade_col = col
        elif 'brats_2020' in col_lower or 'brats2020' in col_lower:
            id_col = col
    
    if grade_col is None:
        print(f"    ❌ 'Grade' column not found!")
        return grade_labels
    if id_col is None:
        print(f"    ❌ 'BraTS_2020_subject_ID' column not found!")
        return grade_labels
    
    print(f"    Using: '{grade_col}' and '{id_col}'")
    
    # Extract labels
    hgg_count = 0
    lgg_count = 0
    
    for _, row in df.iterrows():
        case_id = str(row[id_col]).strip()
        grade = str(row[grade_col]).strip().upper()
        
        # Skip empty/invalid rows
        if pd.isna(row[grade_col]) or pd.isna(row[id_col]):
            continue
        if grade not in ['LGG', 'HGG']:
            continue
        
        if grade == 'HGG':
            grade_labels[case_id] = 1
            hgg_count += 1
        else:
            grade_labels[case_id] = 0
            lgg_count += 1
    
    print(f"\n  ✓ Loaded {len(grade_labels)} cases from Training/")
    print(f"    HGG: {hgg_count}")
    print(f"    LGG: {lgg_count}")
    print(f"\n  Note: Validation/ folder has NO grade labels (competition holdout)")
    print(f"        We will split Training/ into train/val/test")
    
    return grade_labels


# ══════════════════════════════════════════════════════════════════════════════
#                         DISCOVER CASE FOLDERS
# ══════════════════════════════════════════════════════════════════════════════

def discover_cases(grade_labels):
    """
    Find all case folders in Training/ that have grade labels.
    (Validation/ has no labels, so we skip it)
    
    Returns:
        List of (case_path, case_name, grade_label)
    """
    cases = []
    
    # ONLY look in Training folder
    folder_path = BRATS2020_DIR / 'Training'
    
    if not folder_path.exists():
        # Try without subfolder (maybe data is directly in BRATS2020_DIR)
        folder_path = BRATS2020_DIR
    
    if not folder_path.exists():
        print(f"  ❌ Training folder not found at {BRATS2020_DIR}")
        return cases
    
    print(f"\n  Scanning {folder_path}...")
    
    for case_dir in sorted(folder_path.iterdir()):
        if not case_dir.is_dir():
            continue
        
        case_name = case_dir.name
        
        # Must start with BraTS
        if not case_name.startswith('BraTS'):
            continue
        
        # Must have a grade label
        if case_name in grade_labels:
            cases.append((case_dir, case_name, grade_labels[case_name]))
    
    hgg = sum(1 for _, _, g in cases if g == 1)
    lgg = len(cases) - hgg
    print(f"  Found {len(cases)} cases with labels (HGG: {hgg}, LGG: {lgg})")
    
    return cases


# ══════════════════════════════════════════════════════════════════════════════
#                         LOAD CASE DATA
# ══════════════════════════════════════════════════════════════════════════════

def find_file(case_dir: Path, patterns: list):
    """Find first file matching any pattern."""
    for p in patterns:
        hits = list(case_dir.glob(p))
        if hits:
            return hits[0]
    return None


def load_case(case_dir: Path):
    """
    Load 4 MRI modalities + segmentation from case folder.
    
    BraTS 2020 naming:
        {case}_t1.nii.gz
        {case}_t1ce.nii.gz
        {case}_t2.nii.gz
        {case}_flair.nii.gz
        {case}_seg.nii.gz
    
    Returns:
        image: (4, H, W, D) float32
        seg: (H, W, D) int64
    """
    mod_patterns = {
        't1':    ['*_t1.nii*'],
        't1ce':  ['*_t1ce.nii*', '*_t1Gd.nii*'],
        't2':    ['*_t2.nii*'],
        'flair': ['*_flair.nii*'],
    }
    seg_patterns = ['*_seg.nii*']
    
    mods = {}
    for mod, pats in mod_patterns.items():
        f = find_file(case_dir, pats)
        if f is None:
            raise FileNotFoundError(f"Missing {mod} in {case_dir.name}")
        mods[mod] = nib.load(f).get_fdata().astype(np.float32)
    
    seg_f = find_file(case_dir, seg_patterns)
    if seg_f is None:
        raise FileNotFoundError(f"Missing seg in {case_dir.name}")
    seg = nib.load(seg_f).get_fdata().astype(np.int64)
    
    # Stack: t1, t1ce, t2, flair (same order as BraTS 2023)
    image = np.stack([mods['t1'], mods['t1ce'], mods['t2'], mods['flair']], axis=0)
    
    return image, seg


# ══════════════════════════════════════════════════════════════════════════════
#                         PREPROCESSING FUNCTIONS
#              (Same as your BraTS 2023 segmentation preprocessing)
# ══════════════════════════════════════════════════════════════════════════════

# Import your existing tumor_aware_crop
# Make sure tumor_aware_crop.py is in the same directory or Python path
try:
    from tumor_aware_crop import tumor_aware_crop
    print("  ✓ Using existing tumor_aware_crop.py")
    USE_EXISTING_CROP = True
except ImportError:
    print("  ⚠️ tumor_aware_crop.py not found, using built-in version")
    USE_EXISTING_CROP = False


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalize per modality. Brain voxels only.
    Clip to [-5, 5]. Background stays 0.
    
    IDENTICAL to your preprocess_all.py
    """
    out = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        mod = image[i]
        mask = mod > 0
        if mask.sum() == 0:
            continue
        
        mean = mod[mask].mean()
        std = mod[mask].std()
        if std == 0:
            std = 1.0
        
        normed = (mod - mean) / std
        normed = np.clip(normed, -5, 5)
        normed[~mask] = 0
        out[i] = normed
    
    return out


def fix_labels(seg: np.ndarray) -> np.ndarray:
    """
    BraTS 2020: label 4 = Enhancing Tumor
    BraTS 2023: label 3 = Enhancing Tumor
    
    Convert 4 → 3
    """
    seg_fixed = seg.copy()
    seg_fixed[seg_fixed == 4] = 3
    return seg_fixed


def _builtin_tumor_aware_crop(image: np.ndarray, seg: np.ndarray, 
                               target_size=(128, 128, 128), margin=10):
    """
    Fallback crop function if tumor_aware_crop.py not found.
    """
    from scipy.ndimage import zoom
    
    tumor_mask = seg > 0
    
    if tumor_mask.sum() == 0:
        center = [s // 2 for s in image.shape[1:]]
    else:
        coords = np.where(tumor_mask)
        mins = [c.min() for c in coords]
        maxs = [c.max() for c in coords]
        center = [(mi + ma) // 2 for mi, ma in zip(mins, maxs)]
    
    starts = []
    for c, t, s in zip(center, target_size, image.shape[1:]):
        start = c - t // 2
        start = max(0, min(start, s - t))
        starts.append(start)
    
    ends = [st + t for st, t in zip(starts, target_size)]
    ends = [min(e, s) for e, s in zip(ends, image.shape[1:])]
    
    image_crop = image[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    seg_crop = seg[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    
    current = image_crop.shape[1:]
    if current != target_size:
        zoom_factors = [t / c for t, c in zip(target_size, current)]
        
        image_resized = np.zeros((4,) + target_size, dtype=np.float32)
        for m in range(4):
            image_resized[m] = zoom(image_crop[m], zoom_factors, order=1)
        seg_resized = zoom(seg_crop, zoom_factors, order=0).astype(np.int64)
        
        return image_resized, seg_resized, {}
    
    if any(c < t for c, t in zip(current, target_size)):
        pad_width = [(0, 0)]
        for c, t in zip(current, target_size):
            pad = t - c
            pad_width.append((pad // 2, pad - pad // 2))
        
        image_crop = np.pad(image_crop, pad_width, mode='constant')
        seg_crop = np.pad(seg_crop, pad_width[1:], mode='constant')
    
    return image_crop, seg_crop, {}


def preprocess_case(case_dir: Path):
    """Full preprocessing pipeline for one case."""
    image_raw, seg_raw = load_case(case_dir)
    
    # Fix BraTS 2020 labels (4 → 3)
    seg_fixed = fix_labels(seg_raw)
    
    # Normalize (same as BraTS 2023)
    image_norm = normalize(image_raw)
    
    # Tumor-aware crop (use your existing function if available)
    if USE_EXISTING_CROP:
        image_final, seg_final, _ = tumor_aware_crop(image_norm, seg_fixed)
    else:
        image_final, seg_final, _ = _builtin_tumor_aware_crop(image_norm, seg_fixed)
    
    return image_final.astype(np.float32), seg_final.astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
#                         SPLIT DATA
# ══════════════════════════════════════════════════════════════════════════════

def create_splits(cases):
    """
    Create stratified train/val/test splits.
    Maintains HGG:LGG ratio in each split.
    """
    from sklearn.model_selection import train_test_split
    
    hgg = [c for c in cases if c[2] == 1]
    lgg = [c for c in cases if c[2] == 0]
    
    def split_group(group):
        if len(group) < 3:
            return group, [], []
        train, temp = train_test_split(group, test_size=0.3, random_state=42)
        if len(temp) < 2:
            return train, temp, []
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        return train, val, test
    
    hgg_train, hgg_val, hgg_test = split_group(hgg)
    lgg_train, lgg_val, lgg_test = split_group(lgg)
    
    train = hgg_train + lgg_train
    val = hgg_val + lgg_val
    test = hgg_test + lgg_test
    
    print(f"\n  Split:")
    print(f"    Train: {len(train)} (HGG: {len(hgg_train)}, LGG: {len(lgg_train)})")
    print(f"    Val:   {len(val)} (HGG: {len(hgg_val)}, LGG: {len(lgg_val)})")
    print(f"    Test:  {len(test)} (HGG: {len(hgg_test)}, LGG: {len(lgg_test)})")
    
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
#                         VALIDATION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def run_validation_check():
    """Check preprocessing on 3 sample cases."""
    
    print("\n" + "=" * 70)
    print("  VALIDATION CHECK")
    print("=" * 70)
    
    # Load grades
    print("\n  Loading grade labels from name_mapping.csv...")
    grade_labels = load_grade_labels()
    
    if not grade_labels:
        print("\n  ❌ No labels found!")
        return False
    
    # Find cases
    cases = discover_cases(grade_labels)
    if not cases:
        print("\n  ❌ No cases found!")
        return False
    
    # Sample
    hgg = [c for c in cases if c[2] == 1][:2]
    lgg = [c for c in cases if c[2] == 0][:1]
    samples = hgg + lgg
    
    all_ok = True
    
    for i, (case_dir, name, grade) in enumerate(samples, 1):
        g_str = "HGG" if grade == 1 else "LGG"
        print(f"\n  Case {i}/{len(samples)}: {name} ({g_str})")
        
        try:
            image, seg = preprocess_case(case_dir)
            
            # Checks
            ok = True
            
            # Shape
            shape_ok = image.shape == (4,) + TARGET_SIZE
            print(f"    Shape: {image.shape} {'✓' if shape_ok else '✗'}")
            ok &= shape_ok
            
            # Normalization
            brain = image[image != 0]
            mean_ok = abs(brain.mean()) < 0.15
            std_ok = abs(brain.std() - 1) < 0.3
            print(f"    Mean:  {brain.mean():+.3f} {'✓' if mean_ok else '✗'}")
            print(f"    Std:   {brain.std():.3f} {'✓' if std_ok else '✗'}")
            ok &= mean_ok and std_ok
            
            # Range
            rng_ok = image.min() >= -5 and image.max() <= 5
            print(f"    Range: [{image.min():.1f}, {image.max():.1f}] {'✓' if rng_ok else '✗'}")
            ok &= rng_ok
            
            # Labels
            labels = np.unique(seg).tolist()
            labels_ok = set(labels).issubset({0, 1, 2, 3})
            print(f"    Labels: {labels} {'✓' if labels_ok else '✗'}")
            ok &= labels_ok
            
            # Tumor
            tumor = (seg > 0).sum()
            tumor_ok = tumor > 0
            print(f"    Tumor: {tumor} voxels {'✓' if tumor_ok else '✗'}")
            ok &= tumor_ok
            
            if not ok:
                all_ok = False
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            all_ok = False
    
    print("\n" + "-" * 70)
    print(f"  {'✅ All checks passed!' if all_ok else '⚠️ Some checks failed'}")
    print("-" * 70)
    
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
#                         FULL PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def run_full_preprocessing():
    """Preprocess all BraTS 2020 cases."""
    
    print("\n" + "=" * 70)
    print("  FULL PREPROCESSING")
    print("=" * 70)
    
    # Setup
    for sub in ('train', 'val', 'test'):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print("\n  Loading grade labels...")
    grade_labels = load_grade_labels()
    if not grade_labels:
        return
    
    # Find cases
    cases = discover_cases(grade_labels)
    if not cases:
        return
    
    # Split
    train, val, test = create_splits(cases)
    
    print(f"\n  Output: {OUTPUT_DIR}")
    
    t0 = time.time()
    failed = []
    
    val_segs = {}
    test_segs = {}
    all_grades = {}
    
    for case_list, split in [(train, 'train'), (val, 'val'), (test, 'test')]:
        print(f"\n  Processing {split}...")
        
        for case_dir, name, grade in tqdm(case_list, desc=f"  {split}"):
            out_file = OUTPUT_DIR / split / f"{name}.npz"
            all_grades[name] = grade
            
            if out_file.exists():
                # Still need seg for val/test labels
                if split in ('val', 'test'):
                    try:
                        _, seg = preprocess_case(case_dir)
                        if split == 'val':
                            val_segs[name] = seg.astype(np.uint8)
                        else:
                            test_segs[name] = seg.astype(np.uint8)
                    except:
                        pass
                continue
            
            try:
                image, seg = preprocess_case(case_dir)
                
                if split == 'train':
                    np.savez_compressed(out_file, image=image, seg=seg.astype(np.uint8))
                else:
                    np.savez_compressed(out_file, image=image)
                    if split == 'val':
                        val_segs[name] = seg.astype(np.uint8)
                    else:
                        test_segs[name] = seg.astype(np.uint8)
                
            except Exception as e:
                failed.append((name, str(e)))
    
    # Save label files
    if val_segs:
        np.savez_compressed(OUTPUT_DIR / 'val_labels.npz', **val_segs)
        print(f"\n  ✓ val_labels.npz ({len(val_segs)} cases)")
    
    if test_segs:
        np.savez_compressed(OUTPUT_DIR / 'test_labels.npz', **test_segs)
        print(f"  ✓ test_labels.npz ({len(test_segs)} cases)")
    
    np.savez_compressed(OUTPUT_DIR / 'grade_labels.npz', **all_grades)
    print(f"  ✓ grade_labels.npz ({len(all_grades)} cases)")
    
    # Summary
    elapsed = time.time() - t0
    hgg = sum(1 for g in all_grades.values() if g == 1)
    lgg = len(all_grades) - hgg
    
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"""
  Cases: {len(all_grades)} (HGG: {hgg}, LGG: {lgg})
  
  Split:
    train/: {len(train)}
    val/:   {len(val)}
    test/:  {len(test)}
  
  Files:
    grade_labels.npz  → classification (0=LGG, 1=HGG)
    val_labels.npz    → segmentation ground truth
    test_labels.npz   → segmentation ground truth
  
  Time: {elapsed/60:.1f} min
    """)
    
    if failed:
        print(f"  Failed: {len(failed)}")
        for n, e in failed[:5]:
            print(f"    {n}: {e}")
    
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#                         MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  BraTS 2020 PREPROCESSING")
    print("  Reads Grade from name_mapping.csv")
    print("=" * 70)
    
    if args.check:
        run_validation_check()
    elif args.all:
        run_full_preprocessing()
    else:
        if run_validation_check():
            ans = input("\n  Proceed? (y/n): ").strip().lower()
            if ans in ('y', 'yes'):
                run_full_preprocessing()


if __name__ == '__main__':
    main()