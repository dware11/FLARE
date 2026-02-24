"""
preprocess_all.py 
  1. Load raw .nii.gz (4 modalities + segmentation)
  2. Z-score normalize each modality (brain voxels only)
  3. Tumor-aware crop  →  128 × 128 × 128
       Path A: tumor + margin fits in 128³ → pure crop + pad (lossless)
       Path B: giant tumor exceeds 128 on an axis → crop ROI then zoom
  4. Save as compressed .npz — split correctly:
       train/→ {image, seg}  model needs seg to learn from
       val/ →  {image} only  model must predict seg itself
       test/→  {image} only  model must predict seg itself
       val_labels.npz  →  ground truth   loaded separately for Dice scoring
       test_labels.npz →  ground truth   loaded separately for Dice scoring

Run:
    python all.py --check
    python all.py --all
"""

import numpy as np
import nibabel as nib # NIfTI medical image format reader - .nii.gz files
from pathlib import Path
from tqdm import tqdm # progress bar for the loop
import argparse
import sys
import time


# PATHS  —  imported from config.py
from config import TRAINING_DATA_DIRS, SPLITS_DIR, PREPROCESSED_DIR

# For compatibility, set these from config
RAW_DIRS = TRAINING_DATA_DIRS
OUTPUT_DIR = PREPROCESSED_DIR

# PREPROCESSING PARAMETERS
TARGET_SIZE  = (128, 128, 128)   # final volume size (must be divisible by 8)

# CORE HELPERS
def find_file(case_dir: Path, patterns: list):
    """Return the first file that matches any of the glob patterns."""
    for p in patterns:
        hits = list(case_dir.glob(p))
        if hits:
            return hits[0]
    return None


def load_case(case_dir: Path):
    """
    Load all 4 modalities + segmentation from one case folder.
    Returns image (4, H, W, D) float32 and seg (H, W, D) int64.
    """
    
    # A lookup table mapping each MRI modality to possible filename patterns
    mod_patterns = {
        't1n': ['*-t1n.nii*', '*t1n*.nii*', 't1n.nii*'],
        't1c': ['*-t1c.nii*', '*t1c*.nii*', 't1c.nii*'],
        't2w': ['*-t2w.nii*', '*t2w*.nii*', 't2w.nii*'],
        't2f': ['*-t2f.nii*', '*t2f*.nii*', 't2f.nii*'],
    }
    seg_patterns = ['*-seg.nii*', '*seg*.nii*', 'seg.nii*']

    # An empty dictionary that will store the loaded 3D arrays
    mods = {}

    # loop through -> find file -> for each mod turn into 32 bit with it's values 
    for mod, pats in mod_patterns.items():
        f = find_file(case_dir, pats)
        if f is None:
            raise FileNotFoundError(f"Missing {mod} in {case_dir}")
        mods[mod] = nib.load(f).get_fdata().astype(np.float32)

    # load segementation 
    seg_f = find_file(case_dir, seg_patterns)
    if seg_f is None:
        raise FileNotFoundError(f"Missing segmentation in {case_dir}")
    seg = nib.load(seg_f).get_fdata().astype(np.int64)

    # stack the 4 modalities and return image shape(4, H, w, D) and segmenation (H,W,D) single volume with integer labels 0-3
    image = np.stack([mods['t1n'], mods['t1c'], mods['t2w'], mods['t2f']], axis=0)
    return image, seg


# Z-SCORE NORMALIZATION
def normalize(image: np.ndarray) -> np.ndarray:
    """
    Takes the raw MRI intensities (which vary wildly between scanners/patients)
    and standardizes them so all cases have similar value ranges.
    
    Per-modality Z-score on brain voxels only.
    Background (== 0) stays zero.  Outliers clipped to [-5, 5].
    """
    #  Creates a new array of zeros with the same shape as input (4, 240, 240, 155)
    out = np.zeros_like(image, dtype=np.float32)

    # Loops through each of the 4 modalities separately, 
    for i in range(image.shape[0]):
        mod   = image[i]
        mask  = mod > 0 # less than 0 -> background , more than 0 -> brain tissue
        if mask.sum() == 0:
            continue
        mean = mod[mask].mean()
        std  = mod[mask].std()
        if std == 0:
            std = 1.0
        normed = (mod - mean) / std
        normed = np.clip(normed, -5, 5)
        normed[~mask] = 0
        out[i] = normed
    
    return out


# TUMOR-AWARE CROP  →  128 × 128 × 128
# Imported from tumor_aware_crop.py

from tumor_aware_crop import tumor_aware_crop


# SPLIT LOADER  —  who train,val,test
def load_splits():
    """
    Read the three split files that create_splits.py already made.
    Returns (train_names, val_names, test_names) as sets of strings.

    Any case that is NOT listed in val or test is treated as train.
    This means cases from BraTS-Africa (which may not have been in the
    original split) still get processed — they just land in train/.
    """
    def _read(path: Path):
        if path.exists():
            with open(path) as f:
                return {line.strip() for line in f if line.strip()}
        return set()

    val_names  = _read(SPLITS_DIR / 'val_cases.txt')
    test_names = _read(SPLITS_DIR / 'test_cases.txt')
  
    print(f"  Splits loaded: val={len(val_names)}  test={len(test_names)}  "
          f"(everything else → train)")
    return val_names, test_names


# FULL PIPELINE 
def preprocess_case(case_dir: Path):
    """
    Run the full pipeline on one case.
    Returns (image, seg, info_dict).
    """
    image_raw, seg_raw = load_case(case_dir)

    # 1. normalize
    image_norm = normalize(image_raw)

    # 2. tumor-aware crop to 128³
    image_final, seg_final, crop_info = tumor_aware_crop(image_norm, seg_raw)

    info = {
        'raw_shape':      image_raw.shape,
        'crop_path':      crop_info['path'],          
        'tumor_extents':  crop_info['tumor_extents'],
        'final_shape':    image_final.shape,
        'labels':         crop_info['labels'],
        'tumor_voxels':   crop_info['tumor_preserved'],
    }

    return image_final.astype(np.float32), seg_final.astype(np.int64), info


# VALIDATION CHECK

def run_validation_check():
    """
    Pick 3 cases (one from each raw directory if possible), preprocess them,
    print a detailed quality report, and return True if everything looks good.
    """
    print("\n" + "=" * 70)
    print("  PREPROCESSING VALIDATION CHECK  (3 cases)")
    print("=" * 70)

    # gather one case per raw dir (or skip dirs that don't exist)
    sample_cases = []
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            print(f"Directory not found, skipping: {raw_dir}")
            continue
        subdirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
        if subdirs:
            sample_cases.append(subdirs[0])

    # if we got fewer than 3, just use what we have
    if len(sample_cases) == 0:
        print("No cases found anywhere. Check RAW_DIRS paths.")
        return False

    all_ok = True

    for idx, case_dir in enumerate(sample_cases, 1):
        print(f"\n  ── Case {idx}/{len(sample_cases)}: {case_dir.name} ──")
        try:
            image, seg, info = preprocess_case(case_dir)

            # ── shape check ──
            expected = (4,) + TARGET_SIZE
            shape_ok = (image.shape == expected)
            print(f"    Shape:          {image.shape}   {'✓' if shape_ok else '✗ expected ' + str(expected)}")
            if not shape_ok:
                all_ok = False

            # ── normalization check (mean ≈ 0, std ≈ 1 on brain voxels) ──
            brain = image[image != 0]
            mean_ok = abs(brain.mean()) < 0.15
            std_ok  = abs(brain.std() - 1.0) < 0.3
            print(f"    Norm mean:      {brain.mean():+.4f}   {'✓' if mean_ok else '✗ should be ≈ 0'}")
            print(f"    Norm std:       {brain.std():.4f}    {'✓' if std_ok  else '✗ should be ≈ 1'}")
            if not (mean_ok and std_ok):
                all_ok = False

            # ── value range ──
            rng_ok = (image.min() >= -5.0) and (image.max() <= 5.0)
            print(f"    Value range:    [{image.min():.2f}, {image.max():.2f}]   {'✓' if rng_ok else '✗ outside [-5,5]'}")
            if not rng_ok:
                all_ok = False

            # ── labels ──
            labels = info['labels']
            labels_ok = set(labels).issubset({0, 1, 2, 3})
            print(f"    Labels:         {labels}   {'✓' if labels_ok else '✗ unexpected labels'}")
            if not labels_ok:
                all_ok = False

            # ── tumor preserved ──
            tumor_ok = info['tumor_voxels'] > 0
            print(f"    Tumor voxels:   {info['tumor_voxels']}   {'✓' if tumor_ok else '✗ TUMOR LOST!'}")
            if not tumor_ok:
                all_ok = False

            # ── crop path ──
            print(f"    Crop path:      {info['crop_path']}"
                  f"   {'(no resize)' if 'A' in info['crop_path'] else '(resized — giant tumour)'}")
            print(f"    Tumor extents:  {info['tumor_extents']}  (per axis, before crop)")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            all_ok = False

    # ── summary ──
    print("\n" + "-" * 70)
    if all_ok:
        print("  ✅  All checks passed — safe to run full preprocessing.")
    else:
        print("  ⚠️  Some checks failed — review warnings above before continuing.")
    print("-" * 70)

    return all_ok


# FULL PREPROCESSING RUN

def discover_all_cases():
    """
    Walk all RAW_DIRS, return list of Path objects for every case directory.
    """
    cases = []
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            print(f"  ⚠️  Skipping (not found): {raw_dir}")
            continue
        for d in sorted(raw_dir.iterdir()):
            if d.is_dir():
                cases.append(d)
    return cases


def run_full_preprocessing():
    """
    Preprocess every case and save .npz files.

    Output layout:
        preprocessed/
          train/   →  each .npz contains {image, seg}
          val/     →  each .npz contains {image} ONLY
          test/    →  each .npz contains {image} ONLY
          val_labels.npz   →  ground truth for every val case
          test_labels.npz  →  ground truth for every test case
    """

    # create output subfolders
    for sub in ('train', 'val', 'test'):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # ── figure out who is who ──
    val_names, test_names = load_splits()

    # discover every case on disk
    cases = discover_all_cases()
    print(f"\n  Found {len(cases)} cases across all directories.")
    print(f"  Output → {OUTPUT_DIR}\n")

    t0      = time.time()
    failed  = []
    sizes   = []

    # collect ground truth for val and test
    val_labels  = {}   # case_name  →  seg array
    test_labels = {}

    for case_dir in tqdm(cases, desc="Preprocessing", unit="case"):
        case_name = case_dir.name

        # decide which split this case belongs to
        if case_name in val_names:
            split   = 'val'
            out_dir = OUTPUT_DIR / 'val'
        elif case_name in test_names:
            split   = 'test'
            out_dir = OUTPUT_DIR / 'test'
        else:
            split   = 'train'
            out_dir = OUTPUT_DIR / 'train'

        out_file = out_dir / f"{case_name}.npz"

        # resume-friendly
        if out_file.exists():
            # still need to load seg into the label dict if val/test
            # and the labels file hasn't been written yet
            if split in ('val', 'test'):
                labels_dict = val_labels if split == 'val' else test_labels
                if case_name not in labels_dict:
                    # re-preprocess just to grab the seg (fast, single case)
                    try:
                        _, seg, _ = preprocess_case(case_dir)
                        labels_dict[case_name] = seg.astype(np.uint8)
                    except Exception:
                        pass   # will be caught in the failed list if it matters
            continue

        try:
            image, seg, _ = preprocess_case(case_dir)

            if split == 'train':
                # training: model needs both image and labels to compute loss
                np.savez_compressed(out_file, image=image, seg=seg.astype(np.uint8))

            else:
                # val / test: image ONLY in the .npz
                np.savez_compressed(out_file, image=image)
                # stash the ground truth for the labels file
                if split == 'val':
                    val_labels[case_name]  = seg.astype(np.uint8)
                else:
                    test_labels[case_name] = seg.astype(np.uint8)

            sizes.append(out_file.stat().st_size)

        except Exception as e:
            failed.append((case_name, str(e)))

    # dump the label files 
    if val_labels:
        np.savez_compressed(OUTPUT_DIR / 'val_labels.npz', **val_labels)
        print(f"\n  Saved val_labels.npz  ({len(val_labels)} cases)")
    if test_labels:
        np.savez_compressed(OUTPUT_DIR / 'test_labels.npz', **test_labels)
        print(f"  Saved test_labels.npz ({len(test_labels)} cases)")

    # summary
    elapsed  = time.time() - t0
    total_gb = sum(sizes) / 1e9

    n_train = len([c for c in cases if c.name not in val_names and c.name not in test_names])
    n_val   = len([c for c in cases if c.name in val_names])
    n_test  = len([c for c in cases if c.name in test_names])

    print("\n" + "=" * 70)
    print("  Preprocessing Complete")
    print("=" * 70)
    
    # Count by source directory
    print(f"\n Cases processed by source:")
    for raw_dir in RAW_DIRS:
        dir_name = raw_dir.name
        dir_cases = [c for c in cases if str(raw_dir) in str(c.parent)]
        if dir_cases:
            print(f"     {dir_name}: {len(dir_cases)} cases")
    
    print(f"\n Output distribution:")
    print(f"     train/  : {n_train} cases  (with image + seg)")
    print(f"     val/    : {n_val} cases  (image only)")
    print(f"     test/   : {n_test} cases  (image only)")
    print(f"     ────────────────────────────")
    print(f"     Total   : {n_train + n_val + n_test} cases")
    
    print(f"\n   Storage: {total_gb:.2f} GB")
    print(f"     Time: {elapsed/60:.1f} min")
    
    if failed:
        print(f"\n   Failed: {len(failed)} cases")
        for name, err in failed[:3]:  # Show first 3
            print(f"     {name}: {err}")
        if len(failed) > 3:
            print(f"     ... and {len(failed)-3} more")
    
    print(f"\n  📁 Output location:")
    print(f"     {OUTPUT_DIR}")
    print(f"\n     train/           →  image + seg (.npz)")
    print(f"     val/             →  image only (.npz)")
    print(f"     test/            →  image only (.npz)")
    print(f"     val_labels.npz   →  ground truth (separate file)")
    print(f"     test_labels.npz  →  ground truth (separate file)")

# CLI

def main():
    parser = argparse.ArgumentParser(
        description="BraTS preprocessing: normalize → slice-select → crop → .npz"
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Run the 3-case validation check only'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Skip the validation check and go straight to full run'
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  BraTS PREPROCESSING PIPELINE")
    print("  normalize  →  tumor-aware crop  →  .npz  (train/val/test split)")
    print("=" * 70)

    if args.check:
        # validation only
        run_validation_check()
        return

    if args.all:
        # full run, no check
        run_full_preprocessing()
        return

    # default: check first, then ask
    passed = run_validation_check()

    if not passed:
        print("\n  Validation had warnings. Fix them or run with --all to force.\n")
        sys.exit(1)

    answer = input("\n  Proceed with full preprocessing? (yes / no): ").strip().lower()
    if answer in ('yes', 'y'):
        run_full_preprocessing()
    else:
        print("  Cancelled.")


if __name__ == '__main__':
    main()
    