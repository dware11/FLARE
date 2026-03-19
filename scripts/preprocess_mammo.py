"""
Preprocess BCS-DBT DICOM volumes into cached NPZ files for training.

For each patient view (LCC/RCC/LMLO/RMLO) in the BCS-DBT dataset:
  1. Load DICOM series → sort by InstanceNumber → (D, H, W) array
  2. Normalize to [0, 1]
  3. Breast ROI crop (Otsu)
  4. CLAHE per slice (clip_limit=2.0, tile_grid=8×8)
  5. Trilinear resample → (32, 256, 256)
  6. Save as NPZ with key "arr"
  7. Append entry to manifest JSON

Usage:
    python scripts/preprocess_mammo.py --data-root /scratch/bckk/flare/breast_mammo/raw
    python scripts/preprocess_mammo.py --data-root /scratch/bckk/flare/breast_mammo/raw --force

BCS-DBT raw layout expected:
    raw/
      <patient_id>/
        <view>/               # LCC, RCC, LMLO, RMLO
          *.dcm               # DICOM slice files

Output:
    cache/<patient_id>/<view>.npz     — preprocessed volume
    meta/mammo_manifest.json          — manifest with path, label, patient_id, view

Label mapping (from BCS-DBT):
    Normal     → 0
    Actionable → 0  (not biopsy-proven cancer)
    Benign     → 0
    Cancer     → 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import BREAST_MAMMO_CACHE, BREAST_MAMMO_META, BREAST_MAMMO_MANIFEST
from ml.breast.mammo.mammo_transforms import preprocess_dbt_volume

# BCS-DBT label string → binary int
_LABEL_MAP = {
    "normal":     0,
    "actionable": 0,
    "benign":     0,
    "cancer":     1,
}

VIEWS = {"LCC", "RCC", "LMLO", "RMLO"}
TARGET_SHAPE = (32, 256, 256)


def load_dicom_series(dicom_dir: Path) -> np.ndarray:
    """Load a DICOM series from a directory, sorted by InstanceNumber."""
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom is required: pip install pydicom")

    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        raise ValueError(f"No DICOM files in {dicom_dir}")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        slices.append((getattr(ds, "InstanceNumber", 0), ds.pixel_array))

    slices.sort(key=lambda t: t[0])
    return np.stack([s[1] for s in slices], axis=0)   # (D, H, W)


def get_label_from_patient_dir(patient_dir: Path) -> int:
    """Try to infer label from a BCS-DBT label file or directory name convention."""
    label_file = patient_dir / "label.txt"
    if label_file.exists():
        text = label_file.read_text().strip().lower()
        return _LABEL_MAP.get(text, -1)
    return -1   # unknown; will need manual annotation


def main(data_root: Path, force: bool = False) -> None:
    BREAST_MAMMO_CACHE.mkdir(parents=True, exist_ok=True)
    BREAST_MAMMO_META.mkdir(parents=True, exist_ok=True)

    # Load existing manifest to resume
    if BREAST_MAMMO_MANIFEST.exists() and not force:
        with open(BREAST_MAMMO_MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)
        done = {(e["patient_id"], e["view"]) for e in manifest}
    else:
        manifest = []
        done = set()

    patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories in {data_root}")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        label = get_label_from_patient_dir(patient_dir)

        for view_dir in patient_dir.iterdir():
            view = view_dir.name.upper()
            if view not in VIEWS or not view_dir.is_dir():
                continue
            if (patient_id, view) in done:
                print(f"  skip {patient_id}/{view} (cached)")
                continue

            out_path = BREAST_MAMMO_CACHE / f"{patient_id}_{view}.npz"
            if out_path.exists() and not force:
                print(f"  skip {patient_id}/{view} (npz exists)")
                done.add((patient_id, view))
                manifest.append({"patient_id": patient_id, "view": view,
                                  "path": str(out_path), "label": label})
                continue

            try:
                raw = load_dicom_series(view_dir)
                arr = preprocess_dbt_volume(raw, target_shape=TARGET_SHAPE)
                np.savez_compressed(out_path, arr=arr)
                manifest.append({"patient_id": patient_id, "view": view,
                                  "path": str(out_path), "label": label})
                done.add((patient_id, view))
                print(f"  processed {patient_id}/{view} → {out_path.name}")
            except Exception as exc:
                print(f"  ERROR {patient_id}/{view}: {exc}")

        # Save manifest after each patient (safe resume)
        with open(BREAST_MAMMO_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nDone. Manifest: {BREAST_MAMMO_MANIFEST} ({len(manifest)} entries)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Path to BCS-DBT raw directory (default: BREAST_MAMMO_RAW from config)")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess all patients even if NPZ already exists")
    args = ap.parse_args()

    from src.config import BREAST_MAMMO_RAW
    data_root = args.data_root or BREAST_MAMMO_RAW
    main(data_root=data_root, force=args.force)
