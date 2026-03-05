"""
Demo: DICOM -> HU -> windowing (brain/subdural/bone) -> normalize -> NPZ cache.
Design: preprocessing is offline so inference never re-parses DICOMs.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import RAW_CT, CACHE_CT, META
try:
    from src.debug import dbg as _dbg_log
except ImportError:
    def _dbg_log(_msg: str, data=None, location: str = "") -> None:
        pass
from ml.brain.ct.ct_transforms import dicom_to_hu, hu_to_multiwindow
from scripts.extract_cq500 import choose_best_series


def _dbg(payload: dict) -> None:
    """Forward structured payload to src.debug.dbg (no-op when FLARE_DEBUG not set)."""
    _dbg_log(
        payload.get("message", ""),
        data=payload.get("data"),
        location=payload.get("hypothesisId", ""),
    )

try:
    import pydicom
except ImportError: 
    pydicom = None 

DICOM_EXTS = (".dcm", ".dicom")


def _has_top_level_dicoms(ct_selected: Path) -> bool:
    """Return True if CT_SELECTED has any .dcm/.dicom files directly inside it."""
    return any(
        f.is_file() and f.suffix.lower() in DICOM_EXTS
        for f in ct_selected.iterdir()
    )


def _iter_nested_dicoms(ct_selected: Path):
    """Yield nested DICOM files anywhere under CT_SELECTED (excluding top-level)."""
    for f in ct_selected.rglob("*"):
        if f.is_file() and f.suffix.lower() in DICOM_EXTS and f.parent != ct_selected:
            yield f


def _choose_best_series_in_ct_selected(ct_selected: Path) -> Optional[Path]:
    """
    Use existing choose_best_series() to pick ONE series folder under CT_SELECTED.
    Returns the selected series folder or None.
    """
    series_folders = sorted({f.parent for f in _iter_nested_dicoms(ct_selected)})
    if not series_folders:
        return None

    best = choose_best_series(series_folders)
    if best is None:
        return None

    best_folder, _count, _tier = best
    return best_folder


def link_best_series_into_ct_selected(ct_selected: Path) -> int:
    """
    Idempotent prep step:
    - If CT_SELECTED already has top-level DICOMs, do nothing.
    - If it has no top-level DICOMs but has nested DICOMs, choose the best series
      and create/overwrite symlinks for its .dcm files at CT_SELECTED/<basename>.
    Returns number of links created.
    """
    if not ct_selected.is_dir():
        return 0

    # Already has top-level DICOMs → nothing to do.
    if _has_top_level_dicoms(ct_selected):
        return 0

    # No nested DICOMs at all → nothing we can link.
    if not any(_iter_nested_dicoms(ct_selected)):
        return 0

    series_dir = _choose_best_series_in_ct_selected(ct_selected)
    if series_dir is None or not series_dir.is_dir():
        return 0

    count = 0
    for dcm in series_dir.iterdir():
        if not dcm.is_file() or dcm.suffix.lower() not in DICOM_EXTS:
            continue

        target = ct_selected / dcm.name

        # ln -sf semantics: remove existing file/symlink and recreate.
        try:
            if target.exists() or target.is_symlink():
                target.unlink()
        except FileNotFoundError:
            pass

        target.symlink_to(dcm.resolve())
        count += 1

    return count

def get_middle_dicom_path(ct_dir: Path) -> Path:
    """Return path to the middle slice DICOM in ct_dir by instance number."""
    dicoms = [f for f in ct_dir.iterdir() if f.is_file() and f.suffix.lower() in DICOM_EXTS] 
    dicoms.sort(key=lambda p: (get_instance_number(p), p.name))
    mid = len(dicoms) // 2
    return dicoms[mid]

def get_instance_number(path: Path) -> float:
    """Read InstanceNumber from DICOM header without loading pixels."""
    try: 
        dcm = pydicom.dcmread(str(path), stop_before_pixels=True) 
        return float(getattr(dcm, "InstanceNumber", 0))
    except Exception: 
        return 0 

READS_KEY_FINDINGS = [
    "ICH", "IPH", "IVH", "SDH", "EDH", "SAH",
    "MassEffect", "MidlineShift",
    "Fracture", "CalvarialFracture", "OtherFracture",
]
READS_READERS = ("R1", "R2", "R3")


def load_labels_from_reads(read_csv_path: Path, delimiter: str = ",") -> dict[str, int]:
    """
    Load binary abnormal/normal labels from reads.csv.
    patient_id = row["name"].strip(). abnormal=1 if ANY key finding is 1 for ANY reader (R1/R2/R3).
    """
    import csv
    out = {}
    with open(read_csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            name = (row.get("name", row.get("\ufeffname", "")) or "").strip()
            patient_id = name
            abnormal = 0
            for reader in READS_READERS:
                for col in READS_KEY_FINDINGS:
                    key = f"{reader}:{col}"
                    if key not in row:
                        continue
                    try:
                        v = (row[key] or "").strip()
                        if v in ("1", "1.0") or (v and float(v) >= 0.5):
                            abnormal = 1
                            break
                    except (ValueError, TypeError):
                        pass
                if abnormal:
                    break
            out[patient_id] = abnormal
    return out 


def main(
    limit: Optional[int] = None,
    labels_path: Optional[Path] = None,
    delimiter: Optional[str] = None,
    link_selected_series: bool = False,
):
    """Run CT preprocessing: middle slice -> multi-window -> cache/middle.npz and write manifest."""
    # #region agent log
    import time
    _log_path = ROOT / "debug_pipeline.log"
    def _alog(msg, data=None, hyp=""):
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({"message": msg, "data": data or {}, "hypothesisId": hyp, "location": "preprocess_ct.main", "timestamp": int(time.time() * 1000)}) + "\n")
    _alog("main_entered", {"raw_ct": str(RAW_CT), "raw_ct_exists": RAW_CT.exists()}, "H1")
    # #endregion
    if not RAW_CT.exists():
        print(f"Data path not found: {RAW_CT}")
        print("Create that folder and put patient dirs (CQ500-CT-*/CT_SELECTED/) there, or set DATA_ROOT in src/config.py to your data location.")
        sys.exit(1)
    if pydicom is None:
        print("Install pydicom: pip install pydicom")
        sys.exit(1)
    if delimiter is None:
        delimiter = "\t" if (labels_path and str(labels_path).lower().endswith((".tsv", ".txt"))) else ","
    labels = load_labels_from_reads(labels_path, delimiter=delimiter) if labels_path and labels_path.exists() else {}

    all_dirs = (p for p in RAW_CT.iterdir() if p.is_dir())
    # Support both original naming (CQ500-CT-*) and your current folders (CQ500CT*)
    patient_dirs = sorted(
        p
        for p in all_dirs
        if p.name.startswith("CQ500-CT-") or p.name.startswith("CQ500CT")
    )
    # #region agent log
    _alog("preprocess_start", {"raw_ct": str(RAW_CT), "raw_ct_exists": RAW_CT.exists(), "patient_dirs_count": len(patient_dirs), "labels_count": len(labels)}, "H1")
    # #endregion

    to_process = patient_dirs[:limit] if limit else patient_dirs
    n_total = len(to_process)
    # #region agent log
    _alog("preprocess_to_process", {"to_process_count": n_total, "limit": limit}, "H2")
    # #endregion

    print("Preprocessing CT: middle slice -> multi-window -> cache")
    print(f"  Found {len(patient_dirs)} patient dirs (CQ500-CT-*). Processing {n_total}" + (f" (limit={limit})" if limit else "") + ".")
    if labels:
        print(f"  Labels: {len(labels)} from labels file.")
    else:
        print("  Labels: none (all manifest labels will be -1).")

    META.mkdir(parents=True, exist_ok=True) 
    out_manifest = META / "ct_processed_manifest.json" 

    if out_manifest.exists(): 
        with open(out_manifest, "r", encoding="utf-8") as f: 
            manifest = json.load(f)
        print(f" Loaded existing manifest with {len(manifest)} entries.") 
    else: 
        manifest = []

    processed_ids = {m["patient_id"] for m in manifest} 

    skipped_no_ct = 0 
    skipped_error = 0 

    for i, pdir in enumerate(to_process):
        patient_id = pdir.name.strip() 
        
        if patient_id in processed_ids: 
            print(f" [{i+1}/{n_total}] SKIP {patient_id}: already in manifest")
            continue
        ct_dir = pdir / "CT_SELECTED"
        if not ct_dir.is_dir(): 
            skipped_no_ct += 1 
            continue 
        if link_selected_series: 
            link_best_series_into_ct_selected(ct_dir)
        
        try:     
            dicom_path = get_middle_dicom_path(ct_dir)      # Path Debugging 
            print("Reading DICOM:", dicom_path) 
            print("Exists:", dicom_path.exists())
            print("Absolute:", dicom_path.resolve())
            dcm = pydicom.dcmread(str(dicom_path))
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
            hu = dicom_to_hu(dcm.pixel_array, slope, intercept)
            arr = hu_to_multiwindow(hu)
            # Output: arr (3,256,256) float32 in [0,1]; key "arr" in NPZ.
            out_dir = CACHE_CT / pdir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "middle.npz"
            np.savez_compressed(out_path, arr=arr)
            label = labels.get(patient_id, -1)
            manifest.append({
                "patient_id": pdir.name,
                "path": str(out_path),
                "label": label,
            })
            print(f"  [{i+1}/{n_total}] {pdir.name} -> {out_path.name}")

            # Periodically persist manifest so we don't lose progress on interrupts.
            if (i + 1) % 10 == 0:
                with open(out_manifest, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
        except Exception as e:
            skipped_error += 1
            # #region agent log
            _alog("preprocess_skip", {"patient_id": pdir.name, "error": str(e), "error_type": type(e).__name__}, "H3")
            # #endregion
            _dbg({"hypothesisId": "H3", "message": "exception_skip", "data": {"patient_id": pdir.name, "error": str(e), "error_type": type(e).__name__}})
            print(f"  [{i+1}/{n_total}] SKIP {pdir.name}: {e}")

    # Label join debug summary (once after loop)
    matched = sum(1 for m in manifest if m["label"] != -1)
    unmatched = [m["patient_id"] for m in manifest if m["label"] == -1]
    print("  Loaded labels: {} | Found extracted: {} | Matched: {} | Unmatched sample: {}".format(
        len(labels), len(manifest), matched, unmatched[:10]))

    # #region agent log
    _alog("preprocess_done", {"manifest_count": len(manifest), "manifest_path": str(META / "ct_processed_manifest.json")}, "H4")
    # #endregion
    # Design: manifest is contract between preprocess and inference (path, label).
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _dbg({"hypothesisId": "H4", "message": "manifest_final", "data": {"manifest_count": len(manifest), "patient_ids": [m["patient_id"] for m in manifest]}})

    print("Preprocessing done.")
    print(f"  Processed {len(manifest)}, skipped (error) {skipped_error}, skipped (no CT_SELECTED) {skipped_no_ct}.")
    print(f"  Manifest: {out_manifest} ({len(manifest)} entries)") 

if __name__ == "__main__": 
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", "-n", type=int, default=None,
                    help="Process only N patients (e.g. -n 10 or --limit 10)")
    ap.add_argument("--labels", type=Path, default=None, help="reads.csv or reads.tsv (name, R1:ICH, ...)")
    ap.add_argument("--tsv", action="store_true", help="Labels file is TSV (tab-separated)") 
    ap.add_argument(
        "--link-selected-series",
        action="store_true",
        help="Symlink best CT series DICOMs into CT_SELECTED/ before preprocessing",
    )
    args = ap.parse_args()
    
    labels_path = args.labels or META / "reads.csv" 
    if not labels_path.exists() and (ROOT / "src" / "reads.csv").exists(): 
        labels_path = ROOT / "src" / "reads.csv" 
    delimiter = "\t" if args.tsv else "," 
    main(
        limit=args.limit,
        labels_path=labels_path,
        delimiter=delimiter,
        link_selected_series=args.link_selected_series,
    ) 
    
