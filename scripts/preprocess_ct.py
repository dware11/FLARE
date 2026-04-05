"""
Demo: DICOM -> HU -> windowing (brain/subdural/bone) -> normalize -> NPZ cache.
Design: preprocessing is offline so inference never re-parses DICOMs.
"""
import argparse
import json
import math
import sys
import warnings
import re
from pathlib import Path
from typing import Optional

import numpy as np

# Suppress known JPEG2000 metadata mismatch warnings from CQ500-like DICOMs.
# These are noisy but typically non-fatal during pixel decode.
warnings.filterwarnings(
    "ignore",
    message=r".*Bits Stored value.*doesn't match the JPEG 2000 data.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Bits Stored value '16'.*does not match the component precision value '14'.*",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import RAW_CT, CACHE_CT, META, CT_RSNA_MANIFEST, RSNA_CACHE_DIR
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


def _read_slice_thickness_mm(dcm) -> float:
    """SliceThickness in mm from DICOM header; 0 if missing or invalid."""
    st = getattr(dcm, "SliceThickness", None)
    if st is None:
        return 0.0
    try:
        v = float(st)
        return v if math.isfinite(v) and v > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


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

def get_sorted_dicom_paths(ct_dir: Path) -> list: 
    """ Return list of Dicom paths in ct_dir sorted by InstanceNumber. """
    dicoms = [f for f in ct_dir.iterdir() if f.is_file() and f.suffix.lower() in DICOM_EXTS] 
    dicoms.sort(key=lambda p: (get_instance_number(p), p.name))
    return dicoms 

def center_k_slice_indices(n_slices: int, k: int = 5) -> np.ndarray: 
    """Return k indices centered on the middles slice; clamp to [0, n_slices-1]; pad by repading if n_slices < k """ 
    if n_slices <= 0: 
        return np.array([0] * k, dtype=np.int64)
    mid = n_slices // 2 
    half = k // 2 
    low = max(0, mid - half) 
    high = min(n_slices, low + k)
    low = max(0, high - k)
    indices = np.arange(low, high, dtype=np.int64) 
    if len(indices) < k: 
        pad_left = (k - len(indices)) // 2
        pad_right = k - len(indices) - pad_left
        indices = np.concatenate([
            np.full(pad_left, indices[0] if len(indices) > 0 else 0),
            indices,
            np.full(pad_right, indices[-1] if len(indices) > 0 else 0), 
        ])
    return indices[:k]


def save_ct_volume_npz_from_dicoms(dicom_paths: list, out_path: Path, k_slices: int) -> None:
    """center_k + hu_to_multiwindow; same NPZ keys as CQ500 (arr, slice_indices, slice_thickness_mm)."""
    n_slices = len(dicom_paths)
    indices = center_k_slice_indices(n_slices, k_slices)
    slope = intercept = None
    slices_list = []
    for idx in indices:
        dcm = pydicom.dcmread(str(dicom_paths[idx]))
        if slope is None:
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        hu = dicom_to_hu(dcm.pixel_array, slope, intercept)
        slices_list.append(hu_to_multiwindow(hu))
    stacked = np.stack(slices_list, axis=0).astype(np.float32)
    mid_path = dicom_paths[n_slices // 2]
    mid_header = pydicom.dcmread(str(mid_path), stop_before_pixels=True)
    slice_thickness_mm = np.float32(_read_slice_thickness_mm(mid_header))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        arr=stacked,
        slice_indices=indices,
        slice_thickness_mm=slice_thickness_mm,
    )


def _merge_rsna_manifest(manifest: list) -> list:
    """Dedupe by patient_id. Do not use CQ500 merge (rewrites paths under CACHE_CT)."""
    by_pid: dict[str, dict] = {}
    for m in manifest:
        pid = (m.get("patient_id") or "").strip()
        if not pid:
            continue
        entry = dict(m)
        entry["patient_id"] = pid
        if pid not in by_pid:
            by_pid[pid] = entry
            continue
        cur = by_pid[pid]
        la, lb = int(cur.get("label", -1)), int(entry.get("label", -1))
        if la == -1:
            cur["label"] = lb
        elif lb != -1 and la != lb:
            cur["label"] = max(la, lb)
        pa, pb = Path(cur.get("path", "")), Path(entry.get("path", ""))
        if pb.is_file() and not pa.is_file():
            cur["path"] = entry["path"]
            cur["npz_path"] = entry.get("npz_path", entry["path"])
        if entry.get("raw_dicom_dir") and not cur.get("raw_dicom_dir"):
            cur["raw_dicom_dir"] = entry["raw_dicom_dir"]
    return list(by_pid.values())


"""
Second pass (not implemented): evenly-spaced k slices, custom windows per dataset,
mixed RSNA+CQ500 manifests, 3D models. Keep disabled until explicitly designed.
"""


def _main_rsna(
    k_slices: int,
    force: bool,
    manifest_path: Path,
    max_studies: Optional[int],
    study_ids_file: Optional[Path],
) -> None:
    """Read/write RSNA manifest only; NPZ under RSNA_CACHE_DIR. Never touches ct_processed_manifest.json."""
    if pydicom is None:
        print("Install pydicom: pip install pydicom")
        sys.exit(1)
    if not manifest_path.is_file():
        print(f"RSNA manifest not found: {manifest_path}")
        print("Run: python scripts/prepare_rsna_manifest.py ...")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    merged = _merge_rsna_manifest(manifest)
    by_id: dict[str, dict] = {e["patient_id"]: dict(e) for e in merged}

    allow: Optional[set[str]] = None
    if study_ids_file and study_ids_file.is_file():
        allow = set()
        for line in study_ids_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            allow.add(line)
            allow.add(line.replace("_", "."))

    candidates = [e for e in merged if e.get("raw_dicom_dir")]
    rows: list[dict] = []
    for e in candidates:
        pid = (e.get("patient_id") or "").strip()
        su = (e.get("study_uid") or "").strip()
        if allow is not None and pid not in allow and su not in allow:
            continue
        rows.append(e)

    if max_studies is not None:
        rows = rows[:max_studies]

    expected_name = f"slices_k{k_slices}.npz"
    skipped_raw = 0
    errors = 0

    print(f"RSNA preprocess: {len(rows)} studies (manifest {manifest_path})")
    for i, entry in enumerate(rows):
        pid = entry["patient_id"]
        out_path = RSNA_CACHE_DIR / pid / expected_name
        entry["path"] = str(out_path)
        entry["npz_path"] = str(out_path)
        entry.setdefault("source", "rsna")

        if not force and out_path.is_file():
            print(f"  [{i + 1}/{len(rows)}] SKIP {pid}: {expected_name} exists")
            by_id[pid] = entry
            continue

        raw_single = entry.get("raw_dicom_path", "")
        raw_dir = Path(entry.get("raw_dicom_dir", ""))
        if raw_single and Path(raw_single).is_file():
            dicom_paths = [Path(raw_single)]
        elif raw_dir.is_dir():
            dicom_paths = get_sorted_dicom_paths(raw_dir)
        else:
            print(f"  [{i + 1}/{len(rows)}] SKIP {pid}: no valid dicom source")
            skipped_raw += 1
            continue

        try:
            if not dicom_paths:
                skipped_raw += 1
                continue
            save_ct_volume_npz_from_dicoms(dicom_paths, out_path, k_slices)
            entry["num_slices"] = len(dicom_paths)
            by_id[pid] = entry
            print(f"  [{i + 1}/{len(rows)}] {pid} -> {out_path.name}")
        except Exception as ex:
            errors += 1
            print(f"  [{i + 1}/{len(rows)}] ERROR {pid}: {ex}")

        if (i + 1) % 10 == 0:
            out_sorted = sorted(by_id.values(), key=lambda x: x.get("patient_id", ""))
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(out_sorted, f, indent=2)

    out_sorted = sorted(by_id.values(), key=lambda x: x.get("patient_id", ""))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(out_sorted, f, indent=2)

    print(f"RSNA done. Updated manifest: {manifest_path}  raw_skips={skipped_raw} errors={errors}")


READS_KEY_FINDINGS = [
    "ICH", "IPH", "IVH", "SDH", "EDH", "SAH",
    "MassEffect", "MidlineShift",
    "Fracture", "CalvarialFracture", "OtherFracture",
]
READS_READERS = ("R1", "R2", "R3")

def normalize_cq500_id(pid: str) -> str:
    """
    Canonical CQ500 patient ID: CQ500-CT-###.
    CQ500CT123 -> CQ500-CT-123; CQ500-CT-123 unchanged; other IDs pass through.
    """
    pid = (pid or "").strip()
    m = re.fullmatch(r"CQ500CT(\d+)", pid, flags=re.IGNORECASE)
    if m:
        return f"CQ500-CT-{m.group(1)}"
    m = re.fullmatch(r"CQ500-CT-(\d+)", pid, flags=re.IGNORECASE)
    if m:
        return f"CQ500-CT-{m.group(1)}"
    return pid


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
            key = normalize_cq500_id(name)
            out[key] = max(out.get(key, 0), abnormal)
    return out


def _merge_manifest_by_canonical_id(manifest: list, k_slices: int) -> list:
    """Collapse duplicate manifest rows that differ only by CQ500 ID spelling."""
    expected_name = f"slices_k{k_slices}.npz"
    by_pid: dict[str, dict] = {}
    for m in manifest:
        pid = normalize_cq500_id((m.get("patient_id") or "").strip())
        entry = dict(m)
        entry["patient_id"] = pid
        if pid not in by_pid:
            by_pid[pid] = entry
            continue
        cur = by_pid[pid]
        la, lb = int(cur.get("label", -1)), int(entry.get("label", -1))
        if la == -1:
            cur["label"] = lb
        elif lb != -1 and la != lb:
            cur["label"] = max(la, lb)
        pa, pb = Path(cur.get("path", "")), Path(entry.get("path", ""))
        want = CACHE_CT / pid / expected_name
        if want.exists():
            cur["path"] = str(want)
        elif pb.exists() and not pa.exists():
            cur["path"] = entry["path"]
        elif pb.exists() and pa.name != expected_name and pb.name == expected_name:
            cur["path"] = entry["path"]
    return list(by_pid.values())


def main(
    limit: Optional[int] = None,
    labels_path: Optional[Path] = None,
    delimiter: Optional[str] = None,
    link_selected_series: bool = False,
    k_slices: int = 5,
    slice_strategy: str = "center_k",
    force: bool = False,
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
        print(f"  Loaded existing manifest with {len(manifest)} entries.")
    else:
        manifest = []

    manifest = _merge_manifest_by_canonical_id(manifest, k_slices)
    # Dict for lookup and in-place updates when reprocessing (canonical patient_id keys)
    manifest_by_id = {m["patient_id"]: m for m in manifest}

    skipped_no_ct = 0
    skipped_error = 0
    expected_filename = f"slices_k{k_slices}.npz"

    for i, pdir in enumerate(to_process):
        folder_name = pdir.name.strip()
        patient_id = normalize_cq500_id(folder_name)

        # Skip vs reprocess: --force always reprocess; else skip only if existing cache matches k_slices
        if not force and patient_id in manifest_by_id:
            existing_entry = manifest_by_id[patient_id]
            old_path = Path(existing_entry.get("path", ""))
            if old_path.name == expected_filename and old_path.exists():
                print(f"  [{i+1}/{n_total}] SKIP {folder_name} (id={patient_id}): already has {expected_filename}")
                continue
            # Else: wrong file (e.g. middle.npz) or missing -> reprocess below

        ct_dir = pdir / "CT_SELECTED"
        if not ct_dir.is_dir(): 
            skipped_no_ct += 1 
            continue 
        if link_selected_series: 
            link_best_series_into_ct_selected(ct_dir)
        
        try:
            dicom_paths = get_sorted_dicom_paths(ct_dir)
            if not dicom_paths:
                skipped_error += 1
                print(f"  [{i+1}/{n_total}] SKIP {pdir.name}: no DICOMs")
                continue
            out_dir = CACHE_CT / patient_id
            out_path = out_dir / f"slices_k{k_slices}.npz"
            save_ct_volume_npz_from_dicoms(dicom_paths, out_path, k_slices)
            label = labels.get(patient_id, -1)

            if patient_id in manifest_by_id:
                # Reprocess: update existing manifest entry in-place
                existing_entry = manifest_by_id[patient_id]
                existing_entry["path"] = str(out_path)
                existing_entry["label"] = label
                existing_entry["patient_id"] = patient_id
                tag = f"{folder_name}" if folder_name != patient_id else patient_id
                print(f"  [{i+1}/{n_total}] REPROCESS {tag} -> {patient_id}: upgrading cache to {expected_filename}")
            else:
                new_entry = {
                    "patient_id": patient_id,
                    "path": str(out_path),
                    "label": label,
                }
                manifest.append(new_entry)
                manifest_by_id[patient_id] = new_entry
                tag = f"{folder_name}" if folder_name != patient_id else patient_id
                print(f"  [{i+1}/{n_total}] {tag} -> {patient_id} / {out_path.name}")

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

    manifest = _merge_manifest_by_canonical_id(manifest, k_slices)

    # Label join debug summary (once after loop)
    matched = sum(1 for m in manifest if m["label"] != -1)
    unmatched = [m["patient_id"] for m in manifest if m["label"] == -1]
    pids = [m["patient_id"] for m in manifest]
    n_dup_rows = len(pids) - len(set(pids))
    print(
        "  Summary: total_entries={} unique_patient_id={} duplicate_rows={}".format(
            len(manifest), len(set(pids)), n_dup_rows
        )
    )
    print("  Loaded labels: {} | Matched (label!=-1): {} | Unmatched (label==-1): {}".format(
        len(labels), matched, len(unmatched)))
    print("  Unmatched sample: {}".format(unmatched[:10]))

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
    ap.add_argument(
        "--dataset-type",
        choices=("cq500", "rsna"),
        default="cq500",
        help="cq500: raw/ + ct_processed_manifest.json. rsna: ct_rsna_manifest.json + cache/rsna/ only.",
    )
    ap.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="CQ500: max patients. RSNA: ignored (use --max-studies).",
    )
    ap.add_argument("--labels", type=Path, default=None, help="CQ500 reads.csv / reads.tsv")
    ap.add_argument("--tsv", action="store_true", help="CQ500 labels file is TSV")
    ap.add_argument("--k-slices", type=int, default=5, help="K slices (center_k); RSNA baseline often 21")
    ap.add_argument("--slice-strategy", type=str, default="center_k", choices=["center_k"])
    ap.add_argument(
        "--link-selected-series",
        action="store_true",
        help="CQ500: symlink best series into CT_SELECTED/",
    )
    ap.add_argument("--force", action="store_true", help="Reprocess even if patient is already in manifest")
    ap.add_argument("--rsna-manifest", type=Path, default=None, help=f"default: {CT_RSNA_MANIFEST}")
    ap.add_argument("--max-studies", type=int, default=None, help="RSNA: cap studies from manifest")
    ap.add_argument("--study-ids-file", type=Path, default=None, help="RSNA: one study UID per line")
    args = ap.parse_args()

    if args.dataset_type == "rsna":
        mp = args.rsna_manifest or CT_RSNA_MANIFEST
        _main_rsna(
            k_slices=args.k_slices,
            force=args.force,
            manifest_path=mp,
            max_studies=args.max_studies,
            study_ids_file=args.study_ids_file,
        )
    else:
        labels_path = args.labels or META / "reads.csv"
        if not labels_path.exists() and (ROOT / "src" / "reads.csv").exists():
            labels_path = ROOT / "src" / "reads.csv"
        delimiter = "\t" if args.tsv else ","
        main(
            limit=args.limit,
            labels_path=labels_path,
            delimiter=delimiter,
            link_selected_series=args.link_selected_series,
            k_slices=args.k_slices,
            slice_strategy=args.slice_strategy,
            force=args.force,
        )

