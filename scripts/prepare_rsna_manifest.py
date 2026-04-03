"""
Build meta/ct_rsna_manifest.json for RSNA ICH (separate from CQ500).

- Scans RSNA DICOMs under raw_rsna/, groups by StudyInstanceUID.
- Reads competition CSV: per-image labels -> study label = 1 if any slice positive.
- Picks one series folder when multiple exist (choose_best_series + fallback).
- Writes manifest with path/npz_path under cache/rsna/; then run RSNA preprocess
  (python scripts/preprocess_rsna.py ... or scripts/preprocess_ct.py --dataset-type rsna).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CT_RSNA_MANIFEST,
    META,
    RSNA_LABEL_ID_COLUMN,
    RSNA_LABEL_SUBTYPE_COLUMNS,
    RSNA_LABELS_CSV_DEFAULT,
    RSNA_CACHE_DIR,
    RSNA_RAW_ROOT,
)
from scripts.extract_cq500 import choose_best_series  # noqa: E402

try:
    import pydicom
except ImportError:
    pydicom = None

DICOM_EXTS = (".dcm", ".dicom")


def _iter_dicoms(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in DICOM_EXTS:
            yield p


def _study_sop_inst(path: Path) -> tuple[str, str, float]:
    dcm = pydicom.dcmread(str(path), stop_before_pixels=True)
    suid = str(getattr(dcm, "StudyInstanceUID", "") or "").strip()
    sop = str(getattr(dcm, "SOPInstanceUID", "") or "").strip()
    try:
        inst = float(getattr(dcm, "InstanceNumber", 0) or 0)
    except (TypeError, ValueError):
        inst = 0.0
    return suid, sop, inst


def _register_image_keys(mapping: dict[str, int], raw_id: str, abnormal: int) -> None:
    s = (raw_id or "").strip()
    if not s:
        return
    mapping[s] = max(mapping.get(s, 0), abnormal)
    if s.startswith("ID_"):
        mapping[s[3:]] = max(mapping.get(s[3:], 0), abnormal)


def _row_abnormal_wide(row: dict, subtype_cols: tuple[str, ...]) -> int:
    for col in subtype_cols:
        if col not in row:
            continue
        v = (row[col] or "").strip()
        try:
            if v in ("1", "1.0") or (v and float(v) >= 0.5):
                return 1
        except (ValueError, TypeError):
            if v.lower() in ("true", "yes", "positive"):
                return 1
    return 0


def load_rsna_labels_wide(
    csv_path: Path,
    delimiter: str,
    id_column: str,
    subtype_cols: tuple[str, ...],
) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames and id_column not in reader.fieldnames:
            raise SystemExit(
                f"RSNA CSV missing id column {id_column!r}. Found: {reader.fieldnames}"
            )
        for row in reader:
            rid = (row.get(id_column) or "").strip()
            if not rid:
                continue
            ab = _row_abnormal_wide(row, subtype_cols)
            _register_image_keys(out, rid, ab)
    return out


def load_rsna_labels_long(
    csv_path: Path,
    delimiter: str,
    id_column: str,
    label_column: str,
) -> dict[str, int]:
    agg: dict[str, int] = defaultdict(int)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rid = (row.get(id_column) or "").strip()
            if not rid:
                continue
            v = (row.get(label_column) or "").strip()
            try:
                lab = int(float(v))
            except (ValueError, TypeError):
                lab = 1 if v.lower() in ("true", "yes", "positive") else 0
            agg[rid] = max(agg[rid], 1 if lab >= 1 else 0)
    out = dict(agg)
    for k, v in list(out.items()):
        _register_image_keys(out, k, v)
    return out


def study_uid_to_patient_id(suid: str) -> str:
    safe = suid.replace(".", "_").replace("/", "_").replace("\\", "_").strip("_")
    return safe or "rsna_unknown"


def _pick_series_folder(paths: list[Path]) -> Optional[Path]:
    parent_dirs = sorted({p.parent for p in paths})
    if not parent_dirs:
        return None
    if len(parent_dirs) == 1:
        return parent_dirs[0]
    best = choose_best_series(parent_dirs)
    if best is not None:
        folder, _, _ = best
        return folder
    best_dir = None
    best_count = -1
    for d in parent_dirs:
        count = sum(1 for p in paths if p.parent == d)
        if count > best_count:
            best_count = count
            best_dir = d
    return best_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare RSNA manifest (separate from CQ500).")
    ap.add_argument("--raw-root", type=Path, default=None, help=f"default: {RSNA_RAW_ROOT}")
    ap.add_argument("--labels", type=Path, default=None, help=f"default: {RSNA_LABELS_CSV_DEFAULT}")
    ap.add_argument("--output", type=Path, default=None, help=f"default: {CT_RSNA_MANIFEST}")
    ap.add_argument("--k-slices", type=int, default=21)
    ap.add_argument("--delimiter", type=str, default=",")
    ap.add_argument("--csv-format", choices=("wide", "long"), default="wide")
    ap.add_argument("--long-label-column", type=str, default="Target")
    ap.add_argument("--max-studies", type=int, default=None)
    ap.add_argument("--study-ids-file", type=Path, default=None)
    args = ap.parse_args()

    if pydicom is None:
        print("Install pydicom: pip install pydicom")
        sys.exit(1)

    raw_root = args.raw_root or RSNA_RAW_ROOT
    labels_path = args.labels or RSNA_LABELS_CSV_DEFAULT
    out_path = args.output or CT_RSNA_MANIFEST
    k_slices = int(args.k_slices)

    if not raw_root.is_dir():
        print(f"RSNA raw root not found: {raw_root}")
        sys.exit(1)
    if not labels_path.is_file():
        print(f"RSNA labels CSV not found: {labels_path}")
        sys.exit(1)

    if args.csv_format == "wide":
        img_labels = load_rsna_labels_wide(
            labels_path, args.delimiter, RSNA_LABEL_ID_COLUMN, RSNA_LABEL_SUBTYPE_COLUMNS
        )
    else:
        img_labels = load_rsna_labels_long(
            labels_path, args.delimiter, RSNA_LABEL_ID_COLUMN, args.long_label_column
        )

    study_to_paths: dict[str, list[Path]] = defaultdict(list)
    n_read_err = 0
    for fp in sorted(_iter_dicoms(raw_root), key=lambda p: str(p)):
        try:
            suid, _, _ = _study_sop_inst(fp)
        except Exception:
            n_read_err += 1
            continue
        if not suid:
            n_read_err += 1
            continue
        study_to_paths[suid].append(fp)

    allow_studies: Optional[set[str]] = None
    if args.study_ids_file and args.study_ids_file.is_file():
        allow_studies = set()
        for line in args.study_ids_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            allow_studies.add(line)
            allow_studies.add(line.replace("_", "."))

    manifest: list[dict] = []
    for suid in sorted(study_to_paths.keys()):
        paths = study_to_paths[suid]
        series_dir = _pick_series_folder(paths)
        if series_dir is None:
            continue
        dicom_paths = [p for p in paths if p.parent == series_dir]
        meta_cache: dict[Path, tuple[str, str, float]] = {}
        for p in dicom_paths:
            try:
                meta_cache[p] = _study_sop_inst(p)
            except Exception:
                meta_cache[p] = ("", "", 0.0)
        dicom_paths.sort(key=lambda p: (meta_cache[p][2], p.name))

        slice_labels: list[int] = []
        for p in dicom_paths:
            stem = p.stem
            _, sop, _ = meta_cache[p]
            lab = -1
            for key in (stem, sop, stem.replace("ID_", "")):
                if key and key in img_labels:
                    lab = img_labels[key]
                    break
            slice_labels.append(lab)

        known = [x for x in slice_labels if x >= 0]
        label = max(known) if known else -1

        patient_id = study_uid_to_patient_id(suid)
        if allow_studies is not None:
            if suid not in allow_studies and patient_id not in allow_studies:
                continue

        RSNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        npz_path = RSNA_CACHE_DIR / patient_id / f"slices_k{k_slices}.npz"
        manifest.append(
            {
                "patient_id": patient_id,
                "study_uid": suid,
                "path": str(npz_path),
                "npz_path": str(npz_path),
                "label": int(label),
                "num_slices": len(dicom_paths),
                "raw_dicom_dir": str(series_dir.resolve()),
                "source": "rsna",
            }
        )

    manifest.sort(key=lambda e: e["patient_id"])
    if args.max_studies is not None:
        manifest = manifest[: args.max_studies]

    META.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    n_labeled = sum(1 for e in manifest if e["label"] >= 0)
    print(
        f"Wrote {out_path} with {len(manifest)} studies "
        f"({n_labeled} with label 0/1, {len(manifest) - n_labeled} label -1)."
    )
    print(f"DICOM header read errors (skipped files): {n_read_err}")


if __name__ == "__main__":
    main()
