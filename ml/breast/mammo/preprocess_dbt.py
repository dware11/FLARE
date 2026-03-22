"""Preprocess BCS-DBT DICOM views into NPZ cache + JSON manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pydicom

from .config_dbt import (
    DBT_APPLY_CLAHE,
    DBT_CACHE_ROOT,
    DBT_CLAHE_CLIP_LIMIT,
    DBT_CLAHE_TITLE_GRID_SIZE,
    DBT_ENABLE_ROI_CROP,
    DBT_MANIFEST_PATH,
    DBT_NORMALIZE_TO_UNIT,
    DBT_NPZ_DIR,
    DBT_TARGET_SHAPE,
    RAW_DBT_ROOT,
)
from .labels_dbt import describe_bcs_dbt_schema, map_bcs_dbt_label


def load_dbt_view_dicom_series(dicom_dir: Path) -> np.ndarray:
    files = [p for p in dicom_dir.glob("*.dcm") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        inst_num = getattr(ds, "InstanceNumber", 0)
        img = ds.pixel_array.astype(np.float32)
        slices.append((inst_num, img))

    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=0)


def normalize_to_unit(vol: np.ndarray) -> np.ndarray:
    v_min = float(vol.min())
    v_max = float(vol.max())
    if v_max <= v_min:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - v_min) / (v_max - v_min)


def apply_clahe_per_slice(vol: np.ndarray) -> np.ndarray:
    vol_out = np.empty_like(vol, dtype=np.float32)
    clahe = cv2.createCLAHE(
        clipLimit=DBT_CLAHE_CLIP_LIMIT,
        tileGridSize=DBT_CLAHE_TITLE_GRID_SIZE,
    )
    for i in range(vol.shape[0]):
        sl = vol[i]
        sl_8 = np.clip(sl * 255.0, 0, 255).astype(np.uint8)
        sl_clahe = clahe.apply(sl_8)
        vol_out[i] = sl_clahe.astype(np.float32) / 255.0
    return vol_out


def heuristic_breast_roi_crop(vol: np.ndarray) -> np.ndarray:
    if not DBT_ENABLE_ROI_CROP:
        return vol
    return vol


def resize_volume_to_target(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    target_d, target_h, target_w = target_shape
    d, _, _ = vol.shape

    resized_slices = []
    for i in range(d):
        sl = vol[i]
        sl_resized = cv2.resize(sl, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_slices.append(sl_resized)
    vol_hw = np.stack(resized_slices, axis=0)

    if d == target_d:
        return vol_hw.astype(np.float32)

    z = np.linspace(0.0, 1.0, d)
    z_new = np.linspace(0.0, 1.0, target_d)
    vol_out = np.empty((target_d, target_h, target_w), dtype=np.float32)
    for h_idx in range(target_h):
        for w_idx in range(target_w):
            vol_out[:, h_idx, w_idx] = np.interp(z_new, z, vol_hw[:, h_idx, w_idx])
    return vol_out.astype(np.float32)


def _compute_raw_label_from_one_hot(row: Dict[str, str]) -> str:
    normal = int(row.get("Normal", "0"))
    actionable = int(row.get("Actionable", "0"))
    benign = int(row.get("Benign", "0"))
    cancer = int(row.get("Cancer", "0"))
    if cancer == 1:
        return "cancer"
    if normal == 1:
        return "normal"
    if benign == 1:
        return "benign"
    if actionable == 1:
        return "actionable"
    return "unknown"


def load_exam_manifest(
    paths_csv: Path,
    labels_csv: Path,
    patient_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    import csv

    labels_index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["PatientID"].strip()
            sid = row["StudyUID"].strip()
            view = row["View"].strip().lower()
            labels_index[(pid, sid, view)] = row

    entries: List[Dict] = []
    with paths_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["PatientID"].strip()
            if patient_filter is not None and pid not in patient_filter:
                continue
            sid = row["StudyUID"].strip()
            view = row["View"].strip().lower()
            label_row = labels_index.get((pid, sid, view))
            raw_label = _compute_raw_label_from_one_hot(label_row) if label_row else "unknown"
            dicom_rel = row.get("descriptive_path") or row.get("classic_path") or ""
            entries.append(
                {
                    "patient_id": pid,
                    "exam_id": sid,
                    "view": view,
                    "dicom_rel_path": dicom_rel,
                    "raw_label": raw_label,
                }
            )
    return entries


def ensure_dirs() -> None:
    DBT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    DBT_NPZ_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_bcs_dbt(
    paths_csv: Path,
    labels_csv: Path,
    patient_list: Optional[Path] = None,
    max_patients: Optional[int] = None,
    patient_offset: int = 0,
) -> None:
    ensure_dirs()
    print(describe_bcs_dbt_schema())

    patient_filter: Optional[Set[str]] = None
    if patient_list is not None:
        with patient_list.open(encoding="utf-8") as f:
            patient_filter = {line.strip() for line in f if line.strip()}
        print(f"Restricting to {len(patient_filter)} patients from list {patient_list}")
    elif max_patients is not None and max_patients > 0:
        import csv
        from collections import OrderedDict

        ordered: "OrderedDict[str, None]" = OrderedDict()
        with paths_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["PatientID"].strip()
                if pid and pid not in ordered:
                    ordered[pid] = None

        all_pids = list(ordered.keys())
        start = max(patient_offset, 0)
        end = min(start + max_patients, len(all_pids))
        selected = all_pids[start:end]
        patient_filter = set(selected)
        print(f"Restricting to {len(selected)} patients (indices [{start}, {end}))")

    entries = load_exam_manifest(paths_csv, labels_csv, patient_filter=patient_filter)
    manifest_out: List[Dict] = []
    c, d_t, h_t, w_t = DBT_TARGET_SHAPE

    for row in entries:
        exam_id = row["exam_id"]
        patient_id = row["patient_id"]
        view = row["view"]
        raw_label = row["raw_label"]
        dicom_rel = row["dicom_rel_path"]

        dicom_dir = (RAW_DBT_ROOT / Path(dicom_rel)).parent
        y = map_bcs_dbt_label(raw_label)
        if y not in (0, 1):
            continue

        try:
            vol = load_dbt_view_dicom_series(dicom_dir).astype(np.float32)
        except Exception as e:
            print(f"Failed to load {dicom_dir} ({exam_id}, {view}): {e}")
            continue

        if DBT_NORMALIZE_TO_UNIT:
            vol = normalize_to_unit(vol)
        if DBT_APPLY_CLAHE:
            vol = apply_clahe_per_slice(vol)
        vol = heuristic_breast_roi_crop(vol)
        vol = resize_volume_to_target(vol, (d_t, h_t, w_t))
        vol = vol[np.newaxis, :, :, :]

        out_path = DBT_NPZ_DIR / f"{exam_id}_{view}.npz"
        np.savez_compressed(
            out_path,
            image=vol.astype(np.float32),
            label=np.int64(y),
            exam_id=str(exam_id),
            patient_id=str(patient_id),
            view=str(view),
            raw_label=str(raw_label),
        )
        manifest_out.append(
            {
                "path": str(out_path),
                "label": int(y),
                "exam_id": str(exam_id),
                "patient_id": str(patient_id),
                "view": str(view),
                "raw_label": str(raw_label),
            }
        )

    with DBT_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"Saved {len(manifest_out)} entries to {DBT_MANIFEST_PATH}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Preprocess BCS-DBT into NPZ cache.")
    ap.add_argument("--paths-csv", type=str, required=True)
    ap.add_argument("--labels-csv", type=str, required=True)
    ap.add_argument("--patient-list", type=str, default=None)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--patient-offset", type=int, default=0)
    args = ap.parse_args()

    preprocess_bcs_dbt(
        paths_csv=Path(args.paths_csv),
        labels_csv=Path(args.labels_csv),
        patient_list=Path(args.patient_list) if args.patient_list else None,
        max_patients=args.max_patients,
        patient_offset=args.patient_offset,
    )
