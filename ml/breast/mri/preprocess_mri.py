"""
EA1141-style breast MRI preprocessing: discover DICOM series, pick pre/post1/post2,
cache NPZ + CSV manifest. Resume-safe unless --force.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom

from .config_mri import (
    BREAST_MRI_MANIFEST_PATH,
    BREAST_MRI_NPZ_DIR,
    MRI_ADD_SUBTRACTION,
    MRI_CENTER_CROP_FRAC,
    MRI_NORM_HI_PCT,
    MRI_NORM_LO_PCT,
    MRI_POST1_KEYWORDS,
    MRI_POST2_KEYWORDS,
    MRI_PRE_KEYWORDS,
    MRI_PREPROC_VERSION,
    MRI_ROI_FOREGROUND_THRESHOLD,
    MRI_ROI_MIN_FRACTION,
    MRI_ROI_PADDING_FRAC,
    MRI_ROI_REFERENCE,
    MRI_TARGET_SHAPE,
    ensure_mri_dirs,
)
from .labels_mri import map_mri_label_from_row
from .manifest_mri import init_manifest_csv, should_skip_exam, upsert_row
from .mri_transforms import load_dce_series, preprocess_dce_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("preprocess_mri")


@dataclass
class SeriesInfo:
    series_uid: str
    series_dir: Path
    series_description: str
    protocol_name: str
    series_number: int
    acquisition_time: str
    num_files: int

    def meta_text(self) -> str:
        return f"{self.series_description} {self.protocol_name}".lower()


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def discover_series_under_exam(exam_dir: Path) -> List[SeriesInfo]:
    """Group all DICOM files under exam_dir by SeriesInstanceUID."""
    files = list(exam_dir.rglob("*.dcm")) + list(exam_dir.rglob("*.DCM"))
    by_uid: Dict[str, List[Path]] = {}
    for p in files:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception as e:
            LOG.warning("Skip unreadable DICOM %s: %s", p, e)
            continue
        uid = _safe_str(getattr(ds, "SeriesInstanceUID", ""))
        if not uid:
            continue
        by_uid.setdefault(uid, []).append(p)

    out: List[SeriesInfo] = []
    for uid, paths in by_uid.items():
        paths = sorted(set(paths))
        series_dir = paths[0].parent
        try:
            ds0 = pydicom.dcmread(paths[0], stop_before_pixels=True)
        except Exception:
            continue
        out.append(
            SeriesInfo(
                series_uid=uid,
                series_dir=series_dir,
                series_description=_safe_str(getattr(ds0, "SeriesDescription", "")),
                protocol_name=_safe_str(getattr(ds0, "ProtocolName", "")),
                series_number=_safe_int(getattr(ds0, "SeriesNumber", 0)),
                acquisition_time=_safe_str(getattr(ds0, "AcquisitionTime", "")),
                num_files=len(paths),
            )
        )
    out.sort(key=lambda s: (s.acquisition_time, s.series_number, s.series_uid))
    return out


def _match_phase(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def select_phases(
    series_list: List[SeriesInfo],
) -> Tuple[Optional[SeriesInfo], Optional[SeriesInfo], Optional[SeriesInfo], str, str]:
    """Return pre, post1, post2, selected_by, warning."""
    warning = ""
    assigned: Dict[str, SeriesInfo] = {}
    used_uids: set[str] = set()

    for phase, kws in (
        ("pre", MRI_PRE_KEYWORDS),
        ("post1", MRI_POST1_KEYWORDS),
        ("post2", MRI_POST2_KEYWORDS),
    ):
        candidates = [
            s
            for s in series_list
            if s.series_uid not in used_uids and _match_phase(s.meta_text(), kws)
        ]
        candidates.sort(key=lambda s: (s.acquisition_time, s.series_number))
        if candidates:
            assigned[phase] = candidates[0]
            used_uids.add(candidates[0].series_uid)

    pre, p1, p2 = assigned.get("pre"), assigned.get("post1"), assigned.get("post2")
    selected_by = "keyword"

    if pre is None or p1 is None or p2 is None:
        selected_by = "keyword+fallback_acq_order"
        warning = (
            "Incomplete keyword match; using first 3 series by AcquisitionTime/SeriesNumber."
        )
        series_sorted = sorted(
            series_list, key=lambda s: (s.acquisition_time, s.series_number, s.series_uid)
        )
        if len(series_sorted) < 3:
            return (
                None,
                None,
                None,
                selected_by,
                f"Need 3 series, found {len(series_sorted)}.",
            )
        pre, p1, p2 = series_sorted[0], series_sorted[1], series_sorted[2]

    return pre, p1, p2, selected_by, warning


def exam_dir_for_row(data_root: Path, row: Dict[str, str]) -> Path:
    for key in ("dicom_exam_dir", "exam_path", "dicom_root", "path"):
        v = row.get(key)
        if v and str(v).strip():
            p = Path(str(v).strip())
            if not p.is_absolute():
                p = data_root / p
            return p
    pid = str(row.get("patient_id", "")).strip()
    eid = str(row.get("exam_id", "")).strip()
    return data_root / pid / eid


def load_label_rows(labels_csv: Path) -> List[Dict[str, str]]:
    with labels_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def npz_path_for_exam(out_dir: Path, exam_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in exam_id)
    return out_dir / f"{safe}.npz"


def process_one_exam(
    exam_dir: Path,
    exam_id: str,
    patient_id: str,
    mapped_label: int,
    raw_label: str,
    out_npz: Path,
    add_subtraction: bool,
) -> Dict[str, Any]:
    series_list = discover_series_under_exam(exam_dir)
    pre_s, p1_s, p2_s, selected_by, warn = select_phases(series_list)
    if pre_s is None or p1_s is None or p2_s is None:
        raise RuntimeError(warn or "Phase selection failed.")

    pre_v = load_dce_series(pre_s.series_dir)
    p1_v = load_dce_series(p1_s.series_dir)
    p2_v = load_dce_series(p2_s.series_dir)

    vol, bbox_str, roi_method = preprocess_dce_volume(
        pre_v,
        p1_v,
        p2_v,
        target_shape=MRI_TARGET_SHAPE,
        add_subtraction=add_subtraction,
        lo_pct=MRI_NORM_LO_PCT,
        hi_pct=MRI_NORM_HI_PCT,
        roi_threshold=MRI_ROI_FOREGROUND_THRESHOLD,
        roi_padding_frac=MRI_ROI_PADDING_FRAC,
        roi_reference=MRI_ROI_REFERENCE,
        center_crop_frac=MRI_CENTER_CROP_FRAC,
        roi_min_frac=MRI_ROI_MIN_FRACTION,
    )

    c, d, h, w = vol.shape
    expect_c = 4 if add_subtraction else 3
    assert c == expect_c, f"Channel mismatch: got {c}, expected {expect_c}"
    assert (d, h, w) == MRI_TARGET_SHAPE, f"Bad spatial shape {vol.shape}"

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_suffix(".npz.tmp")
    phase_names = ["pre", "post1", "post2"] + (["sub"] if add_subtraction else [])
    np.savez_compressed(
        tmp,
        image=vol.astype(np.float32),
        label=np.int64(mapped_label),
        exam_id=np.array(exam_id),
        patient_id=np.array(patient_id),
        raw_label=np.array(raw_label),
        phase_names=np.array(phase_names),
        preproc_version=np.array(MRI_PREPROC_VERSION),
    )
    tmp.replace(out_npz)

    norm_desc = (
        f"percentile_clip_{MRI_NORM_LO_PCT}_{MRI_NORM_HI_PCT}_to_01_independent_per_phase"
    )
    return {
        "exam_id": exam_id,
        "patient_id": patient_id,
        "npz_path": str(out_npz.resolve()),
        "label": mapped_label,
        "raw_label": raw_label,
        "phase_names": "|".join(phase_names),
        "phase_pre_series_uid": pre_s.series_uid,
        "phase_post1_series_uid": p1_s.series_uid,
        "phase_post2_series_uid": p2_s.series_uid,
        "selected_by": selected_by,
        "orig_shape_pre": str(pre_v.shape),
        "orig_shape_post1": str(p1_v.shape),
        "orig_shape_post2": str(p2_v.shape),
        "final_shape": str(vol.shape),
        "crop_bbox": bbox_str,
        "normalization": norm_desc,
        "roi_method": roi_method,
        "status": "success",
        "warning": warn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Breast MRI EA1141 preprocessing")
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing patient/exam DICOM folders",
    )
    ap.add_argument(
        "--label-csv",
        type=Path,
        required=True,
        help="CSV with exam_id, patient_id, label columns",
    )
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-npz-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--add-subtraction", action="store_true")
    args = ap.parse_args()

    manifest_path = args.manifest or BREAST_MRI_MANIFEST_PATH
    out_dir = args.out_npz_dir if args.out_npz_dir is not None else BREAST_MRI_NPZ_DIR
    ensure_mri_dirs()
    init_manifest_csv(manifest_path)

    rows = load_label_rows(args.label_csv)
    n_ok = 0
    for row in rows:
        if args.limit and n_ok >= args.limit:
            break
        exam_id = str(row.get("exam_id", "")).strip()
        patient_id = str(row.get("patient_id", "")).strip()
        if not exam_id or not patient_id:
            LOG.warning("Skip row missing exam_id/patient_id: %s", row)
            continue

        y, raw_lbl = map_mri_label_from_row(row)
        if y < 0:
            upsert_row(
                manifest_path,
                {
                    "exam_id": exam_id,
                    "patient_id": patient_id,
                    "npz_path": "",
                    "label": y,
                    "raw_label": raw_lbl,
                    "phase_names": "",
                    "phase_pre_series_uid": "",
                    "phase_post1_series_uid": "",
                    "phase_post2_series_uid": "",
                    "selected_by": "",
                    "orig_shape_pre": "",
                    "orig_shape_post1": "",
                    "orig_shape_post2": "",
                    "final_shape": "",
                    "crop_bbox": "",
                    "normalization": "",
                    "roi_method": "",
                    "status": "skipped_ignore_label",
                    "warning": "label -1",
                },
            )
            continue

        if should_skip_exam(manifest_path, exam_id, args.force):
            LOG.info("Skip existing success: %s", exam_id)
            continue

        exam_dir = exam_dir_for_row(args.data_root, row)
        out_npz = npz_path_for_exam(out_dir, exam_id)

        try:
            if not exam_dir.is_dir():
                raise FileNotFoundError(f"Exam dir not found: {exam_dir}")
            mrow = process_one_exam(
                exam_dir,
                exam_id,
                patient_id,
                y,
                raw_lbl,
                out_npz,
                add_subtraction=args.add_subtraction or MRI_ADD_SUBTRACTION,
            )
            LOG.info("OK %s -> %s", exam_id, out_npz)
        except Exception as e:
            LOG.exception("FAILED %s: %s", exam_id, e)
            mrow = {
                "exam_id": exam_id,
                "patient_id": patient_id,
                "npz_path": str(out_npz) if out_npz.exists() else "",
                "label": y,
                "raw_label": raw_lbl,
                "phase_names": "",
                "phase_pre_series_uid": "",
                "phase_post1_series_uid": "",
                "phase_post2_series_uid": "",
                "selected_by": "",
                "orig_shape_pre": "",
                "orig_shape_post1": "",
                "orig_shape_post2": "",
                "final_shape": "",
                "crop_bbox": "",
                "normalization": "",
                "roi_method": "",
                "status": "failed",
                "warning": str(e)[:500],
            }
        upsert_row(manifest_path, mrow)
        if mrow.get("status") == "success":
            n_ok += 1

    LOG.info("Done. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
