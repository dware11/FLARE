"""CSV manifest: init, load, save, dedupe by exam_id, statuses."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_COLUMNS = [
    "exam_id",
    "patient_id",
    "npz_path",
    "label",
    "raw_label",
    "phase_names",
    "phase_pre_series_uid",
    "phase_post1_series_uid",
    "phase_post2_series_uid",
    "selected_by",
    "orig_shape_pre",
    "orig_shape_post1",
    "orig_shape_post2",
    "final_shape",
    "crop_bbox",
    "normalization",
    "roi_method",
    "status",
    "warning",
]


def init_manifest_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()


def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: str(r.get(k, "")) for k in MANIFEST_COLUMNS})
    tmp.replace(path)


def upsert_row(path: Path, row: Dict[str, Any]) -> None:
    init_manifest_csv(path)
    rows = load_manifest(path)
    eid = str(row["exam_id"])
    rows = [r for r in rows if r.get("exam_id") != eid]
    rows.append({k: row.get(k, "") for k in MANIFEST_COLUMNS})
    save_manifest(path, rows)


def find_row(path: Path, exam_id: str) -> Optional[Dict[str, str]]:
    for r in load_manifest(path):
        if r.get("exam_id") == exam_id:
            return r
    return None


def should_skip_exam(path: Path, exam_id: str, force: bool) -> bool:
    if force:
        return False
    r = find_row(path, exam_id)
    if not r:
        return False
    npz = r.get("npz_path", "")
    return r.get("status") == "success" and bool(npz) and Path(npz).is_file()
