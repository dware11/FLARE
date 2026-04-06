#!/usr/bin/env python3
"""
RSNA preprocessing with literature-aligned 3-channel HU windows (does not touch cache/rsna/).

Windows: brain 40/80, subdural 75/215, bone 600/2800.
Outputs NPZ under cache/rsna_windowed/ and writes a separate manifest (default: ct_rsna_manifest_windowed.json).
Skips existing outputs unless --force.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.brain.ct.ct_transforms import dicom_to_hu, hu_to_multiwindow  # noqa: E402
from scripts.preprocess_ct import (  # noqa: E402
    DICOM_EXTS,
    _merge_rsna_manifest,
    _read_slice_thickness_mm,
    center_k_slice_indices,
    get_sorted_dicom_paths,
)
from src.config import CT_RSNA_MANIFEST, CT_RSNA_MANIFEST_WINDOWED, RSNA_WINDOWED_CACHE_DIR  # noqa: E402

try:
    import pydicom
except ImportError:
    pydicom = None

# Clinically motivated multi-window stack (Radiographics / RSNA ICH literature).
WINDOWS_RSNA_LIT = ((40.0, 80.0), (75.0, 215.0), (600.0, 2800.0))


def save_ct_volume_npz_windowed(
    dicom_paths: List[Path],
    out_path: Path,
    k_slices: int,
) -> None:
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
        slices_list.append(hu_to_multiwindow(hu, windows=WINDOWS_RSNA_LIT))
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


def main() -> None:
    ap = argparse.ArgumentParser(description="RSNA 3-window NPZ cache (rsna_windowed).")
    ap.add_argument("--rsna-manifest", type=Path, default=None, help=f"default: {CT_RSNA_MANIFEST}")
    ap.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help=f"default: {CT_RSNA_MANIFEST_WINDOWED}",
    )
    ap.add_argument("--k-slices", type=int, default=21)
    ap.add_argument("--max-studies", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing windowed NPZ files.")
    args = ap.parse_args()

    if pydicom is None:
        print("Install pydicom: pip install pydicom", file=sys.stderr)
        sys.exit(1)

    manifest_path = args.rsna_manifest or CT_RSNA_MANIFEST
    out_manifest = args.out_manifest or CT_RSNA_MANIFEST_WINDOWED
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    merged = _merge_rsna_manifest(manifest)
    by_id: dict[str, dict] = {e["patient_id"]: dict(e) for e in merged if e.get("patient_id")}

    candidates = [e for e in merged if e.get("raw_dicom_dir") or e.get("raw_dicom_path")]
    rows = list(candidates)
    if args.max_studies is not None:
        rows = rows[: args.max_studies]

    expected_name = f"slices_k{args.k_slices}.npz"
    RSNA_WINDOWED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    skipped_raw = 0
    errors = 0

    print(f"RSNA windowed preprocess: {len(rows)} studies -> {RSNA_WINDOWED_CACHE_DIR}")
    for i, entry in enumerate(rows):
        pid = (entry.get("patient_id") or "").strip()
        if not pid:
            continue
        out_path = RSNA_WINDOWED_CACHE_DIR / pid / expected_name
        new_entry = dict(entry)
        new_entry["path"] = str(out_path)
        new_entry["npz_path"] = str(out_path)
        new_entry["preprocess"] = "rsna_windowed_lit"
        new_entry.setdefault("source", "rsna")

        if not args.force and out_path.is_file():
            print(f"  [{i + 1}/{len(rows)}] SKIP {pid}: exists")
            by_id[pid] = new_entry
            continue

        raw_single = entry.get("raw_dicom_path", "")
        raw_dir = Path(entry.get("raw_dicom_dir", ""))
        if raw_single and Path(raw_single).is_file():
            dicom_paths = [Path(raw_single)]
        elif raw_dir.is_dir():
            dicom_paths = get_sorted_dicom_paths(raw_dir)
        else:
            print(f"  [{i + 1}/{len(rows)}] SKIP {pid}: no raw_dicom_path / raw_dicom_dir")
            skipped_raw += 1
            continue

        try:
            if not dicom_paths:
                skipped_raw += 1
                continue
            save_ct_volume_npz_windowed(dicom_paths, out_path, int(args.k_slices))
            new_entry["num_slices"] = len(dicom_paths)
            by_id[pid] = new_entry
            print(f"  [{i + 1}/{len(rows)}] {pid} -> {out_path}")
        except Exception as ex:
            errors += 1
            print(f"  [{i + 1}/{len(rows)}] ERROR {pid}: {ex}")

        if (i + 1) % 10 == 0:
            out_sorted = sorted(by_id.values(), key=lambda x: x.get("patient_id", ""))
            out_manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(out_manifest, "w", encoding="utf-8") as f:
                json.dump(out_sorted, f, indent=2)

    out_sorted = sorted(by_id.values(), key=lambda x: x.get("patient_id", ""))
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(out_sorted, f, indent=2)

    print(f"Done. Manifest: {out_manifest}  skips={skipped_raw} errors={errors}")


if __name__ == "__main__":
    main()
