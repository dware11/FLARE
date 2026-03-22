"""
Build a separate 3D CT manifest without touching the 2D pipeline.

What this does:
- reads the existing CT manifest for labels
- scans the CT cache folders for available NPZ files
- preserves original depth (5, 7, etc.)
- writes a new manifest: ct_3d_processed_manifest.json

Default behavior:
- if both slices_k7 and slices_k5 exist, choose according to --prefer
- otherwise fall back to whatever exists
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.config import META


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_depth_from_npz(npz_path: Path) -> int:
    arr = np.load(npz_path)["arr"]

    # arr could be:
    # (D, H, W)
    # (C, D, H, W)
    # (D, C, H, W)
    if arr.ndim == 3:
        return int(arr.shape[0])

    if arr.ndim == 4:
        if arr.shape[0] in (1, 3):
            return int(arr.shape[1])   # (C, D, H, W)
        if arr.shape[1] in (1, 3):
            return int(arr.shape[0])   # (D, C, H, W)

    raise ValueError(f"Unexpected NPZ shape {arr.shape} in {npz_path}")


def choose_best_npz(
    patient_dir: Path,
    prefer: str = "k5",
) -> Optional[Tuple[Path, str]]:
    """
    prefer:
      - k5: prefer slices_k5, then k7, then middle
      - k7: prefer slices_k7, then k5, then middle
      - middle: prefer middle, then k5, then k7
      - auto: prefer k7, then k5, then middle
    """
    k5 = patient_dir / "slices_k5.npz"
    k7 = patient_dir / "slices_k7.npz"
    middle = patient_dir / "middle.npz"

    if prefer == "k5":
        candidates = [(k5, "slices_k5"), (k7, "slices_k7"), (middle, "middle")]
    elif prefer == "k7":
        candidates = [(k7, "slices_k7"), (k5, "slices_k5"), (middle, "middle")]
    elif prefer == "middle":
        candidates = [(middle, "middle"), (k5, "slices_k5"), (k7, "slices_k7")]
    else:
        candidates = [(k7, "slices_k7"), (k5, "slices_k5"), (middle, "middle")]

    for path, source in candidates:
        if path.exists():
            return path, source

    return None


def build_label_lookup(old_manifest: List[dict]) -> Dict[str, int]:
    """
    Keep the first real 0/1 label we see for a patient.
    If only -1 exists, store -1.
    """
    lookup: Dict[str, int] = {}

    for e in old_manifest:
        pid = e.get("patient_id")
        if not pid:
            continue

        label = int(e.get("label", -1))

        if pid not in lookup:
            lookup[pid] = label
        else:
            # Prefer a real label over -1
            if lookup[pid] == -1 and label in (0, 1):
                lookup[pid] = label

    return lookup


def main(
    old_manifest_path: Optional[Path] = None,
    out_manifest_path: Optional[Path] = None,
    prefer: str = "k5",
) -> None:
    meta_dir = Path(META)
    ct_root = meta_dir.parent
    cache_ct_dir = ct_root / "cache" / "ct"

    old_manifest_path = old_manifest_path or (meta_dir / "ct_processed_manifest.json")
    out_manifest_path = out_manifest_path or (meta_dir / "ct_3d_processed_manifest.json")

    if not old_manifest_path.exists():
        raise FileNotFoundError(f"Old manifest not found: {old_manifest_path}")
    if not cache_ct_dir.exists():
        raise FileNotFoundError(f"CT cache folder not found: {cache_ct_dir}")

    old_manifest = load_json(old_manifest_path)
    label_lookup = build_label_lookup(old_manifest)

    entries = []
    skipped = []

    patient_dirs = sorted([p for p in cache_ct_dir.iterdir() if p.is_dir()])

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        choice = choose_best_npz(patient_dir, prefer=prefer)

        if choice is None:
            skipped.append({"patient_id": patient_id, "reason": "no_npz_found"})
            continue

        npz_path, source = choice
        depth = get_depth_from_npz(npz_path)
        label = int(label_lookup.get(patient_id, -1))

        entry = {
            "patient_id": patient_id,
            "path": str(npz_path),
            "label": label,
            "depth": depth,
            "source": source,
        }
        entries.append(entry)

    save_json(out_manifest_path, entries)

    counts = Counter([e["label"] for e in entries])
    source_counts = Counter([e["source"] for e in entries])
    depth_counts = Counter([e["depth"] for e in entries])

    print(f"Wrote 3D manifest to: {out_manifest_path}")
    print(f"Total entries: {len(entries)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Label counts: {dict(counts)}")
    print(f"Source counts: {dict(source_counts)}")
    print(f"Depth counts: {dict(depth_counts)}")

    if skipped:
        skipped_path = out_manifest_path.with_name("ct_3d_manifest_skipped.json")
        save_json(skipped_path, skipped)
        print(f"Skipped cases written to: {skipped_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build separate 3D CT manifest")
    ap.add_argument("--old-manifest", type=Path, default=None)
    ap.add_argument("--out-manifest", type=Path, default=None)
    ap.add_argument(
        "--prefer",
        type=str,
        default="k5",
        choices=["k5", "k7", "middle", "auto"],
        help="Which processed NPZ to prefer when multiple exist",
    )
    args = ap.parse_args()

    main(
        old_manifest_path=args.old_manifest,
        out_manifest_path=args.out_manifest,
        prefer=args.prefer,
    )
