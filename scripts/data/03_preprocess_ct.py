#!/usr/bin/env python3
"""
Preprocess organized CT JPEGs into NPZ files (224×224, ImageNet-normalized RGB).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
SIZE = (224, 224)


def _load_rgb(path: Path) -> np.ndarray:
    """Return float32 (H,W,3) in [0,1]."""
    im = Image.open(path)
    im = im.convert("RGB")
    im = im.resize(SIZE, Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def _to_chw(img_hwc: np.ndarray) -> np.ndarray:
    """(H,W,3) -> (3,H,W) ImageNet normalized."""
    x = np.transpose(img_hwc, (2, 0, 1))
    x = (x - MEAN) / STD
    return x.astype(np.float32)


def _label_from_folder(name: str) -> int:
    n = name.lower()
    if n == "normal":
        return 0
    if n == "abnormal":
        return 1
    raise ValueError(f"Unknown label folder: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess CT JPEGs to NPZ for fine-tuning.")
    ap.add_argument(
        "--input-root",
        type=Path,
        default=Path("/scratch/bckk/flare/multimodal_brain/organized/CT"),
        help="Organized CT root (train/val/test × normal/abnormal)",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/bckk/flare/multimodal_brain/preprocessed/CT"),
        help="Output NPZ root",
    )
    args = ap.parse_args()

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    skipped = 0

    for split in ("train", "val", "test"):
        split_dir = args.input_root / split
        if not split_dir.is_dir():
            print(f"[warn] missing split dir: {split_dir}", flush=True)
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label = _label_from_folder(label_dir.name)
            except ValueError:
                continue
            out_split = args.output_root / split
            out_split.mkdir(parents=True, exist_ok=True)
            for p in sorted(label_dir.glob("*")):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue
                stem = p.stem
                out_npz = out_split / f"{stem}.npz"
                if out_npz.exists():
                    h8 = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
                    stem = f"{stem}_{h8}"
                    out_npz = out_split / f"{stem}.npz"
                try:
                    hwc = _load_rgb(p)
                    chw = _to_chw(hwc)
                    np.savez_compressed(
                        out_npz,
                        image=chw,
                        label=np.int32(label),
                        filename=str(p.name),
                    )
                    counts[(split, label_dir.name)] += 1
                except Exception as ex:
                    print(f"[skip] {p}: {ex}", flush=True)
                    skipped += 1

    print("\nProcessed counts (split × label folder):")
    for split in ("train", "val", "test"):
        for lab in ("normal", "abnormal"):
            n = counts.get((split, lab), 0)
            print(f"  {split} / {lab}: {n}")
    print(f"Skipped (corrupt): {skipped}")


if __name__ == "__main__":
    main()
