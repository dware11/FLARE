#!/usr/bin/env python3
"""
Organize Kaggle multimodal brain dataset into stratified train/val/test splits.
70% / 15% / 15% per class per modality. Seed 42.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(folder: Path) -> List[Path]:
    out: List[Path] = []
    if not folder.is_dir():
        return out
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def _unique_dest_name(src: Path, dest_dir: Path) -> Path:
    h = hashlib.md5(str(src.resolve()).encode("utf-8")).hexdigest()[:8]
    ext = src.suffix.lower() or ".jpg"
    candidate = dest_dir / f"{src.stem}_{h}{ext}"
    n = 0
    while candidate.exists():
        n += 1
        candidate = dest_dir / f"{src.stem}_{h}_{n}{ext}"
    return candidate


def _split_indices(n: int, train_frac: float, val_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    if n_train >= n:
        n_train = max(0, n - 1)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    i1 = n_train
    i2 = n_train + n_val
    return perm[:i1], perm[i1:i2], perm[i2:]


def _organize_class(
    files: List[Path],
    label_binary: int,
    label_name: str,
    modality: str,
    out_root: Path,
    rng: np.random.Generator,
    rows: List[Dict[str, object]],
    train_frac: float,
    val_frac: float,
) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    n = len(files)
    if n == 0:
        return counts
    tr_ix, va_ix, te_ix = _split_indices(n, train_frac, val_frac, rng)
    for split, idxs in (("train", tr_ix), ("val", va_ix), ("test", te_ix)):
        dest_dir = out_root / modality / split / label_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i in idxs:
            src = files[int(i)]
            dest = _unique_dest_name(src, dest_dir)
            shutil.copy2(src, dest)
            counts[split] += 1
            rows.append(
                {
                    "filename": dest.name,
                    "path": str(dest.resolve()),
                    "modality": modality,
                    "label": label_name,
                    "label_binary": label_binary,
                    "split": split,
                }
            )
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Organize multimodal brain dataset into stratified splits.")
    ap.add_argument(
        "--source",
        type=Path,
        default=Path("/scratch/bckk/flare/multimodal_brain/Dataset"),
        help="Root of Kaggle download",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/scratch/bckk/flare/multimodal_brain/organized"),
        help="Output organized root",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    src = args.source
    ct_healthy = src / "Brain Tumor CT scan Images" / "Healthy"
    ct_tumor = src / "Brain Tumor CT scan Images" / "Tumor"
    mri_healthy = src / "Brain Tumor MRI images" / "Healthy"
    mri_tumor = src / "Brain Tumor MRI images" / "Tumor"

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, int]] = {}

    def run_modality(modality: str, h_dir: Path, t_dir: Path) -> None:
        h_files = _collect_images(h_dir)
        t_files = _collect_images(t_dir)
        r_h = np.random.default_rng(int(rng.integers(0, 2**31)))
        r_t = np.random.default_rng(int(rng.integers(0, 2**31)))
        ch = _organize_class(
            h_files,
            0,
            "normal",
            modality,
            out_root,
            r_h,
            rows,
            args.train_frac,
            args.val_frac,
        )
        ct = _organize_class(
            t_files,
            1,
            "abnormal",
            modality,
            out_root,
            r_t,
            rows,
            args.train_frac,
            args.val_frac,
        )
        for k in ch:
            summary.setdefault(modality, {}).setdefault(k, 0)
            summary[modality][k] += ch[k] + ct[k]

    run_modality("CT", ct_healthy, ct_tumor)
    run_modality("MRI", mri_healthy, mri_tumor)

    meta_path = out_root / "metadata.csv"
    fieldnames = ["filename", "path", "modality", "label", "label_binary", "split"]
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {meta_path} ({len(rows)} rows)")
    print("\nSummary (counts per modality / split):")
    modalities: Sequence[str] = ("CT", "MRI")
    splits: Sequence[str] = ("train", "val", "test")
    for mod in modalities:
        print(f"  {mod}:")
        for sp in splits:
            c = summary.get(mod, {}).get(sp, 0)
            print(f"    {sp}: {c}")
    print("\nPer-modality × split × label (counts):")
    for mod in modalities:
        for sp in splits:
            for lab in ("normal", "abnormal"):
                n = sum(
                    1
                    for r in rows
                    if r["modality"] == mod and r["split"] == sp and r["label"] == lab
                )
                print(f"  {mod} / {sp} / {lab}: {n}")


if __name__ == "__main__":
    main()
