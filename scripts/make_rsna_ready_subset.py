"""
Build train/val/test RSNA manifest subsets from ct_rsna_manifest.json,
keeping only studies with cached slices_k{K}.npz under RSNA cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CT_META, CT_RSNA_MANIFEST, RSNA_CACHE_DIR  # noqa: E402


def _count_classes(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    n0 = n1 = 0
    for e in rows:
        lab = int(e.get("label", -1))
        if lab == 0:
            n0 += 1
        elif lab == 1:
            n1 += 1
    return n0, n1


def _ready_rows(
    entries: List[Dict[str, Any]],
    cache_root: Path,
    k_slices: int,
) -> List[Dict[str, Any]]:
    ready: List[Dict[str, Any]] = []
    fname = f"slices_k{k_slices}.npz"
    for e in entries:
        pid = e.get("patient_id")
        if pid is None or str(pid).strip() == "":
            continue
        lab = e.get("label", -1)
        try:
            li = int(lab)
        except (TypeError, ValueError):
            continue
        if li not in (0, 1):
            continue
        npz_path = (cache_root / str(pid) / fname).resolve()
        if not npz_path.is_file():
            continue
        s = str(npz_path)
        row = dict(e)
        row["path"] = s
        row["npz_path"] = s
        ready.append(row)
    return ready


def _shuffle_take(rows: List[Dict[str, Any]], rng: np.random.Generator, max_studies: Optional[int]):
    if not rows:
        return rows
    idx = rng.permutation(len(rows))
    out = [rows[i] for i in idx]
    if max_studies is not None:
        out = out[: min(max_studies, len(out))]
    return out


def _apply_max_studies_prefer_ab(
    ab: List[Dict[str, Any]],
    n0: List[Dict[str, Any]],
    rng: np.random.Generator,
    max_studies: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep all abnormals if possible; reduce normals first; else subsample abnormals."""
    n_ab, nn = len(ab), len(n0)
    if n_ab + nn <= max_studies:
        return ab, n0
    if n_ab <= max_studies:
        cap_n = max_studies - n_ab
        if cap_n <= 0:
            idx = rng.choice(n_ab, size=max_studies, replace=False)
            sub_ab = [ab[i] for i in sorted(idx)]
            return sub_ab, []
        if nn <= cap_n:
            return ab, n0
        pick = rng.choice(nn, size=cap_n, replace=False)
        sub_n = [n0[i] for i in sorted(pick)]
        return ab, sub_n
    idx = rng.choice(n_ab, size=max_studies, replace=False)
    return [ab[i] for i in sorted(idx)], []


def _balance_full(
    ready: List[Dict[str, Any]],
    rng: np.random.Generator,
    max_studies: Optional[int],
) -> List[Dict[str, Any]]:
    return _shuffle_take(ready, rng, max_studies)


def _balance_balanced(
    ready: List[Dict[str, Any]],
    rng: np.random.Generator,
    max_studies: Optional[int],
) -> List[Dict[str, Any]]:
    n0_rows = [e for e in ready if int(e["label"]) == 0]
    n1_rows = [e for e in ready if int(e["label"]) == 1]
    n0, n1 = len(n0_rows), len(n1_rows)
    if n0 == 0 or n1 == 0:
        return _shuffle_take(ready, rng, max_studies)
    n_per = min(n0, n1)
    if max_studies is not None:
        n_per = min(n_per, max_studies // 2)
    i0 = rng.choice(len(n0_rows), size=n_per, replace=False)
    i1 = rng.choice(len(n1_rows), size=n_per, replace=False)
    sel = [n0_rows[j] for j in sorted(i0)] + [n1_rows[j] for j in sorted(i1)]
    rng.shuffle(sel)
    return sel


def _balance_ratio(
    ready: List[Dict[str, Any]],
    ratio: float,
    rng: np.random.Generator,
    max_studies: Optional[int],
) -> List[Dict[str, Any]]:
    n0_rows = [e for e in ready if int(e["label"]) == 0]
    n1_rows = [e for e in ready if int(e["label"]) == 1]
    n_ab = len(n1_rows)
    if n_ab == 0:
        return _shuffle_take(ready, rng, max_studies)
    want_n = min(len(n0_rows), int(round(float(ratio) * n_ab)))
    if want_n <= 0:
        ab = list(n1_rows)
        n0_pick: List[Dict[str, Any]] = []
    else:
        pick = rng.choice(len(n0_rows), size=want_n, replace=False)
        n0_pick = [n0_rows[i] for i in sorted(pick)]
        ab = list(n1_rows)
    if max_studies is not None and len(ab) + len(n0_pick) > max_studies:
        ab, n0_pick = _apply_max_studies_prefer_ab(ab, n0_pick, rng, max_studies)
    sel = ab + n0_pick
    rng.shuffle(sel)
    return sel


def _stratified_two_step_split(
    rows: List[Dict[str, Any]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    labels = [int(r["label"]) for r in rows]
    indices = list(range(len(rows)))
    tsum = train_frac + val_frac + test_frac
    if abs(tsum - 1.0) > 1e-6:
        warnings.warn(f"train/val/test fracs sum to {tsum:.6f}, not 1.0; proceeding anyway.", UserWarning)
    test_plus_val = val_frac + test_frac
    if test_plus_val <= 0:
        raise ValueError("val_frac + test_frac must be positive")
    rel_test = test_frac / test_plus_val

    strat1 = labels if len(set(labels)) > 1 else None
    if strat1 is None:
        warnings.warn("Stratify skipped (single class): first split is unstratified.", UserWarning)
    try:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=test_plus_val,
            stratify=strat1,
            random_state=seed,
        )
    except ValueError:
        warnings.warn("Stratified first split failed; falling back to unstratified split.", UserWarning)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=test_plus_val,
            stratify=None,
            random_state=seed,
        )

    temp_labels = [labels[i] for i in temp_idx]
    strat2 = temp_labels if len(set(temp_labels)) > 1 else None
    if strat2 is None and len(temp_idx) > 0:
        warnings.warn(
            "Stratify skipped (single class in val+test pool): second split is unstratified.",
            UserWarning,
        )
    try:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            stratify=strat2,
            random_state=seed + 1,
        )
    except ValueError:
        warnings.warn("Stratified second split failed; falling back to unstratified split.", UserWarning)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            stratify=None,
            random_state=seed + 1,
        )

    return list(train_idx), list(val_idx), list(test_idx)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RSNA ready subset manifests with stratified splits.")
    ap.add_argument("--manifest", type=Path, default=CT_RSNA_MANIFEST)
    ap.add_argument("--cache-root", type=Path, default=RSNA_CACHE_DIR)
    ap.add_argument("--k-slices", type=int, default=21)
    ap.add_argument("--max-studies", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--balance-mode",
        type=str,
        default="full",
        choices=["full", "balanced", "ratio"],
    )
    ap.add_argument(
        "--normal-to-abnormal-ratio",
        type=float,
        default=1.0,
        help="In ratio mode: sample min(n0, round(ratio * n_ab)) normals; all abnormals kept (before max-studies cap).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=CT_META / "subsets",
        help="Directory for subset JSON manifests (default: CT_META/subsets).",
    )
    ap.add_argument("--name-prefix", type=str, default="rsna_ready_k21")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    manifest_path = args.manifest.resolve()
    cache_root = args.cache_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    k = int(args.k_slices)
    prefix = args.name_prefix

    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("Manifest JSON must be a list of objects")

    ready = _ready_rows(entries, cache_root, k)
    counts_before = {
        "normal": _count_classes(ready)[0],
        "abnormal": _count_classes(ready)[1],
        "total": len(ready),
    }
    print(
        f"Ready (NPZ exists, label 0/1): normal={counts_before['normal']} "
        f"abnormal={counts_before['abnormal']} total={counts_before['total']}"
    )

    if args.balance_mode == "full":
        selected = _balance_full(ready, rng, args.max_studies)
    elif args.balance_mode == "balanced":
        selected = _balance_balanced(ready, rng, args.max_studies)
    else:
        selected = _balance_ratio(ready, args.normal_to_abnormal_ratio, rng, args.max_studies)

    counts_after = {
        "normal": _count_classes(selected)[0],
        "abnormal": _count_classes(selected)[1],
        "total": len(selected),
    }
    print(
        f"After balancing: normal={counts_after['normal']} abnormal={counts_after['abnormal']} "
        f"total={counts_after['total']}"
    )

    train_idx, val_idx, test_idx = _stratified_two_step_split(
        selected,
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.seed,
    )

    def rows_with_split(idxs: List[int], split_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in idxs:
            r = dict(selected[i])
            r["split"] = split_name
            out.append(r)
        return out

    train_rows = rows_with_split(train_idx, "train")
    val_rows = rows_with_split(val_idx, "val")
    test_rows = rows_with_split(test_idx, "test")
    full_rows = train_rows + val_rows + test_rows

    def dump(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    dump(output_dir / f"{prefix}.json", full_rows)
    dump(output_dir / f"{prefix}_train.json", train_rows)
    dump(output_dir / f"{prefix}_val.json", val_rows)
    dump(output_dir / f"{prefix}_test.json", test_rows)

    split_counts = {
        "train": {
            "normal": _count_classes(train_rows)[0],
            "abnormal": _count_classes(train_rows)[1],
            "n": len(train_rows),
        },
        "val": {
            "normal": _count_classes(val_rows)[0],
            "abnormal": _count_classes(val_rows)[1],
            "n": len(val_rows),
        },
        "test": {
            "normal": _count_classes(test_rows)[0],
            "abnormal": _count_classes(test_rows)[1],
            "n": len(test_rows),
        },
    }

    summary = {
        "manifest": str(manifest_path),
        "cache_root": str(cache_root),
        "k_slices": k,
        "seed": args.seed,
        "max_studies": args.max_studies,
        "balance_mode": args.balance_mode,
        "normal_to_abnormal_ratio": args.normal_to_abnormal_ratio,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "name_prefix": prefix,
        "output_dir": str(output_dir),
        "class_counts_before_balancing": counts_before,
        "class_counts_after_balancing": counts_after,
        "split_counts": split_counts,
    }
    dump(output_dir / f"{prefix}_subset_summary.json", summary)
    print(f"Wrote manifests under {output_dir}")


if __name__ == "__main__":
    main()