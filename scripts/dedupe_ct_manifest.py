"""
Deduplicate ct_processed_manifest.json by canonical CQ500 patient_id (optional one-off).

Preprocess already merges at load and again before the final write — see preprocess_ct.py.
Use this script when you want to fix the manifest without re-running DICOM preprocessing
(e.g. interrupted run, hand-edited JSON, or copy from another machine).

Usage:
  python scripts/dedupe_ct_manifest.py --dry-run --k-slices 21
  python scripts/dedupe_ct_manifest.py --k-slices 21 --yes
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import META


def _load_preprocess_helpers():
    path = ROOT / "scripts" / "preprocess_ct.py"
    spec = importlib.util.spec_from_file_location("preprocess_ct_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.normalize_cq500_id, mod._merge_manifest_by_canonical_id


def _find_duplicate_groups(manifest: List[dict], normalize) -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for m in manifest:
        pid = normalize((m.get("patient_id") or "").strip())
        by[pid].append(m)
    return {k: v for k, v in by.items() if len(v) > 1}


def main() -> None:
    normalize_cq500_id, merge_fn = _load_preprocess_helpers()

    ap = argparse.ArgumentParser(description="Dedupe CT manifest by canonical CQ500 patient_id")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=META / "ct_processed_manifest.json",
        help="Path to ct_processed_manifest.json",
    )
    ap.add_argument(
        "--k-slices",
        type=int,
        required=True,
        help="K used in slices_k{K}.npz (must match preprocessing; e.g. 15 or 21)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print report only; do not write")
    ap.add_argument("--yes", "-y", action="store_true", help="Write without confirmation")
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .backup.json before overwriting",
    )
    args = ap.parse_args()
    manifest_path: Path = args.manifest

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest: List[dict] = json.load(f)

    n_before = len(manifest)
    groups = _find_duplicate_groups(manifest, normalize_cq500_id)
    merged = merge_fn(manifest, args.k_slices)
    n_after = len(merged)

    print(f"Manifest: {manifest_path}")
    print(f"k-slices: {args.k_slices} (expected NPZ: slices_k{args.k_slices}.npz)")
    print(f"Rows before: {n_before}  after merge: {n_after}  removed: {n_before - n_after}")

    if groups:
        print(f"\nDuplicate canonical-id groups (before merge): {len(groups)}")
        for pid, rows in sorted(groups.items(), key=lambda x: x[0])[:50]:
            raw_ids = [repr((r.get("patient_id") or "")) for r in rows]
            print(f"  {pid}: {len(rows)} rows  raw patient_id values: {', '.join(raw_ids)}")
        if len(groups) > 50:
            print(f"  ... and {len(groups) - 50} more groups")
    else:
        print("\nNo duplicate canonical IDs found (already one row per patient after normalization).")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    if n_before == n_after and not groups:
        print("\nNothing to do; manifest already deduplicated.")
        return

    if not args.yes:
        try:
            ans = input(f"\nOverwrite {manifest_path} with {n_after} rows? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    if not args.no_backup:
        backup = manifest_path.with_suffix(manifest_path.suffix + ".backup.json")
        backup.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote backup: {backup}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote deduplicated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
