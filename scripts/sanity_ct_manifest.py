"""Sanity-check a CT manifest: counts, class balance, examples, missing NPZ, num_slices summary."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity-check a CT manifest JSON.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--examples", type=int, default=5, help="Sample rows to print")
    args = ap.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    n = len(data)
    print(f"studies (rows): {n}")

    labels = [int(e.get("label", -1)) for e in data]
    c = Counter(labels)
    print(f"class balance (label -> count): {dict(sorted(c.items()))}")
    n0, n1 = c.get(0, 0), c.get(1, 0)
    sup = n0 + n1
    if sup:
        print(f"  supervised: {sup}/{n} (normal={n0}, abnormal={n1})")

    paths_ok = [e for e in data if Path(e.get("path", "")).is_file()]
    missing = [e for e in data if not Path(e.get("path", "")).is_file()]
    print(f"npz on disk: {len(paths_ok)} / {n}")
    print(f"missing npz paths: {len(missing)}")
    if missing[:10]:
        print("  first missing (patient_id, path):")
        for e in missing[:10]:
            print(f"    {e.get('patient_id', '?')!r} -> {e.get('path', '')!r}")

    nums = [int(e.get("num_slices", 0) or 0) for e in data]
    pos = [x for x in nums if x > 0]
    if pos:
        print(
            f"num_slices: min={min(pos)} max={max(pos)} mean={sum(pos) / len(pos):.1f} "
            f"(0/missing: {sum(1 for x in nums if x <= 0)})"
        )
    else:
        print("num_slices: not set or all zero")

    ex = min(args.examples, n)
    print(f"\nexample rows (first {ex}):")
    for i, e in enumerate(data[:ex]):
        p = Path(e.get("path", ""))
        print(f"  [{i}] patient_id={e.get('patient_id')!r} label={e.get('label')} npz_exists={p.is_file()}")
        print(f"      path={e.get('path')}")
        if e.get("study_uid"):
            print(f"      study_uid={e.get('study_uid')!r}")
        if e.get("raw_dicom_dir"):
            rd = Path(e["raw_dicom_dir"])
            print(f"      raw_dicom_dir is_dir={rd.is_dir()}")

    if paths_ok:
        p = Path(paths_ok[0]["path"])
        z = np.load(p)
        arr = z["arr"]
        print(f"\nfirst existing NPZ: {p}")
        print(f"  arr shape: {arr.shape} dtype={arr.dtype}")
        if arr.ndim == 4:
            print(f"  (K,3,256,256) OK: {arr.shape[1:] == (3, 256, 256)}")


if __name__ == "__main__":
    main()
