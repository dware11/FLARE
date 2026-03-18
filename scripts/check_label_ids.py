"""
Run from FLARE repo on Delta. Checks label vs folder ID format and fallback matching.
Usage: python check_label_ids.py [path_to_delta_reads.csv]
"""
import csv
import re
from pathlib import Path

# Paths (adjust if your layout differs)
REPO = Path(__file__).resolve().parent if '__file__' in dir() else Path(".")
LABELS_PATH = REPO / "delta_reads.csv"
RAW_CT = Path("/scratch/bckk/flare/ct_brain/raw")  # or from env: Path(os.environ.get("DATA_ROOT", ".")) / "ct_brain" / "raw"

def load_label_ids(path: Path) -> list:
    """Return list of 'name' column values from CSV."""
    ids = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=","):
            name = (row.get("name", row.get("\ufeffname", "")) or "").strip()
            if name:
                ids.append(name)
    return ids

def get_raw_dir_names(raw_root: Path) -> list:
    """Return sorted list of dir names under raw (CQ500*)."""
    if not raw_root.exists():
        return []
    return sorted(
        p.name for p in raw_root.iterdir()
        if p.is_dir() and (p.name.startswith("CQ500-CT-") or p.name.startswith("CQ500CT"))
    )

# --- Normalizers (folder name <-> label file name) ---
def folder_to_label_id(folder_name: str) -> list:
    """Return possible label-file IDs that might correspond to this folder name."""
    candidates = [folder_name]
    # CQ500CT0 -> CQ500-CT-0, CQ500CT10 -> CQ500-CT-10
    m = re.match(r"CQ500CT(\d+)$", folder_name, re.I)
    if m:
        candidates.append(f"CQ500-CT-{m.group(1)}")
    # CQ500-CT-0 -> CQ500CT0
    m = re.match(r"CQ500-CT-(\d+)$", folder_name, re.I)
    if m:
        candidates.append(f"CQ500CT{m.group(1)}")
    return candidates

def label_to_folder_id(label_name: str) -> list:
    """Return possible folder names that might correspond to this label ID."""
    candidates = [label_name]
    m = re.match(r"CQ500-CT-(\d+)$", label_name, re.I)
    if m:
        candidates.append(f"CQ500CT{m.group(1)}")
    m = re.match(r"CQ500CT(\d+)$", label_name, re.I)
    if m:
        candidates.append(f"CQ500-CT-{m.group(1)}")
    return candidates

def main():
    import sys
    labels_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LABELS_PATH
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return

    label_ids = load_label_ids(labels_path)
    label_set = set(label_ids)
    raw_dirs = get_raw_dir_names(RAW_CT)

    print("=== 1) Format in labels file (delta_reads.csv) ===")
    print(f"  Total rows (name): {len(label_ids)}")
    print(f"  Sample (first 5): {label_ids[:5]}")
    print()

    print("=== 2) Format in raw dirs (ct_brain/raw) ===")
    print(f"  Total dirs (CQ500*): {len(raw_dirs)}")
    print(f"  Sample (first 5): {raw_dirs[:5]}")
    print()

    # Exact match
    exact = [d for d in raw_dirs if d in label_set]
    print("=== 3) Exact match (folder name in labels file) ===")
    print(f"  Matched: {len(exact)} / {len(raw_dirs)}")
    if len(raw_dirs) > 0 and len(exact) < len(raw_dirs):
        missing = set(raw_dirs) - set(exact)
        print(f"  Unmatched sample: {sorted(missing)[:10]}")
    print()

    # Fallback: for each raw dir, try folder_to_label_id and see if any candidate is in label_set
    fallback_matched = []
    fallback_map = {}
    for d in raw_dirs:
        if d in label_set:
            fallback_matched.append(d)
            fallback_map[d] = d
            continue
        for cand in folder_to_label_id(d):
            if cand in label_set:
                fallback_matched.append(d)
                fallback_map[d] = cand
                break

    print("=== 4) With fallback (CQ500CT0 <-> CQ500-CT-0) ===")
    print(f"  Matched: {len(fallback_matched)} / {len(raw_dirs)}")
    if fallback_map and list(fallback_map.items())[0][0] != list(fallback_map.items())[0][1]:
        print(f"  Example mapping (folder -> label key): {list(fallback_map.items())[:5]}")
    still_missing = set(raw_dirs) - set(fallback_matched)
    if still_missing:
        print(f"  Still unmatched sample: {sorted(still_missing)[:10]}")
    print()

    # Labels that don't have a raw dir (exact or fallback)
    folder_set = set(raw_dirs)
    folder_set_with_fallback = set()
    for lab in label_ids:
        for cand in label_to_folder_id(lab):
            if cand in raw_dirs:
                folder_set_with_fallback.add(cand)
                break
    in_labels_not_raw = set(label_ids) - set(fallback_map.get(d, d) for d in fallback_matched)
    print("=== 5) In labels file but no matching raw dir ===")
    print(f"  Count: {len(set(label_ids) - set(fallback_map.values()))} (labels without a folder)")
    print(f"  Sample: {list(set(label_ids) - set(fallback_map.values()))[:5]}")

if __name__ == "__main__":
    main()