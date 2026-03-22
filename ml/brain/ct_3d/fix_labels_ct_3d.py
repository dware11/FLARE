import csv
import json
from pathlib import Path

META = Path("/scratch/bckk/flare/ct_brain/meta")

reads_path = META / "reads.csv"
manifest_path = META / "ct_3d_processed_manifest.json"
out_path = META / "ct_3d_processed_manifest_labeled.json"


def normalize_id(name: str) -> str:
    return name.replace("-", "").strip()


def compute_label(row: dict) -> int:
    bleed_keys = ["ICH", "IPH", "IVH", "SDH", "EDH", "SAH"]

    for key, val in row.items():
        if any(b in key for b in bleed_keys):
            if str(val).strip() == "1":
                return 1
    return 0


label_map = {}

with open(reads_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = normalize_id(row["name"])
        label_map[pid] = compute_label(row)

print(f"Loaded labels for {len(label_map)} patients from {reads_path}")

with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

fixed = 0
already_labeled = 0
missing = 0

for entry in manifest:
    pid = entry["patient_id"]

    if entry.get("label", -1) in (0, 1):
        already_labeled += 1
        continue

    if pid in label_map:
        entry["label"] = label_map[pid]
        fixed += 1
    else:
        missing += 1

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("----- RESULTS -----")
print(f"Already labeled: {already_labeled}")
print(f"Recovered labels: {fixed}")
print(f"Still missing: {missing}")
print(f"Saved to: {out_path}")
