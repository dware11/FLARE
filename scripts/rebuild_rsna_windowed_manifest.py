import csv
import json
from pathlib import Path

CACHE_ROOT = Path("/scratch/bckk/flare/ct_brain/cache/rsna_windowed")
OUT_PATH = Path("/scratch/bckk/flare/ct_brain/meta/ct_rsna_manifest_windowed_rebuilt.json")

LABEL_CANDIDATES = [
    Path("/scratch/bckk/flare/ct_brain/raw_rsna/rsna-intracranial-hemorrhage-detection/stage_2_train.csv"),
]

label_map = {}
label_file_used = None

for cand in LABEL_CANDIDATES:
    if cand.exists():
        label_file_used = cand
        with open(cand, newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []

            pid_col = None
            label_col = None

            for c in cols:
                cl = c.lower()
                if pid_col is None and cl in {"patient_id", "patientid", "id", "study_uid"}:
                    pid_col = c
                if label_col is None and cl in {"label", "any", "target", "abnormal"}:
                    label_col = c

            if pid_col is None:
                pid_col = cols[0]
            if label_col is None:
                for c in cols:
                    if c != pid_col:
                        label_col = c
                        break

            for row in reader:
                pid = str(row[pid_col]).strip()
                raw = row[label_col]
                try:
                    lab = int(float(raw))
                except Exception:
                    continue
                label_map[pid] = lab
        break

if label_file_used is None:
    raise FileNotFoundError("Could not find an RSNA label CSV in expected locations.")

records = []
for npz in sorted(CACHE_ROOT.rglob("slices_k21.npz")):
    patient_id = npz.parent.name
    if patient_id not in label_map:
        continue
    records.append({
        "patient_id": patient_id,
        "study_uid": patient_id,
        "path": str(npz),
        "npz_path": str(npz),
        "label": int(label_map[patient_id]),
        "source": "rsna",
        "preprocess": "rsna_windowed_lit",
    })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(records, f, indent=2)

n0 = sum(1 for r in records if r["label"] == 0)
n1 = sum(1 for r in records if r["label"] == 1)

print("label_file_used:", label_file_used)
print("wrote:", OUT_PATH)
print("count:", len(records))
print("normal:", n0)
print("abnormal:", n1)
