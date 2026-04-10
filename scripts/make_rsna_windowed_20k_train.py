import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

rebuilt_manifest = Path("/scratch/bckk/flare/ct_brain/meta/ct_rsna_manifest_windowed_rebuilt.json")
old_val = Path("/scratch/bckk/flare/ct_brain/meta/subsets/rsna_ready_k21_16k_val.json")
old_test = Path("/scratch/bckk/flare/ct_brain/meta/subsets/rsna_ready_k21_16k_test.json")

out_dir = Path("/scratch/bckk/flare/ct_brain/meta/subsets")
out_train = out_dir / "rsna_windowed_k21_20k_train.json"
out_val = out_dir / "rsna_windowed_k21_20k_val.json"
out_test = out_dir / "rsna_windowed_k21_20k_test.json"

data = json.loads(rebuilt_manifest.read_text())
val = json.loads(old_val.read_text())
test = json.loads(old_test.read_text())

held_out = {r["patient_id"] for r in val} | {r["patient_id"] for r in test}
train_pool = [r for r in data if r["patient_id"] not in held_out]

normal = [r for r in train_pool if int(r["label"]) == 0]
abnormal = [r for r in train_pool if int(r["label"]) == 1]

random.shuffle(normal)
random.shuffle(abnormal)

target = 10000

train = normal[:target] + abnormal[:target]
random.shuffle(train)

out_dir.mkdir(parents=True, exist_ok=True)

out_train.write_text(json.dumps(train, indent=2))
out_val.write_text(json.dumps(val, indent=2))
out_test.write_text(json.dumps(test, indent=2))

print("train:", len(train))
print("normal:", sum(1 for r in train if int(r["label"]) == 0))
print("abnormal:", sum(1 for r in train if int(r["label"]) == 1))
print("saved:", out_train)
