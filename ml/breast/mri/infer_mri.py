"""Run exam-level inference; write CSV: exam_id, patient_id, label, p_mri."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.breast.mri.config_mri import (
    BREAST_MRI_MANIFEST_PATH,
    BREAST_MRI_PRED_CSV,
    MRI_NUM_CHANNELS,
    ensure_mri_dirs,
)
from ml.breast.mri.dataset_mri import BreastMRIDataset
from ml.breast.mri.model_mri import build_mri_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=BREAST_MRI_MANIFEST_PATH)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    out = args.out_csv or BREAST_MRI_PRED_CSV
    ensure_mri_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_ch = int(ckpt.get("in_channels", MRI_NUM_CHANNELS))
    model = build_mri_model(in_channels=in_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = BreastMRIDataset(args.manifest, split="all", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows: list[dict] = []
    with torch.no_grad():
        for x, y, eid, pid, _ in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.sigmoid(logits).view(-1).cpu().numpy()
            for i in range(x.size(0)):
                rows.append(
                    {
                        "exam_id": str(eid[i]),
                        "patient_id": str(pid[i]),
                        "label": float(y[i].item()),
                        "p_mri": float(p[i]),
                    }
                )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exam_id", "patient_id", "label", "p_mri"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
