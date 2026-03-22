"""Run DBT inference; write view-level and exam-level prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.breast.mammo.config_dbt import (
    DBT_DEFAULT_VIEW_AGG,
    DBT_EXAM_PRED_CSV,
    DBT_VIEW_PRED_CSV,
)
from ml.breast.mammo.dataset_dbt import DBTViewDataset
from ml.breast.mammo.model_dbt import build_dbt_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--view-out-csv", type=Path, default=DBT_VIEW_PRED_CSV)
    ap.add_argument("--exam-out-csv", type=Path, default=DBT_EXAM_PRED_CSV)
    ap.add_argument("--view-agg", type=str, default=DBT_DEFAULT_VIEW_AGG, choices=["mean", "max"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_dbt_model(num_classes=2).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = DBTViewDataset(split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    view_rows: list[dict] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    labels: dict[tuple[str, str], float] = {}

    with torch.no_grad():
        for x, y, exam_id, patient_id, view in loader:
            x = x.to(device)
            logits = model(x)
            p1 = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

            for i in range(len(p1)):
                key = (str(exam_id[i]), str(patient_id[i]))
                prob = float(p1[i])
                lab = float(y[i].item())
                grouped[key].append(prob)
                labels[key] = lab
                view_rows.append(
                    {
                        "exam_id": key[0],
                        "patient_id": key[1],
                        "view": str(view[i]),
                        "label": lab,
                        "p_dbt_view": prob,
                    }
                )

    exam_rows: list[dict] = []
    for (exam_id, patient_id), probs in grouped.items():
        agg = max(probs) if args.view_agg == "max" else (sum(probs) / len(probs))
        exam_rows.append(
            {
                "exam_id": exam_id,
                "patient_id": patient_id,
                "label": labels[(exam_id, patient_id)],
                "p_dbt_exam": float(agg),
            }
        )

    args.view_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.view_out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["exam_id", "patient_id", "view", "label", "p_dbt_view"]
        )
        w.writeheader()
        w.writerows(view_rows)

    args.exam_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.exam_out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exam_id", "patient_id", "label", "p_dbt_exam"])
        w.writeheader()
        w.writerows(exam_rows)

    print(f"Wrote {len(view_rows)} view rows to {args.view_out_csv}")
    print(f"Wrote {len(exam_rows)} exam rows to {args.exam_out_csv}")


if __name__ == "__main__":
    main()
