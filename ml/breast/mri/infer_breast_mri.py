"""
Inference for the 3D breast MRI (DCE-MRI) classifier.

Usage:
    python ml/breast/mri/infer_breast_mri.py --split test --out-csv results/mri_test.csv
    python ml/breast/mri/infer_breast_mri.py --npz /path/to/patient.npz
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.config import BREAST_MRI_CKPT
from ml.breast.mri.model_breast_mri import build_breast_mri_model
from ml.breast.mri.dataset_breast_mri import BreastMRIDataset


def load_model(checkpoint: Path = BREAST_MRI_CKPT) -> torch.nn.Module:
    model = build_breast_mri_model(in_channels=3, num_classes=2)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def run_batch(split: str = "test", out_csv: Path = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    ds = BreastMRIDataset(split=split, augment=False)

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, label, patient_id = ds[i]
            x = x.unsqueeze(0).to(device)
            prob = torch.softmax(model(x), dim=1)[0, 1].item()
            rows.append({
                "patient_id": patient_id,
                "true_label": label,
                "pred_label": int(prob >= 0.5),
                "p_cancer":   round(prob, 4),
            })

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient_id", "true_label", "pred_label", "p_cancer"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} predictions → {out_csv}")
    else:
        for r in rows:
            print(r)


def run_single(npz_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    arr = np.load(npz_path)["arr"].astype(np.float32)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0, 1].item()
    print(f"{npz_path.name}: P_cancer={prob:.4f} → {'cancer' if prob >= 0.5 else 'normal'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="test")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--npz",     type=Path, default=None)
    args = ap.parse_args()
    if args.npz:
        run_single(args.npz)
    else:
        run_batch(split=args.split, out_csv=args.out_csv)
