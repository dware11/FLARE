"""
Inference for the 3D DBT mammography classifier.

Per-patient prediction: run model on all available views (LCC/RCC/LMLO/RMLO),
then average softmax scores across views → single P_cancer ∈ [0,1] per patient.

Usage:
    python ml/breast/mammo/infer_mammo.py --split test --out-csv results/mammo_test.csv
    python ml/breast/mammo/infer_mammo.py --npz /path/to/view.npz  # single view
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.config import BREAST_MAMMO_CKPT
from ml.breast.mammo.model_mammo import build_mammo_model
from ml.breast.mammo.dataset_mammo import MammoDataset


def load_model(checkpoint: Path = BREAST_MAMMO_CKPT) -> torch.nn.Module:
    model = build_mammo_model(num_classes=2)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def predict_npz(model: torch.nn.Module, npz_path: Path, device: torch.device) -> float:
    """Return P_cancer for a single cached DBT view NPZ."""
    arr = np.load(npz_path)["arr"].astype(np.float32)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 32, 256, 256)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return prob


def run_batch(split: str = "test", out_csv: Path = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    ds = MammoDataset(split=split, augment=False)

    # Collect per-view scores, then aggregate per patient
    patient_scores: dict[str, list[float]] = defaultdict(list)
    patient_labels: dict[str, int] = {}

    for i in range(len(ds)):
        x, label, patient_id, view = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        patient_scores[patient_id].append(prob)
        if label >= 0:
            patient_labels[patient_id] = label

    rows = []
    for pid, scores in patient_scores.items():
        p_cancer = float(np.mean(scores))
        pred_label = 1 if p_cancer >= 0.5 else 0
        true_label = patient_labels.get(pid, -1)
        rows.append({
            "patient_id": pid,
            "true_label": true_label,
            "pred_label": pred_label,
            "p_cancer": round(p_cancer, 4),
            "n_views": len(scores),
        })

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient_id", "true_label", "pred_label", "p_cancer", "n_views"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} patient predictions → {out_csv}")
    else:
        for r in rows:
            print(r)


def run_single(npz_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    p = predict_npz(model, npz_path, device)
    print(f"{npz_path.name}: P_cancer={p:.4f} → {'cancer' if p >= 0.5 else 'normal'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",    default="test")
    ap.add_argument("--out-csv",  type=Path, default=None)
    ap.add_argument("--npz",      type=Path, default=None)
    args = ap.parse_args()
    if args.npz:
        run_single(args.npz)
    else:
        run_batch(split=args.split, out_csv=args.out_csv)
