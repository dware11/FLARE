"""
Fusion evaluation on EA1141 paired dataset.

Compares three fusion strategies against individual model AUCs:
  - Strategy 1: Weighted average (α sweep 0.0 → 1.0)
  - Strategy 2: Learned linear fusion (fitted on val, evaluated on test)
  - Strategy 3: Confidence-weighted fusion

Usage:
    python ml/breast/fusion/eval_fusion.py \
        --mammo-csv results/mammo_ea1141.csv \
        --mri-csv   results/mri_ea1141.csv

Both CSVs must have columns: patient_id, true_label, p_cancer
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ml.breast.fusion.fuse import weighted_avg, confidence_weighted, LearnedLinearFusion


def load_predictions(csv_path: Path) -> dict[str, tuple[float, int]]:
    """Load {patient_id: (p_cancer, true_label)} from inference CSV."""
    result = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            result[row["patient_id"]] = (
                float(row["p_cancer"]),
                int(row["true_label"]),
            )
    return result


def compute_auc(scores: list, labels: list) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def align(
    mammo_preds: dict,
    mri_preds:   dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Return (p_mammo, p_mri, labels, patient_ids) for patients in both dicts."""
    common = [pid for pid in mammo_preds if pid in mri_preds and mammo_preds[pid][1] >= 0]
    p_mammo = np.array([mammo_preds[pid][0] for pid in common])
    p_mri   = np.array([mri_preds[pid][0]   for pid in common])
    labels  = np.array([mammo_preds[pid][1]  for pid in common])
    return p_mammo, p_mri, labels, common


def main(mammo_csv: Path, mri_csv: Path) -> None:
    mammo_preds = load_predictions(mammo_csv)
    mri_preds   = load_predictions(mri_csv)

    p_mammo, p_mri, labels, pids = align(mammo_preds, mri_preds)
    n = len(pids)
    print(f"\nEA1141 fusion evaluation — {n} paired patients\n{'='*50}")

    auc_mammo = compute_auc(p_mammo.tolist(), labels.tolist())
    auc_mri   = compute_auc(p_mri.tolist(),   labels.tolist())
    print(f"Mammography-only AUC : {auc_mammo:.4f}")
    print(f"MRI-only AUC         : {auc_mri:.4f}")

    # Strategy 1: weighted average sweep
    best_alpha, best_auc_wa = 0.5, 0.0
    for alpha in np.linspace(0.0, 1.0, 21):
        fused = weighted_avg(p_mammo, p_mri, alpha=alpha)
        auc   = compute_auc(fused.tolist(), labels.tolist())
        if auc > best_auc_wa:
            best_auc_wa, best_alpha = auc, float(alpha)
    print(f"\nStrategy 1 — Weighted avg (best α={best_alpha:.2f}): AUC={best_auc_wa:.4f}")

    # Strategy 2: learned linear (use first 60% as pseudo-val, rest as test)
    n_val = int(0.6 * n)
    fusion_model = LearnedLinearFusion()
    fusion_model.fit(p_mammo[:n_val], p_mri[:n_val], labels[:n_val])
    fused_learned = fusion_model.predict(p_mammo[n_val:], p_mri[n_val:])
    auc_learned   = compute_auc(fused_learned.tolist(), labels[n_val:].tolist())
    print(f"Strategy 2 — Learned linear ({fusion_model}): AUC={auc_learned:.4f}")

    # Strategy 3: confidence-weighted
    fused_conf = confidence_weighted(p_mammo, p_mri)
    auc_conf   = compute_auc(fused_conf.tolist(), labels.tolist())
    print(f"Strategy 3 — Confidence-weighted: AUC={auc_conf:.4f}")

    best_fused = max(best_auc_wa, auc_learned, auc_conf)
    baseline   = max(auc_mammo, auc_mri)
    gain       = best_fused - baseline
    print(f"\nBest single-modality AUC : {baseline:.4f}")
    print(f"Best fusion AUC           : {best_fused:.4f}  (gain: {gain:+.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mammo-csv", type=Path, required=True)
    ap.add_argument("--mri-csv",   type=Path, required=True)
    args = ap.parse_args()
    main(args.mammo_csv, args.mri_csv)
