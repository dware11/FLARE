"""
Threshold sweep on the validation set using the saved best checkpoint.
Reports accuracy, sensitivity, specificity, precision, F1 for each threshold.
No training; use same DATA_ROOT and checkpoint as your baseline run.

How to run on Delta:
  cd /scratch/bckk/flare/projects/FLARE
  export DATA_ROOT=/scratch/bckk/flare/ct_brain

  # Default thresholds 0.45, 0.5, 0.55, 0.6, 0.65:
  python ml/brain/ct/threshold_sweep.py

  # Custom thresholds and save table:
  python ml/brain/ct/threshold_sweep.py --thresholds 0.4 0.5 0.55 0.6 0.65 0.7 --out outputs/ct_threshold_sweep_val.txt

  # Different checkpoint (e.g. after another run):
  python ml/brain/ct/threshold_sweep.py --checkpoint /scratch/bckk/flare/ct_brain/outputs/ct_baseline_best.pt

Training: Training uses cross-entropy; the decision boundary is effectively 0.5 (argmax).
To use a chosen threshold at inference, change infer.py where pred = argmax(probs) to
pred = 1 if probs[1] >= YOUR_THRESHOLD else 0 (see infer.py ~line 206, 284, 424).
"""
import argparse
from pathlib import Path
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS
from ml.brain.ct.dataset_ct import CTDataset
from ml.brain.ct.model_ct import (
    build_ct_model,
    build_ct_sequence_model,
    patient_logits_from_model,
)

DEFAULT_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]


def _metrics(all_y, preds):
    """Compute acc, sens, spec, prec, f1 from binary y and preds (0=normal, 1=abnormal)."""
    tp = sum(1 for i in range(len(all_y)) if preds[i] == 1 and all_y[i] == 1)
    tn = sum(1 for i in range(len(all_y)) if preds[i] == 0 and all_y[i] == 0)
    fp = sum(1 for i in range(len(all_y)) if preds[i] == 1 and all_y[i] == 0)
    fn = sum(1 for i in range(len(all_y)) if preds[i] == 0 and all_y[i] == 1)
    n = len(all_y)
    acc = (tp + tn) / n if n else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return acc, sens, spec, prec, f1


def main(
    checkpoint: Path = None,
    batch_size: int = 16,
    agg: Optional[str] = None,
    thresholds: list = None,
    out_file: Path = None,
) -> None:
    if checkpoint is None:
        checkpoint = OUTPUTS / "ct_baseline_best.pt"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    val_ds = CTDataset(split="val")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    model_kind = ckpt.get("model_kind", "legacy")
    agg_eff = agg if agg is not None else ckpt.get("agg", "max")
    rnn_type = ckpt.get("rnn_type", "lstm")

    if model_kind == "sequence":
        model = build_ct_sequence_model(rnn_type=rnn_type)
    else:
        model = build_ct_model()
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    all_y = []
    all_probs_pos = []

    model.eval()
    with torch.no_grad():
        for X, y, _ in val_loader:
            X = X.to(device)
            logits = patient_logits_from_model(model, X, agg=agg_eff)
            probs = torch.softmax(logits, dim=1)
            for i in range(y.size(0)):
                if y[i].item() not in (0, 1):
                    continue
                all_y.append(y[i].item())
                all_probs_pos.append(probs[i, 1].item())

    if not all_y:
        print("No validation samples with label 0 or 1.")
        return

    lines = []
    header = f"{'Threshold':<10} {'Acc':<8} {'Sens':<8} {'Spec':<8} {'Prec':<8} {'F1':<8}"
    lines.append(header)
    lines.append("-" * len(header))

    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in all_probs_pos]
        acc, sens, spec, prec, f1 = _metrics(all_y, preds)
        lines.append(f"{thresh:<10.2f} {acc:<8.4f} {sens:<8.4f} {spec:<8.4f} {prec:<8.4f} {f1:<8.4f}")

    for line in lines:
        print(line)

    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Threshold sweep on CT validation set")
    ap.add_argument("--checkpoint", type=Path, default=None, help=f"Default: {OUTPUTS}/ct_baseline_best.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--agg",
        type=str,
        default=None,
        choices=["mean", "max"],
        help="Legacy models only; default is value stored in checkpoint or max",
    )
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                    help="Space-separated thresholds, e.g. 0.45 0.5 0.55 0.6 0.65")
    ap.add_argument("--out", type=Path, default=None, help="Save table to this file (e.g. ct_threshold_sweep_val.txt)")
    args = ap.parse_args()
    main(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        agg=args.agg,
        thresholds=args.thresholds,
        out_file=args.out,
    )
