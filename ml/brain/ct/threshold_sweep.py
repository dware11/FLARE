"""
Threshold sweep on the validation set using the saved best checkpoint.
Reports accuracy, sensitivity, specificity, precision, F1 for each threshold.
No training; use same DATA_ROOT and checkpoint as your baseline run.

Also prints **suggested operating points** (validation set):
  - threshold maximizing **F1**
  - threshold maximizing **Youden's J** (sensitivity + specificity - 1)
  - threshold with **sensitivity >= floor** that maximizes **F1** among those

Use these after training; keep **ROC-AUC** as the threshold-free summary metric.

How to run on Delta:
  cd /scratch/bckk/flare/projects/FLARE
  export DATA_ROOT=/scratch/bckk/flare/ct_brain

  python ml/brain/ct/threshold_sweep.py

  python ml/brain/ct/threshold_sweep.py --sens-floor 0.7 --dense-steps 200

Training checkpoints are selected by **val_auc** in train_ct.py when both classes appear in val.
For deployment, set infer ABNORMAL_THRESHOLD to your chosen sweep threshold.
"""
import argparse
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS
from ml.brain.ct.dataset_ct import CTDataset
from ml.brain.ct.model_ct import (
    build_ct_model,
    build_ct_sequence_model,
    load_sequence_weights_compat,
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


def _preds_at_threshold(probs: List[float], thresh: float) -> List[int]:
    return [1 if p >= thresh else 0 for p in probs]


def _dense_threshold_sweep(
    all_y: List[int],
    all_probs_pos: List[float],
    n_steps: int = 200,
) -> List[Tuple[float, float, float, float, float, float, float]]:
    """Return list of (thresh, acc, sens, spec, prec, f1, youden_j)."""
    out = []
    lo, hi = 0.02, 0.98
    for t in np.linspace(lo, hi, n_steps):
        preds = _preds_at_threshold(all_probs_pos, float(t))
        acc, sens, spec, prec, f1 = _metrics(all_y, preds)
        j = sens + spec - 1.0
        out.append((float(t), acc, sens, spec, prec, f1, j))
    return out


def _print_recommendations(
    all_y: List[int],
    all_probs_pos: List[float],
    dense_steps: int,
    sens_floor: float,
) -> None:
    grid = _dense_threshold_sweep(all_y, all_probs_pos, n_steps=dense_steps)

    best_f1 = max(grid, key=lambda r: r[5])
    best_j = max(grid, key=lambda r: r[6])

    feasible = [r for r in grid if r[2] >= sens_floor]
    best_f1_under_floor = max(feasible, key=lambda r: r[5]) if feasible else None

    print()
    print("--- Suggested thresholds (validation set; tune on val, report on test) ---")
    print(
        f"  Max F1:        t={best_f1[0]:.3f}  F1={best_f1[5]:.4f}  "
        f"sens={best_f1[2]:.4f}  spec={best_f1[3]:.4f}  prec={best_f1[4]:.4f}"
    )
    print(
        f"  Max Youden J:  t={best_j[0]:.3f}  J={best_j[6]:.4f}  "
        f"sens={best_j[2]:.4f}  spec={best_j[3]:.4f}  F1={best_j[5]:.4f}"
    )
    if best_f1_under_floor is not None:
        r = best_f1_under_floor
        print(
            f"  Max F1 w/ sens>={sens_floor}: t={r[0]:.3f}  F1={r[5]:.4f}  "
            f"sens={r[2]:.4f}  spec={r[3]:.4f}"
        )
    else:
        print(f"  Max F1 w/ sens>={sens_floor}: (no threshold in grid meets floor; lower floor or check model)")
    print("--- Use AUC as threshold-free quality; pick one t above for demos / clinical-style reporting ---")
    print()


def main(
    checkpoint: Path = None,
    batch_size: int = 16,
    agg: Optional[str] = None,
    thresholds: list = None,
    out_file: Path = None,
    dense_steps: int = 200,
    sens_floor: float = 0.5,
    no_recommend: bool = False,
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
    rnn_type = ckpt.get("rnn_type", "gru")
    sequence_pool = ckpt.get("sequence_pool", "max")
    sequence_topk = ckpt.get("sequence_topk", 3)
    rnn_dropout = ckpt.get("rnn_dropout", 0.25)
    use_thickness = ckpt.get("use_thickness", False)
    state = ckpt["model"]

    if model_kind == "sequence":
        if "head.weight" in state and "slice_head.weight" not in state:
            sequence_pool = "center"
        model = build_ct_sequence_model(
            rnn_type=rnn_type,
            sequence_pool=sequence_pool,
            topk=sequence_topk,
            rnn_dropout=rnn_dropout,
            use_thickness=use_thickness,
        )
        load_sequence_weights_compat(model, state)
    else:
        model = build_ct_model()
        model.load_state_dict(state, strict=True)
    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    all_y = []
    all_probs_pos = []

    model.eval()
    with torch.no_grad():
        for X, y, _, thick in val_loader:
            X = X.to(device)
            thick = thick.to(device)
            if not use_thickness:
                thick = None
            logits = patient_logits_from_model(model, X, agg=agg_eff, thickness=thick)
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
        preds = _preds_at_threshold(all_probs_pos, thresh)
        acc, sens, spec, prec, f1 = _metrics(all_y, preds)
        lines.append(f"{thresh:<10.2f} {acc:<8.4f} {sens:<8.4f} {spec:<8.4f} {prec:<8.4f} {f1:<8.4f}")

    for line in lines:
        print(line)

    if not no_recommend:
        _print_recommendations(all_y, all_probs_pos, dense_steps=dense_steps, sens_floor=sens_floor)

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
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Space-separated thresholds for the printed table",
    )
    ap.add_argument("--out", type=Path, default=None, help="Save table to this file (e.g. ct_threshold_sweep_val.txt)")
    ap.add_argument("--dense-steps", type=int, default=200, help="Grid size for F1 / Youden / sens-floor search")
    ap.add_argument(
        "--sens-floor",
        type=float,
        default=0.5,
        help="Minimum sensitivity when reporting 'max F1 under sens floor'",
    )
    ap.add_argument("--no-recommend", action="store_true", help="Skip F1/Youden summary block")
    args = ap.parse_args()
    main(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        agg=args.agg,
        thresholds=args.thresholds,
        out_file=args.out,
        dense_steps=args.dense_steps,
        sens_floor=args.sens_floor,
        no_recommend=args.no_recommend,
    )
