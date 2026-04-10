#!/usr/bin/env python3
"""
Bar charts for CT model results from prediction CSVs (val / test).

Example:
  cd /scratch/bckk/flare/projects/FLARE
  python scripts/plot_ct_results_bars.py \\
    --val-csv /scratch/bckk/flare/ct_brain/outputs/ct_val_predictions.csv \\
    --test-csv /scratch/bckk/flare/ct_brain/outputs/ct_test_predictions.csv \\
    --out /scratch/bckk/flare/ct_brain/outputs/ct_metrics_bars.png

Optional: pass --manual-json for a second series (e.g. baseline) to compare.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
except ImportError as e:
    raise SystemExit("Install matplotlib: pip install matplotlib") from e


def _confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def _metrics_from_conf(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    if prec + sens > 0:
        f1 = 2.0 * prec * sens / (prec + sens)
    else:
        f1 = 0.0
    return {"accuracy": acc, "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}


def roc_auc_trapezoid(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Binary ROC AUC via rank / Mann–Whitney; robust tie handling."""
    y = y_true.astype(np.int64)
    s = scores.astype(np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = _average_ranks(s)
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average rank for each element (1-based), ties get mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_v = values[order]
    n = len(sorted_v)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        mean_rank = (i + j + 1) / 2.0  # 1-based average of positions i..j-1
        ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def load_predictions_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return y_true, y_prob_abnormal (float in [0,1])."""
    y_true: List[int] = []
    y_score: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            y_true.append(int(row["true_label"]))
            y_score.append(float(row["p_abnormal"]))
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_score, dtype=np.float64)


def metrics_at_threshold(y_true: np.ndarray, p_abn: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (p_abn >= thr).astype(np.int64)
    tp, tn, fp, fn = _confusion_binary(y_true, y_pred)
    out = _metrics_from_conf(tp, tn, fp, fn)
    out["roc_auc"] = roc_auc_trapezoid(y_true, p_abn)
    return out


def plot_grouped_bars(
    series: Dict[str, Dict[str, float]],
    out_path: Path,
    title: str,
    figsize: Tuple[float, float] = (9.0, 5.2),
) -> None:
    """
    series: name -> {metric_key -> value in [0,1]}
    Plots all metrics except roc_auc as bars; roc_auc as separate if present.
    """
    metric_order = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
    names = list(series.keys())
    first = next(iter(series.values()))
    keys = [k for k in metric_order if k in first]

    x = np.arange(len(keys))
    width = min(0.8 / max(len(names), 1), 0.35)

    fig, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(names):
        vals = [series[name].get(k, 0.0) for k in keys]
        offset = width * (i - (len(names) - 1) / 2)
        bars = ax.bar(x + offset, vals, width, label=name)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    min(v + 0.02, 0.98),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in keys], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bar plot CT metrics from prediction CSVs.")
    ap.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Validation predictions CSV (patient_id, true_label, p_abnormal, ...)",
    )
    ap.add_argument("--test-csv", type=Path, default=None, help="Test predictions CSV")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("ct_metrics_bars.png"),
        help="Output image path (.png or .pdf)",
    )
    ap.add_argument("--title", type=str, default="CT brain hemorrhage — model performance")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for hard labels (accuracy, sens, spec, prec, f1)",
    )
    ap.add_argument(
        "--manual-json",
        type=Path,
        default=None,
        help='Optional JSON: {"Baseline": {"accuracy": 0.7, ...}, "Ours": {...}}',
    )
    args = ap.parse_args()

    series: Dict[str, Dict[str, float]] = {}

    if args.val_csv and args.val_csv.is_file():
        yt, ps = load_predictions_csv(args.val_csv)
        series["Validation"] = metrics_at_threshold(yt, ps, args.threshold)
    if args.test_csv and args.test_csv.is_file():
        yt, ps = load_predictions_csv(args.test_csv)
        series["Test"] = metrics_at_threshold(yt, ps, args.threshold)

    if args.manual_json and args.manual_json.is_file():
        extra = json.loads(args.manual_json.read_text(encoding="utf-8"))
        if not isinstance(extra, dict):
            raise SystemExit("--manual-json must be a JSON object: {series_name: {metric: float}}")
        for k, v in extra.items():
            if isinstance(v, dict):
                series[str(k)] = {str(mk): float(mv) for mk, mv in v.items()}

    if not series:
        raise SystemExit("Provide at least one of --val-csv / --test-csv / --manual-json with valid paths.")

    plot_grouped_bars(series, args.out, title=args.title)

    print("Metrics used:")
    for name, m in series.items():
        print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items() if np.isfinite(v)))


if __name__ == "__main__":
    main()
