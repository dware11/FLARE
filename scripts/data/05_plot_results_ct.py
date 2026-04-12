#!/usr/bin/env python3
"""
Generate training curves, ROC, threshold sweep, confusion matrix, and comparison CSV
from multimodal_finetune_ct results.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_RESULTS = Path("/scratch/bckk/flare/ct_brain/outputs/multimodal_finetune_ct/results.json")
DEFAULT_OUT = Path("/scratch/bckk/flare/ct_brain/outputs/multimodal_finetune_ct")


def _load_results(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thresh: float) -> np.ndarray:
    y_pred = (y_score >= thresh).astype(np.int64)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=np.float64)


def plot_training_curves(history: List[Dict[str, Any]], best_epoch: int, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    tr_loss = [h["tr_loss"] for h in history]
    vl_loss = [h["vl_loss"] for h in history]
    tr_auc = [h["tr_auc"] for h in history]
    vl_auc = [h["vl_auc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=12)

    ax1.plot(epochs, tr_loss, label="Train loss")
    ax1.plot(epochs, vl_loss, label="Val loss")
    if best_epoch > 0:
        ax1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="Best epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, tr_auc, label="Train AUC")
    ax2.plot(epochs, vl_auc, label="Val AUC")
    if best_epoch > 0:
        ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="Best epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    test_auc: float,
    out_path: Path,
) -> None:
    from sklearn.metrics import roc_curve

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(np.unique(y_true)) < 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "ROC unavailable (single class in test)", ha="center", va="center")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    auc_txt = f"{test_auc:.4f}" if np.isfinite(test_auc) else "nan"
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc_txt})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)

    y_pred = (y_score >= 0.5).astype(np.int64)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ax.scatter([fpr_pt], [tpr_pt], s=80, c="red", zorder=5, label="Op. point @ thresh=0.5")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_sweep(sweep: List[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = [row["threshold"] for row in sweep]
    sens = [row["sensitivity"] for row in sweep]
    spec = [row["specificity"] for row in sweep]
    acc = [row["accuracy"] for row in sweep]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(th, sens, marker="o", label="Sensitivity")
    ax.plot(th, spec, marker="s", label="Specificity")
    ax.plot(th, acc, marker="^", label="Accuracy")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold 0.5")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred Normal", "Pred Abnormal"],
        yticklabels=["True Normal", "True Abnormal"],
        title="Confusion @ threshold=0.5",
    )
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", color="w" if cm[i, j] > thresh else "k")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fmt_pct(v: Any) -> str:
    try:
        x = float(v)
        if not np.isfinite(x):
            return ""
        return f"{100.0 * x:.1f}%"
    except (TypeError, ValueError):
        return ""


def write_comparison_csv(results: Dict[str, Any], out_csv: Path) -> None:
    live_val_auc = results.get("best_val_auc", "")
    live_test_auc = results.get("test_auc", "")
    live_acc = results.get("test_acc", "")
    live_sens = results.get("test_sensitivity", "")
    live_spec = results.get("test_specificity", "")

    def _fmt_auc(v: Any) -> str:
        try:
            x = float(v)
            return f"{x:.4f}" if np.isfinite(x) else ""
        except (TypeError, ValueError):
            return ""

    rows = [
        ("resnet18_baseline", "CQ500", "", "0.8200", "", "", ""),
        ("densenet_gru_cq500", "CQ500", "", "0.8360", "", "", ""),
        ("rsna_exp1_16k", "RSNA", "", "0.9587", "87.7%", "88.9%", "86.6%"),
        ("rsna_windowed_exp2", "RSNA", "0.9595", "0.9459", "87.0%", "86.2%", "87.8%"),
        ("cq500_generalization", "CQ500 (eval)", "", "0.7359", "", "57.1%", "90.6%"),
        (
            "multimodal_finetune",
            "Multimodal",
            _fmt_auc(live_val_auc),
            _fmt_auc(live_test_auc),
            _fmt_pct(live_acc),
            _fmt_pct(live_sens),
            _fmt_pct(live_spec),
        ),
    ]
    header = ["Experiment", "Dataset", "Val AUC", "Test AUC", "Accuracy", "Sensitivity", "Specificity"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("\n=== Comparison table ===", flush=True)
    colw = [22, 18, 10, 10, 12, 14, 14]
    hdr = " | ".join(h.ljust(colw[i]) for i, h in enumerate(header))
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for row in rows:
        print(" | ".join(str(c).ljust(colw[i]) for i, c in enumerate(row)), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot CT fine-tune results.")
    ap.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    data = _load_results(args.results_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = data.get("history", [])
    best_epoch = int(data.get("best_epoch", -1))
    plot_training_curves(
        history,
        best_epoch,
        args.output_dir / "training_curves.png",
        "CT Fine-tune — multimodal_finetune_ct",
    )

    y_true = np.asarray(data.get("test_y_true", []), dtype=np.int64)
    y_score = np.asarray(data.get("test_y_score", []), dtype=np.float64)
    ta = data.get("test_auc")
    try:
        test_auc = float(ta) if ta is not None else float("nan")
    except (TypeError, ValueError):
        test_auc = float("nan")

    plot_roc(y_true, y_score, test_auc, args.output_dir / "roc_curve.png")

    sweep = data.get("threshold_sweep", [])
    if sweep:
        plot_threshold_sweep(sweep, args.output_dir / "threshold_sweep.png")

    if y_true.size > 0:
        cm = _confusion_at_threshold(y_true, y_score, 0.5)
        plot_confusion(cm, args.output_dir / "confusion_matrix.png")

    write_comparison_csv(data, args.output_dir / "comparison_table.csv")
    print(f"\nSaved plots and {args.output_dir / 'comparison_table.csv'}", flush=True)


if __name__ == "__main__":
    main()
