#!/usr/bin/env python3
"""
Bar charts for MRI brain results (BRISC-style), matching the style of plot_ct_results_bars.py.

FLARE MRI uses one production model per task:
  • BRISC classification — EfficientNet-B0 (train_classification.py)
  • BRISC segmentation — Swin-UNet (train_swin_unet.py)
  • BraTS segmentation — 3D U-Net (segmentation_brats/train_local.py)

This script supports:
  1) Classification results.json (test_accuracy, test_f1, best_val_f1; optional checkpoint)
  2) Segmentation results.json (val_dice / test_dice / test_iou); use --pipeline for titles
  3) Binary CSV — same columns as CT (true_label, p_abnormal) if you export them

Examples:
  cd /scratch/bckk/flare/projects/FLARE

  # Classification (validation metrics from best_model.pth if present)
  python scripts/plot_mri_results_bars.py \\
    --results-json /scratch/bckk/flare/mri_brain/classification/outputs/output/results.json \\
    --checkpoint /scratch/bckk/flare/mri_brain/classification/outputs/output/best_model.pth \\
    --out /scratch/bckk/flare/mri_brain/classification/outputs/output/mri_classification_bars.png

  # BRISC segmentation (canonical: Swin-UNet — use swin_unet outputs path or --pipeline)
  python scripts/plot_mri_results_bars.py \\
    --results-json /scratch/bckk/flare/mri_brain/segmentation_brisc/outputs/swin_unet/results.json \\
    --out /scratch/bckk/flare/mri_brain/segmentation_brisc/outputs/swin_unet/mri_segmentation_bars.png \\
    --task segmentation \\
    --pipeline brisc-segmentation-swin

  # Entire MRI pipeline on one chart (BRISC cls + BRISC seg + optional BraTS; test metrics)
  python scripts/plot_mri_results_bars.py --whole-pipeline \\
    --mri-root /scratch/bckk/flare/mri_brain \\
    --out /scratch/bckk/flare/mri_brain/mri_pipeline_overview_bars.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
except ImportError as e:
    raise SystemExit("Install matplotlib: pip install matplotlib") from e


# --- optional: same binary CSV path as CT -------------------------------------------------
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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_v = values[order]
    n = len(sorted_v)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        mean_rank = (i + j + 1) / 2.0
        ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def roc_auc_trapezoid(y_true: np.ndarray, scores: np.ndarray) -> float:
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


def load_predictions_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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


# --- plotting -----------------------------------------------------------------------------
def plot_grouped_bars(
    series: Dict[str, Dict[str, float]],
    out_path: Path,
    title: str,
    metric_order: List[str],
    figsize: Tuple[float, float] = (9.0, 5.2),
    ylim: Tuple[float, float] = (0.0, 1.05),
) -> None:
    names = list(series.keys())
    first = next(iter(series.values()))
    keys = [k for k in metric_order if k in first]

    x = np.arange(len(keys))
    width = min(0.8 / max(len(names), 1), 0.35)

    fig, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(names):
        vals = [series[name].get(k, float("nan")) for k in keys]
        offset = width * (i - (len(names) - 1) / 2)
        bars = ax.bar(x + offset, vals, width, label=name)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    min(v + 0.02, ylim[1] - 0.02),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in keys], rotation=15, ha="right")
    ax.set_ylim(*ylim)
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


def plot_whole_pipeline_bars(
    bars: List[Tuple[str, float]],
    out_path: Path,
    title: str,
    figsize: Tuple[float, float] = (11.0, 5.0),
    bar_color: str = "#4472C4",
    ylim: Tuple[float, float] = (0.0, 1.05),
) -> None:
    """One row of bars; labels may contain newlines for two-line x tick text."""
    if not bars:
        raise ValueError("bars list is empty")
    labels = [b[0] for b in bars]
    vals = [float(b[1]) for b in bars]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    rects = ax.bar(x, vals, color=bar_color, width=0.72, edgecolor="white", linewidth=0.8)
    for rect, v in zip(rects, vals):
        if np.isfinite(v):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                min(v + 0.02, ylim[1] - 0.02),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Score (test)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.is_file():
            return p
    return None


def collect_whole_pipeline_bars(
    mri_root: Path,
    classification_results: Optional[Path],
    brisc_segmentation_results: Optional[Path],
    brats_segmentation_results: Optional[Path],
    flare_repo_root: Optional[Path] = None,
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Build (label, value) pairs for test-set headline metrics. Returns (bars, warnings).
    """
    warnings: List[str] = []
    bars: List[Tuple[str, float]] = []

    cls_candidates = [
        mri_root / "classification/outputs/output/results.json",
        mri_root / "classification/outputs/results.json",
    ]
    if flare_repo_root is not None:
        cls_candidates.append(
            flare_repo_root / "ml/brain/mri/classification/outputs/output/results.json"
        )
        cls_candidates.append(flare_repo_root / "ml/brain/mri/classification/outputs/results.json")
    cls_path = classification_results or _first_existing(cls_candidates)
    if cls_path:
        data = json.loads(cls_path.read_text(encoding="utf-8"))
        if "test_accuracy" in data and "test_f1" in data:
            bars.append(
                ("BRISC classification\n(EfficientNet-B0)\ntest accuracy", float(data["test_accuracy"]))
            )
            bars.append(
                ("BRISC classification\n(EfficientNet-B0)\ntest weighted F1", float(data["test_f1"]))
            )
        else:
            warnings.append(f"Skip classification: missing test keys in {cls_path}")
    else:
        warnings.append("Skip classification: results.json not found under mri_root")

    seg_candidates = [
        mri_root / "segmentation_brisc/outputs/swin_unet/results.json",
        mri_root / "segmentation_brisc/swin_unet/results.json",
    ]
    if flare_repo_root is not None:
        seg_candidates.append(
            flare_repo_root / "ml/brain/mri/segmentation_brisc/swin_unet/results.json"
        )
        seg_candidates.append(
            flare_repo_root / "ml/brain/mri/segmentation_brisc/outputs/swin_unet/results.json"
        )
    seg_path = brisc_segmentation_results or _first_existing(seg_candidates)
    if seg_path:
        data = json.loads(seg_path.read_text(encoding="utf-8"))
        if "test_dice" in data and "test_iou" in data:
            bars.append(("BRISC segmentation\n(Swin-UNet)\ntest Dice", float(data["test_dice"])))
            bars.append(("BRISC segmentation\n(Swin-UNet)\ntest mIoU", float(data["test_iou"])))
        else:
            warnings.append(f"Skip BRISC segmentation: missing test_dice/test_iou in {seg_path}")
    else:
        warnings.append("Skip BRISC segmentation: Swin-UNet results.json not found")

    brats_candidates = [
        mri_root / "segmentation_brats/outputs/results.json",
        mri_root / "segmentation_brats/results.json",
    ]
    if flare_repo_root is not None:
        brats_candidates.append(
            flare_repo_root / "ml/brain/mri/segmentation_brats/outputs/results.json"
        )
    brats_path = brats_segmentation_results or _first_existing(brats_candidates)
    if brats_path and brats_path.is_file():
        data = json.loads(brats_path.read_text(encoding="utf-8"))
        if "test_dice" in data and "test_iou" in data:
            bars.append(("BraTS segmentation\n(3D U-Net)\ntest Dice", float(data["test_dice"])))
            bars.append(("BraTS segmentation\n(3D U-Net)\ntest mIoU", float(data["test_iou"])))
        else:
            warnings.append(f"BraTS file present but missing test_dice/test_iou: {brats_path}")
    elif brats_segmentation_results:
        warnings.append(f"BraTS results not found: {brats_segmentation_results}")

    return bars, warnings


def _load_checkpoint_val_metrics(checkpoint: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    if checkpoint is None or not checkpoint.is_file():
        return None, None
    try:
        import torch
    except ImportError:
        print("Warning: torch not installed; cannot read --checkpoint for validation metrics.")
        return None, None
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return ckpt.get("val_acc"), ckpt.get("val_f1")


def build_classification_series(
    data: dict,
    checkpoint: Optional[Path],
    val_accuracy: Optional[float],
    val_f1: Optional[float],
) -> Dict[str, Dict[str, float]]:
    test_acc = float(data["test_accuracy"])
    test_f1 = float(data["test_f1"])
    va = data.get("val_accuracy")
    va = float(va) if va is not None else None
    vf = float(data.get("best_val_f1", float("nan")))

    ck_va, ck_vf = _load_checkpoint_val_metrics(checkpoint)
    if va is None and ck_va is not None:
        va = float(ck_va)
    if ck_vf is not None:
        vf = float(ck_vf)
    if val_accuracy is not None:
        va = float(val_accuracy)
    if val_f1 is not None:
        vf = float(val_f1)
    if va is None:
        va = float("nan")

    return {
        "Validation": {"accuracy": va, "weighted_f1": vf},
        "Test": {"accuracy": test_acc, "weighted_f1": test_f1},
    }


def build_segmentation_series(data: dict) -> Dict[str, Dict[str, float]]:
    return {
        "Performance": {
            "val_dice": float(data["best_val_dice"]),
            "test_dice": float(data["test_dice"]),
            "test_iou": float(data["test_iou"]),
        }
    }


def detect_task(data: dict) -> str:
    if "test_dice" in data and "test_iou" in data:
        return "segmentation"
    if "test_accuracy" in data and "test_f1" in data:
        return "classification"
    return "unknown"


# FLARE MRI: one production model per task (slide / report titles)
PIPELINE_TITLES = {
    "brisc-classification": "BRISC classification — EfficientNet-B0",
    "brisc-segmentation-swin": "BRISC segmentation — Swin-UNet",
    "brats-segmentation": "BraTS segmentation — 3D U-Net",
}


def default_title_for_results(
    task: str,
    pipeline: str,
    results_path: Optional[Path],
    data: dict,
) -> str:
    """Pick a chart title; --pipeline overrides when not 'auto'."""
    if pipeline != "auto" and pipeline in PIPELINE_TITLES:
        return PIPELINE_TITLES[pipeline]
    if task == "classification":
        return PIPELINE_TITLES["brisc-classification"]
    if task == "segmentation":
        path_s = str(results_path or "").lower()
        model_s = str(data.get("model", "")).lower()
        if "brats" in path_s:
            return PIPELINE_TITLES["brats-segmentation"]
        if "swin" in path_s or "swin" in model_s:
            return PIPELINE_TITLES["brisc-segmentation-swin"]
        # Older BRISC runs without model tag: neutral title (avoid naming alternate archs on figures)
        return "BRISC segmentation"
    return "MRI brain — model performance"


def main() -> None:
    ap = argparse.ArgumentParser(description="Bar plot MRI metrics from results.json or binary CSVs.")
    ap.add_argument(
        "--whole-pipeline",
        action="store_true",
        help="One chart: BRISC classification + BRISC Swin-UNet seg (+ BraTS if results.json exists). Uses test metrics.",
    )
    ap.add_argument(
        "--mri-root",
        type=Path,
        default=None,
        help="Root for default results paths (default: $MRI_BRAIN_ROOT or /scratch/bckk/flare/mri_brain).",
    )
    ap.add_argument(
        "--classification-results",
        type=Path,
        default=None,
        help="With --whole-pipeline: override path to classification results.json",
    )
    ap.add_argument(
        "--brisc-segmentation-results",
        type=Path,
        default=None,
        help="With --whole-pipeline: override path to Swin-UNet / BRISC seg results.json",
    )
    ap.add_argument(
        "--brats-segmentation-results",
        type=Path,
        default=None,
        help="With --whole-pipeline: BraTS 3D U-Net results.json (optional)",
    )
    ap.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="results.json from MRI classification or segmentation training",
    )
    ap.add_argument(
        "--task",
        choices=("auto", "classification", "segmentation", "binary_csv"),
        default="auto",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="best_model.pth (classification) to read val_acc and val_f1",
    )
    ap.add_argument(
        "--val-accuracy",
        type=float,
        default=None,
        help="Override validation accuracy if checkpoint is unavailable",
    )
    ap.add_argument(
        "--val-f1",
        type=float,
        default=None,
        help="Override validation weighted F1 if checkpoint is unavailable",
    )
    ap.add_argument("--val-csv", type=Path, default=None, help="Binary MRI/CT-style val predictions CSV")
    ap.add_argument("--test-csv", type=Path, default=None, help="Binary test predictions CSV")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("mri_metrics_bars.png"))
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument(
        "--pipeline",
        choices=("auto", "brisc-classification", "brisc-segmentation-swin", "brats-segmentation"),
        default="auto",
        help="Default chart title for FLARE MRI (one model per task: EfficientNet-B0, Swin-UNet, 3D U-Net).",
    )
    ap.add_argument(
        "--manual-json",
        type=Path,
        default=None,
        help='Merge extra series: {"Name": {"metric": float, ...}}',
    )
    args = ap.parse_args()

    if args.whole_pipeline:
        mri_root = args.mri_root or Path(
            os.environ.get("MRI_BRAIN_ROOT", "/scratch/bckk/flare/mri_brain")
        )
        flare_root = Path(__file__).resolve().parents[1]
        bars, warns = collect_whole_pipeline_bars(
            mri_root,
            args.classification_results,
            args.brisc_segmentation_results,
            args.brats_segmentation_results,
            flare_repo_root=flare_root,
        )
        for w in warns:
            print(f"Note: {w}")
        if not bars:
            raise SystemExit(
                "No metrics collected. Check --mri-root and results.json paths, or pass explicit --*-results overrides."
            )
        title = args.title or "FLARE MRI pipeline — test performance (BRISC + optional BraTS)"
        plot_whole_pipeline_bars(bars, args.out, title=title)
        print("Bars:")
        for lab, v in bars:
            print(f"  {lab.replace(chr(10), ' / ')}: {v:.4f}")
        return

    series: Dict[str, Dict[str, float]] = {}
    metric_order: List[str] = []
    default_title = "MRI brain — model performance"
    results_path_for_title: Optional[Path] = None
    data_for_title: dict = {}

    if args.task == "binary_csv" or (args.val_csv or args.test_csv):
        if args.val_csv and args.val_csv.is_file():
            yt, ps = load_predictions_csv(args.val_csv)
            series["Validation"] = metrics_at_threshold(yt, ps, args.threshold)
        if args.test_csv and args.test_csv.is_file():
            yt, ps = load_predictions_csv(args.test_csv)
            series["Test"] = metrics_at_threshold(yt, ps, args.threshold)
        metric_order = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
        default_title = "MRI — binary classification performance"

    elif args.results_json and args.results_json.is_file():
        data = json.loads(args.results_json.read_text(encoding="utf-8"))
        results_path_for_title = args.results_json
        data_for_title = data
        task = args.task if args.task != "auto" else detect_task(data)
        if task == "unknown":
            raise SystemExit(
                "Could not infer --task from results.json; use --task classification or segmentation."
            )
        if task == "classification":
            series = build_classification_series(
                data, args.checkpoint, args.val_accuracy, args.val_f1
            )
            metric_order = ["accuracy", "weighted_f1"]
        else:
            series = build_segmentation_series(data)
            metric_order = ["val_dice", "test_dice", "test_iou"]
        default_title = default_title_for_results(
            task, args.pipeline, results_path_for_title, data_for_title
        )

    if args.manual_json and args.manual_json.is_file():
        extra = json.loads(args.manual_json.read_text(encoding="utf-8"))
        if not isinstance(extra, dict):
            raise SystemExit("--manual-json must be an object: {series_name: {metric: float}}")
        for k, v in extra.items():
            if isinstance(v, dict):
                series[str(k)] = {str(mk): float(mv) for mk, mv in v.items()}

    if not series:
        raise SystemExit(
            "Provide --results-json and/or --val-csv / --test-csv / --manual-json with valid paths."
        )

    if not metric_order:
        metric_order = list(next(iter(series.values())).keys())

    title = args.title or default_title
    plot_grouped_bars(series, args.out, title=title, metric_order=metric_order)

    print("Metrics used:")
    for name, m in series.items():
        print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items() if np.isfinite(v)))


if __name__ == "__main__":
    main()
