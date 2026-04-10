#!/usr/bin/env python3
"""
Evaluate CT brain models (ResNet18 middle-slice baseline, DenseNet121 legacy, DenseNet+BiGRU)
on a manifest-backed train/val/test split. Writes metrics JSON and ROC/confusion plots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.brain.ct.dataset_ct import CTDataset, ct_batch_collate, unpack_ct_batch  # noqa: E402
from ml.brain.ct.infer import _normalize_state_dict_keys  # noqa: E402
from ml.brain.ct.model_ct import (  # noqa: E402
    build_ct_model,
    build_ct_sequence_model,
    is_sequence_ct_model,
    load_sequence_weights_compat,
    patient_logits_from_model,
)
from ml.brain.ct.threshold_util import resolve_abnormal_threshold  # noqa: E402
from src.config import CT_MANIFEST, CT_RSNA_MANIFEST  # noqa: E402


def _torch_load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_manifest_path(dataset: str, manifest_override: Optional[Path]) -> Path:
    if manifest_override is not None:
        return Path(manifest_override)
    if dataset == "cq500":
        return CT_MANIFEST
    if dataset == "rsna":
        return CT_RSNA_MANIFEST
    if dataset == "combined":
        return _build_combined_manifest()
    raise ValueError(f"Unknown dataset={dataset!r}")


def _build_combined_manifest() -> Path:
    cache_dir = ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "ct_eval_combined_manifest.json"
    merged: List[dict] = []
    seen_paths: set = set()
    for path in (CT_MANIFEST, CT_RSNA_MANIFEST):
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            p = str(Path(e.get("path", "")).resolve())
            if p in seen_paths:
                continue
            seen_paths.add(p)
            merged.append(e)
    if not merged:
        raise FileNotFoundError(
            f"No entries to merge; check CT_MANIFEST={CT_MANIFEST} and CT_RSNA_MANIFEST={CT_RSNA_MANIFEST}"
        )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    return out


def load_model_for_eval(
    checkpoint: Path,
    model: str,
) -> Tuple[nn.Module, str, str, Optional[int], Dict[str, Any]]:
    ckpt = _torch_load_checkpoint(checkpoint)
    meta: Dict[str, Any] = {}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model"}
    elif isinstance(ckpt, dict) and any(
        k in ckpt for k in ("conv1.weight", "features.conv0.weight", "backbone.conv0.weight", "slice_head.weight")
    ):
        state = ckpt
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint}")
    state = _normalize_state_dict_keys(state)

    agg = meta.get("agg", "max") if meta else "max"
    k_slices = meta.get("k_slices")
    if k_slices is not None:
        k_slices = int(k_slices)

    model = model.strip().lower()
    if model == "resnet18":
        m = build_ct_model(num_classes=2, pretrained=False)
        r = m.load_state_dict(state, strict=False)
        if r.missing_keys:
            print(f"[warn] resnet18 load missing_keys (first 8): {r.missing_keys[:8]}", flush=True)
        if r.unexpected_keys:
            print(f"[warn] resnet18 load unexpected_keys (first 8): {r.unexpected_keys[:8]}", flush=True)
        return m, "resnet18", agg, k_slices, meta

    if model == "densenet_legacy":
        m = build_ct_model(pretrained=False)
        try:
            m.load_state_dict(state, strict=True)
        except RuntimeError:
            m.load_state_dict(state, strict=False)
            print("[warn] densenet_legacy strict=False (key mismatch)", flush=True)
        return m, "legacy", agg, k_slices, meta

    if model == "densenet_gru":
        rnn_type = meta.get("rnn_type", "gru")
        sequence_pool = meta.get("sequence_pool", "topk_mean")
        sequence_topk = int(meta.get("sequence_topk", 3))
        rnn_dropout = float(meta.get("rnn_dropout", 0.25))
        use_thickness = bool(meta.get("use_thickness", False))
        if "head.weight" in state and "slice_head.weight" not in state:
            sequence_pool = "center"
        m = build_ct_sequence_model(
            rnn_type=rnn_type,
            sequence_pool=sequence_pool,
            topk=sequence_topk,
            rnn_dropout=rnn_dropout,
            use_thickness=use_thickness,
        )
        load_sequence_weights_compat(m, state)
        return m, "sequence", agg, k_slices, meta

    raise ValueError(f"Unknown --model {model!r}; use resnet18 | densenet_legacy | densenet_gru")


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    agg: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    ys: List[int] = []
    probs_ab: List[float] = []
    ids: List[str] = []
    use_thick = is_sequence_ct_model(model) and getattr(model, "thickness_fusion", None) is not None
    for batch in loader:
        x, y, pids, thick = unpack_ct_batch(batch)
        x = x.to(device)
        y = y.to(device)
        thick_t = thick.to(device) if use_thick else None
        logits = patient_logits_from_model(model, x, agg=agg, thickness=thick_t)
        p = torch.softmax(logits, dim=1)
        for i in range(y.size(0)):
            yi = int(y[i].item())
            if yi not in (0, 1):
                continue
            ys.append(yi)
            probs_ab.append(float(p[i, 1].item()))
            ids.append(str(pids[i]))
    return np.asarray(ys, dtype=np.int64), np.asarray(probs_ab, dtype=np.float64), ids


def metrics_at_threshold(y_true: np.ndarray, p_ab: np.ndarray, thr: float) -> Dict[str, Any]:
    y_pred = (p_ab >= thr).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out: Dict[str, Any] = {
        "threshold": thr,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1_binary": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "missed_abnormals": int(fn),
        "false_abnormals": int(fp),
    }
    return out


def safe_roc_auc(y_true: np.ndarray, p_ab: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p_ab))


def split_counts(manifest_path: Path, expected_k: Optional[int], seed: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sp in ("train", "val", "test"):
        ds = CTDataset(
            manifest_path=manifest_path,
            split=sp,
            expected_k_slices=expected_k,
            seed=seed,
        )
        counts[sp] = len(ds)
    return counts


def _save_roc_png(path: Path, y_true: np.ndarray, p_ab: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, p_ab)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {safe_roc_auc(y_true, p_ab):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (abnormal vs normal)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_cm_png(path: Path, cm: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Pred Normal", "Pred Abnormal"],
        yticklabels=["True Normal", "True Abnormal"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="w" if cm[i, j] > thresh else "k")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate FLARE CT models on test split.")
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "densenet_legacy", "densenet_gru"],
        help="Architecture to instantiate (must match checkpoint).",
    )
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt / .pth checkpoint.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Primary P(abnormal) threshold.")
    ap.add_argument(
        "--secondary-threshold",
        type=float,
        default=None,
        help="Optional second threshold.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="combined",
        choices=["cq500", "rsna", "combined"],
        help="Which manifest to use.",
    )
    ap.add_argument("--manifest", type=Path, default=None, help="Override manifest JSON path.")
    ap.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--k-slices", type=int, default=None)
    ap.add_argument("--preliminary", action="store_true")
    ap.add_argument("--dual-default-thresholds", action="store_true")
    args = ap.parse_args()

    manifest_path = _resolve_manifest_path(args.dataset, args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    model, loaded_kind, agg, k_ckpt, ckpt_meta = load_model_for_eval(args.checkpoint, args.model)
    expected_k = int(args.k_slices) if args.k_slices is not None else k_ckpt

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")

    eval_ds = CTDataset(
        manifest_path=manifest_path,
        split=args.split,
        expected_k_slices=expected_k,
        seed=42,
    )
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ct_batch_collate,
    )

    y_true, p_ab, patient_ids = collect_predictions(model, loader, device, agg=agg)
    roc_auc = safe_roc_auc(y_true, p_ab)

    thr_primary = float(args.threshold)
    thr_secondary = args.secondary_threshold
    if args.dual_default_thresholds and args.model == "densenet_gru":
        thr_primary = 0.613
        thr_secondary = 0.488

    primary_thr_resolved = resolve_abnormal_threshold(cli=thr_primary, default=0.5, ckpt=ckpt_meta)

    thresholds_payload: Dict[str, Any] = {}
    m0 = metrics_at_threshold(y_true, p_ab, primary_thr_resolved)
    thresholds_payload[f"{primary_thr_resolved:.6f}"] = m0

    if thr_secondary is not None:
        ts = float(thr_secondary)
        m1 = metrics_at_threshold(y_true, p_ab, ts)
        thresholds_payload[f"{ts:.6f}"] = m1

    counts = split_counts(manifest_path, expected_k, seed=42)
    status = "PRELIMINARY" if args.preliminary else "FINAL"

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = ROOT / "results" / args.model.replace("+", "_")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model": args.model,
        "loaded_kind": loaded_kind,
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(manifest_path.resolve()),
        "dataset": args.dataset,
        "split": args.split,
        "status": status,
        "roc_auc": roc_auc,
        "train_samples": counts["train"],
        "val_samples": counts["val"],
        "test_samples": counts["test"],
        "evaluated_samples": int(len(y_true)),
        "expected_k_slices": expected_k,
        "agg": agg,
        "thresholds": thresholds_payload,
        "primary_threshold_resolved": primary_thr_resolved,
    }

    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _save_roc_png(out_dir / "roc_curve.png", y_true, p_ab)
    _save_cm_png(
        out_dir / "confusion_matrix.png",
        np.array(m0["confusion_matrix"]),
        f"Confusion @ t={primary_thr_resolved:.4f}",
    )

    print(
        f"\nAccuracy={m0['accuracy']:.4f} ROC-AUC={roc_auc:.4f} "
        f"Recall={m0['recall']:.4f} Spec={m0['specificity']:.4f} "
        f"F1(bin)={m0['f1_binary']:.4f} F1(macro)={m0['f1_macro']:.4f} @ t={primary_thr_resolved:.4f}"
    )
    print(f"Confusion [tn fp; fn tp]:\n{np.array(m0['confusion_matrix'])}")
    if thr_secondary is not None:
        key2 = f"{float(thr_secondary):.6f}"
        if key2 in thresholds_payload:
            m1 = thresholds_payload[key2]
            print(
                f"\n@ t={float(thr_secondary):.4f}: Recall={m1['recall']:.4f} "
                f"missed_abnormals={m1['missed_abnormals']}"
            )


if __name__ == "__main__":
    main()
