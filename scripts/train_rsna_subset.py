"""
Train CT sequence model on pre-split RSNA subset manifests (skip internal 70/15/15).

Use --checkpoint path/to/last.pt (or best.pt from this trainer) to resume; each epoch
writes exp_dir/last.pt with optimizer and scheduler state.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.brain.ct.dataset_ct import CTDataset  # noqa: E402
from ml.brain.ct.model_ct import (  # noqa: E402
    build_ct_sequence_model,
    is_sequence_ct_model,
    patient_logits_from_model,
)
from ml.brain.ct.threshold_util import resolve_abnormal_threshold  # noqa: E402
from ml.brain.ct.train_ct import (  # noqa: E402
    focal_cross_entropy,
    _compute_class_weights_inverse_freq,
    _count_normal_abnormal,
    _imbalance_ratio,
    _study_level_sampler_weights,
)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_probs_labels_ids(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    agg: str,
    use_thickness: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    ys: List[int] = []
    ps: List[float] = []
    ids: List[str] = []
    with torch.no_grad():
        for batch in loader:
            X, y, pid, thick = batch
            thick = thick.to(device)
            X = X.to(device)
            if not use_thickness:
                thick = None
            logits = patient_logits_from_model(model, X, agg=agg, thickness=thick)
            probs = torch.softmax(logits, dim=1)
            for i in range(y.size(0)):
                yi = int(y[i].item())
                if yi not in (0, 1):
                    continue
                ys.append(yi)
                ps.append(float(probs[i, 1].item()))
                p = pid[i]
                ids.append(str(p) if not isinstance(p, str) else p)
    return np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.float64), ids


def _preds_at_threshold(probs: np.ndarray, thr: float) -> np.ndarray:
    return (probs >= thr).astype(np.int64)


def _f1_binary(tp: float, fp: float, fn: float) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec <= 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = float(cm[0, 0]), float(cm[0, 1]), float(cm[1, 0]), float(cm[1, 1])
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = _f1_binary(tp, fp, fn)
    rec = sens
    return {"accuracy": acc, "precision": prec, "recall": rec, "sensitivity": sens, "specificity": spec, "f1": f1}


def _roc_auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _tune_threshold_f1(y: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 1001):
        pred = _preds_at_threshold(probs, float(t))
        cm = confusion_matrix(y, pred, labels=[0, 1])
        m = _metrics_from_cm(cm)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t, best_f1


def _write_predictions_csv(path: Path, patient_ids: List[str], y_true: np.ndarray, p_ab: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "y_true", "p_abnormal", "y_pred"])
        for i in range(len(patient_ids)):
            w.writerow([patient_ids[i], int(y_true[i]), f"{p_ab[i]:.8f}", int(y_pred[i])])


def _write_cm_csv(path: Path, cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "Pred_0_Normal", "Pred_1_Abnormal"])
        w.writerow(["True_0_Normal", int(cm[0, 0]), int(cm[0, 1])])
        w.writerow(["True_1_Abnormal", int(cm[1, 0]), int(cm[1, 1])])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CT model on RSNA subset manifests.")
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--val-manifest", type=Path, required=True)
    ap.add_argument("--test-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--experiment-name", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-slices", type=int, default=21)
    ap.add_argument("--threshold", type=float, default=None, help="P(abnormal) cutoff; else env/ckpt/default")
    ap.add_argument("--focal-gamma", type=float, default=1.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--auc-patience", type=int, default=5)
    ap.add_argument("--auc-min-delta", type=float, default=1e-5)
    ap.add_argument(
        "--imbalance-mode",
        type=str,
        default="none",
        choices=["none", "class_weights", "weighted_sampler", "both"],
    )
    ap.add_argument(
        "--class-weight-normal",
        type=float,
        default=None,
        dest="class_weight_normal",
    )
    ap.add_argument(
        "--class-weight-abnormal",
        type=float,
        default=None,
        dest="class_weight_abnormal",
    )
    ap.add_argument(
        "--tune-f1-on-val",
        action="store_true",
        help="Search thresholds on val probs (dense grid) for max F1; use for reported val/test metrics.",
    )
    ap.add_argument(
        "--use-thickness",
        action="store_true",
        help="Fuse slice thickness from NPZ (must be present in cache).",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Load this .pt before training: full resume if it contains optimizer, scheduler, "
            "and last_epoch (e.g. last.pt or a newer best.pt); otherwise weights-only warm start."
        ),
    )
    args = ap.parse_args()

    imbalance_mode = args.imbalance_mode
    if imbalance_mode not in ("none", "class_weights", "weighted_sampler", "both"):
        raise ValueError("invalid imbalance-mode")
    if (args.class_weight_normal is None) != (args.class_weight_abnormal is None):
        raise ValueError("Provide both class weights or neither.")
    if args.class_weight_normal is not None and imbalance_mode in ("none", "weighted_sampler"):
        raise ValueError("Manual class weights require class_weights or both imbalance-mode.")

    _set_seeds(args.seed)

    exp_dir = (args.output_dir / args.experiment_name).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_config["exp_dir"] = str(exp_dir)
    with open(exp_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    eval_threshold_train = resolve_abnormal_threshold(cli=args.threshold, default=0.5, ckpt=None)

    k_slices = int(args.k_slices)
    train_ds = CTDataset(
        manifest_path=args.train_manifest,
        split="train",
        skip_manifest_split=True,
        expected_k_slices=k_slices,
        seed=args.seed,
    )
    val_ds = CTDataset(
        manifest_path=args.val_manifest,
        split="val",
        skip_manifest_split=True,
        expected_k_slices=k_slices,
        seed=args.seed,
    )
    test_ds = CTDataset(
        manifest_path=args.test_manifest,
        split="test",
        skip_manifest_split=True,
        expected_k_slices=k_slices,
        seed=args.seed,
    )

    tr0, tr1 = _count_normal_abnormal(train_ds)
    va0, va1 = _count_normal_abnormal(val_ds)
    te0, te1 = _count_normal_abnormal(test_ds)
    train_total = tr0 + tr1
    ir_train = _imbalance_ratio(tr0, tr1)

    use_loss_weights = imbalance_mode in ("class_weights", "both")
    sampler_used = imbalance_mode in ("weighted_sampler", "both")
    if use_loss_weights:
        if args.class_weight_normal is not None:
            cw_list = [float(args.class_weight_normal), float(args.class_weight_abnormal)]
        else:
            cw_list = _compute_class_weights_inverse_freq(tr0, tr1, train_total)
    else:
        cw_list = [1.0, 1.0]

    if sampler_used:
        sampler_weights = _study_level_sampler_weights(train_ds)
        sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=0
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_kind = "sequence"
    agg = "max"
    rnn_type = "gru"
    sequence_pool = "topk_mean"
    sequence_topk = 3
    rnn_dropout = 0.25
    use_thickness = bool(args.use_thickness)

    model = build_ct_sequence_model(
        rnn_type=rnn_type,
        sequence_pool=sequence_pool,
        topk=sequence_topk,
        rnn_dropout=rnn_dropout,
        use_thickness=use_thickness,
    )
    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    head_params = list(model.slice_head.parameters())
    fusion_params = list(model.thickness_fusion.parameters()) if model.thickness_fusion is not None else []
    opt = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()), "lr": args.lr * 0.1},
            {"params": list(model.rnn.parameters()) + head_params + fusion_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    best_val_loss = float("inf")
    best_val_auc = -1.0
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    start_epoch = 0

    if args.checkpoint is not None:
        ckpt_in = args.checkpoint.expanduser().resolve()
        if not ckpt_in.is_file():
            raise FileNotFoundError(f"--checkpoint not found: {ckpt_in}")
        try:
            ckpt0 = torch.load(ckpt_in, map_location=device, weights_only=False)
        except TypeError:
            ckpt0 = torch.load(ckpt_in, map_location=device)
        if not isinstance(ckpt0, dict) or "model" not in ckpt0:
            raise ValueError(f"Checkpoint must be a dict with a 'model' key: {ckpt_in}")
        model.load_state_dict(ckpt0["model"], strict=True)
        has_resume = (
            ckpt0.get("optimizer_state_dict") is not None
            and ckpt0.get("scheduler_state_dict") is not None
            and ckpt0.get("last_epoch") is not None
        )
        if has_resume:
            opt.load_state_dict(ckpt0["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt0["scheduler_state_dict"])
            start_epoch = int(ckpt0["last_epoch"]) + 1
            bvl = ckpt0.get("best_val_loss")
            best_val_loss = float(bvl) if bvl is not None and math.isfinite(float(bvl)) else float("inf")
            bva = ckpt0.get("best_val_auc")
            best_val_auc = float(bva) if bva is not None and math.isfinite(float(bva)) else -1.0
            epochs_without_improvement = int(ckpt0.get("epochs_without_improvement", 0))
            history = list(ckpt0.get("history", []))
            print(
                f"Resumed training from {ckpt_in} (next epoch {start_epoch + 1} / {args.epochs})",
                flush=True,
            )
        else:
            print(
                f"Loaded weights from {ckpt_in}; optimizer fresh; starting epoch 1.",
                flush=True,
            )

    class_weights = torch.tensor(cw_list, dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    uniform_w = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    val_ce_criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=uniform_w)

    focal_gamma = float(args.focal_gamma)
    grad_clip = float(args.grad_clip)

    def _train_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if focal_gamma > 0:
            return focal_cross_entropy(logits, y, focal_gamma, class_weights, ignore_index=-1)
        return ce_criterion(logits, y)

    def _val_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if focal_gamma > 0:
            return focal_cross_entropy(logits, y, focal_gamma, uniform_w, ignore_index=-1)
        return val_ce_criterion(logits, y)

    def _fmt(x: float) -> str:
        return f"{x:.4f}" if math.isfinite(x) else "nan"

    def _resume_payload(ep_done: int) -> Dict[str, Any]:
        return {
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "last_epoch": ep_done,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }

    for ep in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_finite = 0
        for batch in train_loader:
            X, y, _, thick = batch
            thick = thick.to(device)
            X, y = X.to(device), y.to(device)
            if not use_thickness:
                thick = None
            opt.zero_grad()
            logits = patient_logits_from_model(model, X, agg=agg, thickness=thick)
            loss = _train_loss(logits, y)
            if not torch.isfinite(loss):
                print("  [warn] non-finite loss in train batch; skipping step", flush=True)
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_loss += loss.item()
            train_finite += 1

        if train_finite == 0:
            print(f"Epoch {ep+1}: no finite train batches; stopping.", flush=True)
            break

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        tp = tn = fp = fn = 0
        all_y: List[int] = []
        all_preds: List[int] = []
        all_probs_pos: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                X, y, _, thick = batch
                thick = thick.to(device)
                X, y = X.to(device), y.to(device)
                if not use_thickness:
                    thick = None
                logits = patient_logits_from_model(model, X, agg=agg, thickness=thick)
                batch_loss = _val_loss(logits, y)
                bl = batch_loss.item()
                if math.isfinite(bl):
                    val_loss += bl
                probs = torch.softmax(logits, dim=1)
                for i in range(y.size(0)):
                    if y[i].item() not in (0, 1):
                        continue
                    p1 = probs[i, 1].item()
                    pred_i = 1 if p1 >= eval_threshold_train else 0
                    total += 1
                    all_y.append(int(y[i].item()))
                    all_preds.append(pred_i)
                    all_probs_pos.append(p1)
                    if pred_i == y[i].item():
                        correct += 1
                    if pred_i == 1 and y[i].item() == 1:
                        tp += 1
                    elif pred_i == 0 and y[i].item() == 0:
                        tn += 1
                    elif pred_i == 1 and y[i].item() == 0:
                        fp += 1
                    else:
                        fn += 1

        train_avg = train_loss / train_finite if train_finite > 0 else float("nan")
        val_avg = val_loss / len(val_loader) if len(val_loader) > 0 else float("nan")
        acc = correct / total if total > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_ep = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        use_auc_selection = total > 0 and len(set(all_y)) == 2
        roc_auc = float(roc_auc_score(all_y, all_probs_pos)) if use_auc_selection else float("nan")

        if is_sequence_ct_model(model):
            scheduler.step()

        improved = False
        selection = "none"
        cm_for_ckpt = None
        if total > 0:
            cm_for_ckpt = confusion_matrix(all_y, all_preds, labels=[0, 1])

        if len(val_loader) > 0 and math.isfinite(val_avg):
            if use_auc_selection and math.isfinite(roc_auc):
                if roc_auc > best_val_auc + args.auc_min_delta:
                    best_val_auc = roc_auc
                    improved = True
                    selection = "val_auc"
            else:
                if val_avg < best_val_loss:
                    best_val_loss = val_avg
                    improved = True
                    selection = "val_loss"

            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        history.append(
            {
                "epoch": ep + 1,
                "train_loss": train_avg,
                "val_loss": val_avg,
                "val_accuracy": acc,
                "val_precision": prec,
                "val_recall": sens,
                "val_f1_at_eval_threshold": f1_ep,
                "val_roc_auc": roc_auc,
                "val_sensitivity": sens,
                "val_specificity": spec,
                "best_val_auc_so_far": best_val_auc,
                "selection": selection,
            }
        )

        print(
            f"Epoch {ep+1} / {args.epochs} train_loss={_fmt(train_avg)} val_loss={_fmt(val_avg)} "
            f"val_auc={_fmt(roc_auc)}",
            flush=True,
        )

        ck_common: Dict[str, Any] = {
            "epoch": ep,
            "model_kind": model_kind,
            "agg": agg,
            "rnn_type": rnn_type,
            "sequence_pool": sequence_pool,
            "sequence_topk": sequence_topk,
            "rnn_dropout": rnn_dropout,
            "use_thickness": use_thickness,
            "focal_gamma": focal_gamma,
            "selection_metric": selection,
            "best_val_auc": best_val_auc if use_auc_selection else None,
            "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
            "k_slices": k_slices,
            "class_weights": cw_list,
            "eval_threshold": eval_threshold_train,
            "train_counts": {"normal": tr0, "abnormal": tr1},
            "val_counts": {"normal": va0, "abnormal": va1},
            "test_counts": {"normal": te0, "abnormal": te1},
            "imbalance_ratio_train_abnormal_over_normal": ir_train if math.isfinite(ir_train) else None,
            "imbalance_mode": imbalance_mode,
            "class_weights_used_in_loss": use_loss_weights,
            "weighted_sampler_used": sampler_used,
            "val_confusion_matrix_at_eval_threshold": cm_for_ckpt.tolist()
            if cm_for_ckpt is not None
            else None,
        }

        if len(val_loader) > 0 and math.isfinite(val_avg):
            if improved:
                ckpt_path = exp_dir / "best.pt"
                payload = {
                    "model": model.state_dict(),
                    **ck_common,
                    **_resume_payload(ep),
                }
                torch.save(payload, ckpt_path)
                print(f" -> saved {ckpt_path} (best by {selection})", flush=True)
            elif epochs_without_improvement >= args.auc_patience:
                print(
                    f"Early stopping: no improvement for {args.auc_patience} epochs "
                    f"({'val_auc' if use_auc_selection else 'val_loss'})",
                    flush=True,
                )
                last_path = exp_dir / "last.pt"
                torch.save(
                    {"model": model.state_dict(), **ck_common, **_resume_payload(ep)},
                    last_path,
                )
                break

        last_path = exp_dir / "last.pt"
        torch.save(
            {"model": model.state_dict(), **ck_common, **_resume_payload(ep)},
            last_path,
        )

        if not math.isfinite(train_avg) or not math.isfinite(val_avg):
            print("Non-finite loss; stopping.", flush=True)
            break

    with open(exp_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt_file = exp_dir / "best.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"No checkpoint saved at {ckpt_file}; train/val may be empty.")

    try:
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(ckpt["model"])
    eval_thr = resolve_abnormal_threshold(cli=args.threshold, default=0.5, ckpt=ckpt)

    y_val, p_val, id_val = _collect_probs_labels_ids(model, val_loader, device, agg, use_thickness)
    y_te, p_te, id_te = _collect_probs_labels_ids(model, test_loader, device, agg, use_thickness)

    tuned_thr: Optional[float] = None
    tuned_f1_val: Optional[float] = None
    report_thr = eval_thr
    if args.tune_f1_on_val and len(y_val) > 0:
        tuned_thr, tuned_f1_val = _tune_threshold_f1(y_val, p_val)
        report_thr = tuned_thr

    pred_val = _preds_at_threshold(p_val, report_thr)
    pred_te = _preds_at_threshold(p_te, report_thr)

    _write_predictions_csv(exp_dir / "val_predictions.csv", id_val, y_val, p_val, pred_val)
    _write_predictions_csv(exp_dir / "test_predictions.csv", id_te, y_te, p_te, pred_te)

    cm_val = confusion_matrix(y_val, pred_val, labels=[0, 1])
    cm_te = confusion_matrix(y_te, pred_te, labels=[0, 1])
    _write_cm_csv(exp_dir / "val_confusion_matrix.csv", cm_val)
    _write_cm_csv(exp_dir / "test_confusion_matrix.csv", cm_te)

    m_val = _metrics_from_cm(cm_val)
    m_te = _metrics_from_cm(cm_te)
    auc_val = _roc_auc_safe(y_val, p_val)
    auc_te = _roc_auc_safe(y_te, p_te)

    support_val = {"normal": int((y_val == 0).sum()), "abnormal": int((y_val == 1).sum()), "total": int(len(y_val))}
    support_te = {"normal": int((y_te == 0).sum()), "abnormal": int((y_te == 1).sum()), "total": int(len(y_te))}

    metrics = {
        "resolved_threshold": eval_thr,
        "tuned_threshold": tuned_thr,
        "tuned_f1_on_val": tuned_f1_val,
        "threshold_used_for_reporting": report_thr,
        "best_val_auc_training": float(best_val_auc) if math.isfinite(best_val_auc) and best_val_auc >= 0 else None,
        "val": {**m_val, "roc_auc": auc_val, "support": support_val},
        "test": {**m_te, "roc_auc": auc_te, "support": support_te},
        "train_counts": {"normal": tr0, "abnormal": tr1},
    }
    with open(exp_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "RSNA subset training metrics",
        f"resolved_threshold (CLI/env/ckpt/default): {eval_thr}",
        f"tuned_threshold (val max F1): {tuned_thr}",
        f"tuned_f1_on_val: {tuned_f1_val}",
        f"threshold_used_for_reporting: {report_thr}",
        f"best_val_auc (training selection): {metrics['best_val_auc_training']}",
        "",
        "Validation:",
        f"  accuracy={m_val['accuracy']:.4f} precision={m_val['precision']:.4f} recall={m_val['recall']:.4f} "
        f"f1={m_val['f1']:.4f} roc_auc={auc_val} sens={m_val['sensitivity']:.4f} spec={m_val['specificity']:.4f}",
        f"  support: {support_val}",
        "",
        "Test:",
        f"  accuracy={m_te['accuracy']:.4f} precision={m_te['precision']:.4f} recall={m_te['recall']:.4f} "
        f"f1={m_te['f1']:.4f} roc_auc={auc_te} sens={m_te['sensitivity']:.4f} spec={m_te['specificity']:.4f}",
        f"  support: {support_te}",
    ]
    (exp_dir / "metrics_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Artifacts in {exp_dir}", flush=True)


if __name__ == "__main__":
    main()