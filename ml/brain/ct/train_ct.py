# -----------------------------------------------------------------------------
# Imbalance Ablation Study
# Keep all hyperparameters fixed (CLI defaults match project best config).
# Only change imbalance mode:
#
#   1. baseline:       --imbalance-mode none
#   2. class weights:  --imbalance-mode class_weights
#   3. sampler:        --imbalance-mode weighted_sampler
#   4. both:           --imbalance-mode both
#
# Risks if imbalance handling is too strong: overcorrection -> high FP, specificity
# drop, or less stable training (noisy gradients from aggressive sampling + weights).
# -----------------------------------------------------------------------------
import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple
import sys

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS
from ml.brain.ct.threshold_util import resolve_abnormal_threshold
from ml.brain.ct.dataset_ct import CTDataset, ct_batch_collate, unpack_ct_batch
from ml.brain.ct.model_ct import (
    build_ct_model,
    build_ct_sequence_model,
    is_sequence_ct_model,
    patient_logits_from_model,
)


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    class_weights: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Multi-class focal loss; ignores positions where target == ignore_index."""
    # Class weights applied once inside CE (per-class term); focal modulates (1-pt)^gamma * ce.
    # Sampler (if any) is separate — do not multiply weights again outside this path.
    ce = F.cross_entropy(logits, target, reduction="none", weight=class_weights, ignore_index=ignore_index)
    mask = target != ignore_index
    if not mask.any():
        return logits.sum() * 0.0
    pt = torch.exp(-ce.clamp(max=50.0))
    focal = (1.0 - pt).pow(gamma) * ce
    return focal[mask].mean()


def _count_normal_abnormal(dataset: CTDataset) -> Tuple[int, int]:
    n0 = n1 = 0
    for e in dataset.entries:
        lab = int(e.get("label", -1))
        if lab == 0:
            n0 += 1
        elif lab == 1:
            n1 += 1
    return n0, n1


def _imbalance_ratio(n_normal: int, n_abnormal: int) -> float:
    if n_normal <= 0:
        return float("inf") if n_abnormal > 0 else float("nan")
    return n_abnormal / n_normal


def _compute_class_weights_inverse_freq(
    n_normal: int, n_abnormal: int, total: int
) -> List[float]:
    """weight_c = total / (2 * count_c) for binary classes; train split only."""
    if n_normal <= 0 or n_abnormal <= 0:
        return [1.0, 1.0]
    w0 = total / (2.0 * n_normal)
    w1 = total / (2.0 * n_abnormal)
    return [w0, w1]


def _study_level_sampler_weights(train_ds: CTDataset) -> torch.Tensor:
    """One weight per manifest row (study); inverse class frequency for WeightedRandomSampler."""
    n0, n1 = _count_normal_abnormal(train_ds)
    w0 = 1.0 / n0 if n0 > 0 else 1.0
    w1 = 1.0 / n1 if n1 > 0 else 1.0
    weights: List[float] = []
    for e in train_ds.entries:
        lab = int(e.get("label", -1))
        weights.append(w0 if lab == 0 else w1)
    return torch.tensor(weights, dtype=torch.double)


def main(
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-5,
    weight_decay: float = 1e-4,
    k_slices: int = 15,
    agg: str = "max",
    model_kind: str = "sequence",
    rnn_type: str = "gru",
    grad_clip: float = 1.0,
    sequence_pool: str = "topk_mean",
    sequence_topk: int = 3,
    rnn_dropout: float = 0.25,
    use_thickness: bool = False,
    focal_gamma: float = 1.5,
    auc_early_stop_patience: int = 5,
    auc_min_delta: float = 1e-5,
    manifest_path: Optional[Path] = None,
    threshold_cli: Optional[float] = None,
    imbalance_mode: str = "none",
    class_weight_normal: Optional[float] = None,
    class_weight_abnormal: Optional[float] = None,
) -> None:
    """
    Train CT classifier on cached k-slice NPZ; saves best checkpoint to OUTPUTS/ct_baseline_best.pt.

    When validation has both classes, the **best checkpoint** is chosen by **highest val ROC-AUC**
    (medical ranking metric); early stopping also watches AUC plateaus. Val loss is logged for
    monitoring only in that mode. If val AUC is undefined (single class in val), falls back to
    val loss for save/stop.

    After backward, **gradient clipping** uses torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    (default 1.0) to stabilize CNN+RNN optimization.

    Threshold affects **reported** val confusion / metrics only, not loss or sampling.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    if imbalance_mode not in ("none", "class_weights", "weighted_sampler", "both"):
        raise ValueError(
            f"imbalance_mode must be none|class_weights|weighted_sampler|both, got {imbalance_mode!r}"
        )
    if (class_weight_normal is None) != (class_weight_abnormal is None):
        raise ValueError("Provide both --class-weight-normal and --class-weight-abnormal, or neither.")
    if class_weight_normal is not None and imbalance_mode in ("none", "weighted_sampler"):
        raise ValueError(
            "Manual class weights only apply with --imbalance-mode class_weights or both "
            f"(got {imbalance_mode})."
        )

    # Same rule as infer.py: --threshold > $ABNORMAL_THRESHOLD > 0.5 (evaluation only).
    eval_threshold = resolve_abnormal_threshold(cli=threshold_cli, default=0.5)
    print(
        f"[threshold] Val confusion/sens/spec use P(abnormal)>={eval_threshold:.4f} "
        f"(set --threshold or env ABNORMAL_THRESHOLD to match deployment)"
    )

    train_ds = CTDataset(manifest_path=manifest_path, split="train", expected_k_slices=k_slices)
    val_ds = CTDataset(manifest_path=manifest_path, split="val", expected_k_slices=k_slices)
    test_ds = CTDataset(manifest_path=manifest_path, split="test", expected_k_slices=k_slices)

    tr0, tr1 = _count_normal_abnormal(train_ds)
    va0, va1 = _count_normal_abnormal(val_ds)
    te0, te1 = _count_normal_abnormal(test_ds)
    train_total = tr0 + tr1
    ir_train = _imbalance_ratio(tr0, tr1)

    use_loss_weights = imbalance_mode in ("class_weights", "both")
    sampler_used = imbalance_mode in ("weighted_sampler", "both")

    if use_loss_weights:
        if class_weight_normal is not None:
            cw_list = [float(class_weight_normal), float(class_weight_abnormal)]
        else:
            cw_list = _compute_class_weights_inverse_freq(tr0, tr1, train_total)
    else:
        cw_list = [1.0, 1.0]

    loss_desc = f"focal (gamma={focal_gamma})" if focal_gamma > 0 else "weighted CE"
    if math.isfinite(ir_train):
        _irr_s = f"{ir_train:.4f}"
    elif ir_train == float("inf"):
        _irr_s = "inf"
    else:
        _irr_s = "nan"
    print(
        "[imbalance audit]\n"
        f"train_counts: normal={tr0} abnormal={tr1}\n"
        f"val_counts: normal={va0} abnormal={va1}\n"
        f"test_counts: normal={te0} abnormal={te1}\n"
        f"imbalance_ratio (abnormal/normal, train): {_irr_s}\n"
        f"class_weights (loss): {cw_list}\n"
        f"class_weights_used (in loss): {use_loss_weights}\n"
        f"sampler: {'WeightedRandomSampler (study-level)' if sampler_used else 'none'}\n"
        f"loss: {loss_desc}"
    )

    if sampler_used:
        sampler_weights = _study_level_sampler_weights(train_ds)
        sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            collate_fn=ct_batch_collate,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=ct_batch_collate,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ct_batch_collate
    )

    if model_kind == "sequence":
        model = build_ct_sequence_model(
            rnn_type=rnn_type,
            sequence_pool=sequence_pool,
            topk=sequence_topk,
            rnn_dropout=rnn_dropout,
            use_thickness=use_thickness,
        )
    else:
        model = build_ct_model()

    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    if is_sequence_ct_model(model):
        head_params = list(model.slice_head.parameters())
        fusion_params = list(model.thickness_fusion.parameters()) if model.thickness_fusion is not None else []
        opt = torch.optim.AdamW(
            [
                {"params": list(model.backbone.parameters()), "lr": lr * 0.1},
                {"params": list(model.rnn.parameters()) + head_params + fusion_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    else:
        backbone_params = list(model.features.parameters())
        classifier_params = list(model.classifier.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr * 0.1},
                {"params": classifier_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=2,
        )

    class_weights = torch.tensor(cw_list, dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)

    # Val: unweighted loss only (no class weights in CE/focal on validation).
    uniform_w = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    val_ce_criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=uniform_w)

    print(
        "Run metadata: k_slices={} class_weights[normal,abnormal]={} focal_gamma={} "
        "sequence_pool={} sequence_topk={} eval_threshold={:.4f} imbalance_mode={}".format(
            k_slices,
            cw_list,
            focal_gamma,
            sequence_pool,
            sequence_topk,
            eval_threshold,
            imbalance_mode,
        )
    )
    print(
        "Training: train/val sizes {}, {} | device {} | model={} | epochs {} | batch_size {} | lr {} | "
        "grad_clip {} (clip_grad_norm_)".format(
            len(train_ds), len(val_ds), device, model_kind, epochs, batch_size, lr, grad_clip
        )
    )
    if is_sequence_ct_model(model):
        print(
            f"  sequence_pool={sequence_pool} topk={sequence_topk} rnn_dropout={rnn_dropout} "
            f"use_thickness={use_thickness} focal_gamma={focal_gamma} | "
            f"checkpoint/early-stop: val_auc (fallback val_loss if AUC undefined)"
        )
    if not is_sequence_ct_model(model):
        print(f"  legacy aggregation over slices: {agg}")
    print("Starting training...")

    best_val_loss = float("inf")
    best_val_auc = -1.0
    epochs_without_improvement = 0

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

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_finite = 0
        for batch in train_loader:
            X, y, _, thick = unpack_ct_batch(batch)
            thick = thick.to(device)
            X, y = X.to(device), y.to(device)
            if not use_thickness:
                thick = None
            opt.zero_grad()
            logits = patient_logits_from_model(model, X, agg=agg, thickness=thick)
            loss = _train_loss(logits, y)
            if not torch.isfinite(loss):
                print("  [warn] non-finite loss in train batch; skipping step")
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_loss += loss.item()
            train_finite += 1

        if train_finite == 0:
            print("Epoch {}: no finite train batches; stopping.".format(ep + 1))
            break

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        tp = tn = fp = fn = 0

        all_y = []
        all_preds = []
        all_probs_pos = []

        with torch.no_grad():
            for batch in val_loader:
                X, y, _, thick = unpack_ct_batch(batch)
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
                    pred_i = 1 if p1 >= eval_threshold else 0
                    total += 1
                    all_y.append(y[i].item())
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
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

        use_auc_selection = total > 0 and len(set(all_y)) == 2
        if use_auc_selection:
            roc_auc = float(roc_auc_score(all_y, all_probs_pos))
        else:
            roc_auc = float("nan")

        cm_for_ckpt = None
        if total > 0:
            cm = confusion_matrix(all_y, all_preds, labels=[0, 1])
            cm_for_ckpt = cm
            print(f"Confusion matrix (true=rows, pred=cols) [0=normal, 1=abnormal]:\n{cm}")

        print(
            f"Epoch {ep+1} / {epochs} "
            f"train_loss={_fmt(train_avg)} val_loss={_fmt(val_avg)} "
            f"acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} prec={prec:.4f} f1={f1:.4f} roc_auc={_fmt(roc_auc)}"
        )

        if is_sequence_ct_model(model):
            scheduler.step()
        else:
            if math.isfinite(val_avg):
                scheduler.step(val_avg)

        improved = False
        selection = "none"

        if len(val_loader) > 0 and math.isfinite(val_avg):
            if use_auc_selection and math.isfinite(roc_auc):
                if roc_auc > best_val_auc + auc_min_delta:
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
                ckpt = OUTPUTS / "ct_baseline_best.pt"
                payload = {
                    "model": model.state_dict(),
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
                    "eval_threshold": eval_threshold,
                    "train_counts": {"normal": tr0, "abnormal": tr1},
                    "val_counts": {"normal": va0, "abnormal": va1},
                    "test_counts": {"normal": te0, "abnormal": te1},
                    "imbalance_ratio_train_abnormal_over_normal": ir_train
                    if math.isfinite(ir_train)
                    else None,
                    "imbalance_mode": imbalance_mode,
                    "class_weights_used_in_loss": use_loss_weights,
                    "weighted_sampler_used": sampler_used,
                    "val_confusion_matrix_at_eval_threshold": cm_for_ckpt.tolist()
                    if cm_for_ckpt is not None
                    else None,
                }
                torch.save(payload, ckpt)
                print(f" -> saved {ckpt} (best by {selection})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= auc_early_stop_patience:
                    print(
                        f"Early stopping: no improvement on "
                        f"{'val_auc' if use_auc_selection else 'val_loss'} "
                        f"for {auc_early_stop_patience} epochs"
                    )
                    break
        elif len(val_loader) > 0 and not math.isfinite(val_avg):
            print("  [warn] val_loss is not finite; not saving checkpoint this epoch")

        if not math.isfinite(train_avg) or not math.isfinite(val_avg):
            print("Epoch {}: non-finite loss; stopping training.".format(ep + 1))
            break

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5, help="Default tuned for sequence+RNN stability")
    ap.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="clip_grad_norm_ max norm (0 disables). Stabilizes CNN+RNN.",
    )
    ap.add_argument("--k-slices", type=int, default=15, help="Informational; NPZ k is defined at preprocess time")
    ap.add_argument("--agg", type=str, default="max", choices=["mean", "max"], help="Legacy model only: slice pooling")
    ap.add_argument(
        "--model",
        type=str,
        default="sequence",
        choices=["sequence", "legacy"],
        dest="model_kind",
        help="sequence: DenseNet + BiLSTM/BiGRU; legacy: DenseNet per slice + agg",
    )
    ap.add_argument(
        "--rnn-type",
        type=str,
        default="gru",
        choices=["lstm", "gru"],
        help="Default gru: lighter, often stable (Wang-style pipelines use GRU stages).",
    )
    ap.add_argument(
        "--sequence-pool",
        type=str,
        default="topk_mean",
        choices=["center", "max", "topk_mean"],
        help="How to combine per-slice logits after RNN (max recommended for focal bleeds).",
    )
    ap.add_argument("--sequence-topk", type=int, default=3, help="For topk_mean pooling only")
    ap.add_argument("--rnn-dropout", type=float, default=0.25, dest="rnn_dropout")
    ap.add_argument(
        "--use-thickness",
        action="store_true",
        help="Fuse normalized slice thickness from NPZ (requires preprocess with slice_thickness_mm).",
    )
    ap.add_argument(
        "--focal-gamma",
        type=float,
        default=1.5,
        help="If >0, focal loss (e.g. 2.0 for imbalanced positives). 0 = weighted CE.",
    )
    ap.add_argument(
        "--auc-patience",
        type=int,
        default=5,
        dest="auc_early_stop_patience",
        help="Early stop when val_auc (or val_loss fallback) does not improve for this many epochs.",
    )
    ap.add_argument(
        "--auc-min-delta",
        type=float,
        default=1e-5,
        help="Minimum AUC gain to count as improvement when selecting checkpoints.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override path to ct_processed_manifest.json (default: META from src.config)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="P(abnormal) cutoff for val confusion metrics; else env ABNORMAL_THRESHOLD else 0.5",
    )
    ap.add_argument(
        "--imbalance-mode",
        type=str,
        default="none",
        choices=["none", "class_weights", "weighted_sampler", "both"],
        help="Imbalance ablation: none | class_weights (loss only) | weighted_sampler (train only) | both",
    )
    ap.add_argument(
        "--class-weight-normal",
        type=float,
        default=None,
        dest="class_weight_normal",
        help="Override loss weight for class 0 (must use with --class-weight-abnormal; class_weights/both only)",
    )
    ap.add_argument(
        "--class-weight-abnormal",
        type=float,
        default=None,
        dest="class_weight_abnormal",
        help="Override loss weight for class 1 (must use with --class-weight-normal; class_weights/both only)",
    )
    args = ap.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        k_slices=args.k_slices,
        agg=args.agg,
        model_kind=args.model_kind,
        rnn_type=args.rnn_type,
        grad_clip=args.grad_clip,
        sequence_pool=args.sequence_pool,
        sequence_topk=args.sequence_topk,
        rnn_dropout=args.rnn_dropout,
        use_thickness=args.use_thickness,
        focal_gamma=args.focal_gamma,
        auc_early_stop_patience=args.auc_early_stop_patience,
        auc_min_delta=args.auc_min_delta,
        manifest_path=args.manifest,
        threshold_cli=args.threshold,
        imbalance_mode=args.imbalance_mode,
        class_weight_normal=args.class_weight_normal,
        class_weight_abnormal=args.class_weight_abnormal,
    )
