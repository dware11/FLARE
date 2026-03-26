import argparse
import math
from pathlib import Path
from typing import Optional
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS
from ml.brain.ct.dataset_ct import CTDataset
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
    ce = F.cross_entropy(logits, target, reduction="none", weight=class_weights, ignore_index=ignore_index)
    mask = target != ignore_index
    if not mask.any():
        return logits.sum() * 0.0
    pt = torch.exp(-ce.clamp(max=50.0))
    focal = (1.0 - pt).pow(gamma) * ce
    return focal[mask].mean()


def main(
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    k_slices: int = 15,
    agg: str = "max",
    model_kind: str = "sequence",
    rnn_type: str = "gru",
    grad_clip: float = 1.0,
    sequence_pool: str = "max",
    sequence_topk: int = 3,
    rnn_dropout: float = 0.25,
    use_thickness: bool = False,
    focal_gamma: float = 0.0,
    auc_early_stop_patience: int = 5,
    auc_min_delta: float = 1e-5,
    manifest_path: Optional[Path] = None,
) -> None:
    """
    Train CT classifier on cached k-slice NPZ; saves best checkpoint to OUTPUTS/ct_baseline_best.pt.

    When validation has both classes, the **best checkpoint** is chosen by **highest val ROC-AUC**
    (medical ranking metric); early stopping also watches AUC plateaus. Val loss is logged for
    monitoring only in that mode. If val AUC is undefined (single class in val), falls back to
    val loss for save/stop.

    After backward, **gradient clipping** uses torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    (default 1.0) to stabilize CNN+RNN optimization.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    train_ds = CTDataset(manifest_path=manifest_path, split="train")
    val_ds = CTDataset(manifest_path=manifest_path, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

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

    class_weights = torch.tensor([1.5, 1.0], dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)

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

    def _loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if focal_gamma > 0:
            return focal_cross_entropy(logits, y, focal_gamma, class_weights, ignore_index=-1)
        return ce_criterion(logits, y)

    def _fmt(x: float) -> str:
        return f"{x:.4f}" if math.isfinite(x) else "nan"

    for ep in range(epochs):
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
            loss = _loss(logits, y)
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
                X, y, _, thick = batch
                thick = thick.to(device)
                X, y = X.to(device), y.to(device)
                if not use_thickness:
                    thick = None
                logits = patient_logits_from_model(model, X, agg=agg, thickness=thick)
                batch_loss = _loss(logits, y)
                bl = batch_loss.item()
                if math.isfinite(bl):
                    val_loss += bl
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                for i in range(y.size(0)):
                    if y[i].item() not in (0, 1):
                        continue
                    total += 1
                    all_y.append(y[i].item())
                    all_preds.append(preds[i].item())
                    all_probs_pos.append(probs[i, 1].item())
                    if preds[i].item() == y[i].item():
                        correct += 1
                    if preds[i].item() == 1 and y[i].item() == 1:
                        tp += 1
                    elif preds[i].item() == 0 and y[i].item() == 0:
                        tn += 1
                    elif preds[i].item() == 1 and y[i].item() == 0:
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

        if total > 0:
            cm = confusion_matrix(all_y, all_preds, labels=[0, 1])
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
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5, help="Default tuned for sequence+RNN stability")
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
        default="max",
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
        default=0.0,
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
    )
