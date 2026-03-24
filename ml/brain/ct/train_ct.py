import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
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


def main(
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    k_slices: int = 15,
    agg: str = "max",
    model_kind: str = "sequence",
    rnn_type: str = "lstm",
    grad_clip: float = 1.0,
) -> None:
    """
    Train CT classifier on cached k-slice NPZ; saves best checkpoint to OUTPUTS/ct_baseline_best.pt.
    Default: DenseNet slice encoder + bidirectional LSTM (sequence). Use --model legacy for per-slice logits + pool.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    train_ds = CTDataset(split="train")
    val_ds = CTDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if model_kind == "sequence":
        model = build_ct_sequence_model(rnn_type=rnn_type)
    else:
        model = build_ct_model()

    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    if is_sequence_ct_model(model):
        opt = torch.optim.AdamW(
            [
                {"params": list(model.backbone.parameters()), "lr": lr * 0.1},
                {"params": list(model.rnn.parameters()) + list(model.head.parameters()), "lr": lr},
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
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)

    print(
        "Training: train/val sizes {}, {} | device {} | model={} | epochs {} | batch_size {} | lr {} | grad_clip {}".format(
            len(train_ds), len(val_ds), device, model_kind, epochs, batch_size, lr, grad_clip
        )
    )
    if not is_sequence_ct_model(model):
        print(f"  legacy aggregation over slices: {agg}")
    print("Starting training...")

    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0

    def _fmt(x: float) -> str:
        return f"{x:.4f}" if math.isfinite(x) else "nan"

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_finite = 0
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = patient_logits_from_model(model, X, agg=agg)
            loss = criterion(logits, y)
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
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                logits = patient_logits_from_model(model, X, agg=agg)
                batch_loss = criterion(logits, y)
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

        if total > 0 and len(set(all_y)) == 2:
            roc_auc = roc_auc_score(all_y, all_probs_pos)
        else:
            roc_auc = 0.0

        if total > 0:
            cm = confusion_matrix(all_y, all_preds, labels=[0, 1])
            print(f"Confusion matrix (true=rows, pred=cols) [0=normal, 1=abnormal]:\n{cm}")

        print(
            f"Epoch {ep+1} / {epochs} "
            f"train_loss={_fmt(train_avg)} val_loss={_fmt(val_avg)} "
            f"acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} prec={prec:.4f} f1={f1:.4f} roc_auc={roc_auc:.4f}"
        )

        if is_sequence_ct_model(model):
            scheduler.step()
        else:
            if math.isfinite(val_avg):
                scheduler.step(val_avg)

        if len(val_loader) > 0 and math.isfinite(val_avg):
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                epochs_without_improvement = 0

                ckpt = OUTPUTS / "ct_baseline_best.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": ep,
                        "model_kind": model_kind,
                        "agg": agg,
                        "rnn_type": rnn_type,
                    },
                    ckpt,
                )
                print(f" -> saved {ckpt}")
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping: no val loss improvement for {patience} epochs")
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
    ap.add_argument("--lr", type=float, default=5e-5, help="Default tuned for sequence+LSTM stability")
    ap.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max grad norm (0 disables). Reduces NaN risk with RNN+CNN.",
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
    ap.add_argument("--rnn-type", type=str, default="lstm", choices=["lstm", "gru"])
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
    )
