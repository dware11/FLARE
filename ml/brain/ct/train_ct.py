import argparse
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
from ml.brain.ct.model_ct import build_ct_model


def main(
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    k_slices: int = 5,
    agg: str = "mean",
) -> None:
    """Train ResNet18 on CT cache; saves best checkpoint to OUTPUTS/ct_baseline_best.pt."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    train_ds = CTDataset(split="train")
    val_ds = CTDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_ct_model()
    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    print(
        "Training: train/val sizes {}, {} | device {} | epochs {} | batch_size {} | lr {}".format(
            len(train_ds), len(val_ds), device, epochs, batch_size, lr
        )
    )
    print("Starting training...")

    # Optimizer: backbone (features) at 0.1*lr, classifier at lr 
    backbone_params = list(model.features.parameters())
    classifier_params = list (model.classifier.parameters()) 

    opt = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": classifier_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    # Emphasize class 0 (normal) to reduce FPs and raise specificity
    class_weights = torch.tensor([1.5, 1.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)

    # LR scheduler on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")

    patience = 5 
    epochs_without_improvement = 0 

    for ep in range(epochs):
        # --------------------
        # Train epoch
        # --------------------
        model.train()
        train_loss = 0.0
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            B, k, C, H, W = X.shape
            X_flat = X.view(B * k, C, H, W)
            opt.zero_grad()
            logits_flat = model(X_flat)
            logits_per_slice = logits_flat.view(B, k, -1)
            if agg == "mean":
                patient_logits = logits_per_slice.mean(dim=1)
            else:
                patient_logits = logits_per_slice.max(dim=1).values
            loss = criterion(patient_logits, y)
            loss.backward()
            opt.step() 
            train_loss += loss.item()
        # --------------------
        # Validation + metrics
        # --------------------
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
                B, k, C, H, W = X.shape
                X_flat = X.view(B * k, C, H, W)
                logits_flat = model(X_flat)
                logits_per_slice = logits_flat.view(B, k, -1)
                if agg == "mean":
                    patient_logits = logits_per_slice.mean(dim=1)
                else:
                    patient_logits = logits_per_slice.max(dim=1).values
                batch_loss = criterion(patient_logits, y)
                val_loss += batch_loss.item()
                probs = torch.softmax(patient_logits, dim=1)
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
            
        train_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        val_avg = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

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
            f"train_loss={train_avg:.4f} val_loss={val_avg:.4f} "
            f"acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} prec={prec:.4f} f1={f1:.4f} roc_auc={roc_auc:.4f}"
        )

        scheduler.step(val_avg)

        if len(val_loader) > 0: 
            if val_avg < best_val_loss: 
                best_val_loss = val_avg 
                epochs_without_improvement = 0 
                
                ckpt = OUTPUTS / "ct_baseline_best.pt" 
                torch.save({"model": model.state_dict(), "epoch": ep}, ckpt) 
                print(f" -> saved {ckpt}") 
            else: 
                epochs_without_improvement += 1 

                if epochs_without_improvement >= patience: 
                    print(f"Early stopping: no val loss improvement for {patience} epochs")
                    break

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    ap.add_argument("--k-slices", type=int, default=5)
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "max"])
    args = ap.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        k_slices=args.k_slices,
        agg=args.agg,
    )
