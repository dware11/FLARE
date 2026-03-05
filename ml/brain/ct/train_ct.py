import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS
from ml.brain.ct.dataset_ct import CTDataset
from ml.brain.ct.model_ct import build_ct_model


def main(epochs: int = 10, batch_size: int = 4, lr: float = 1e-4) -> None:
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

    # Optimizer with weight decay
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # LR scheduler on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    best_val_loss = float("inf")

    for ep in range(epochs):
        # --------------------
        # Train epoch
        # --------------------
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
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

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                batch_loss = criterion(logits, y)
                val_loss += batch_loss.item()

                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                total += y.size(0)
                correct += (preds == y).sum().item()
                tp += ((preds == 1) & (y == 1)).sum().item()
                tn += ((preds == 0) & (y == 0)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()

        train_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        val_avg = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        acc = correct / total if total > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

        print(
            f"Epoch {ep+1} / {epochs} "
            f"train_loss={train_avg:.4f} val_loss={val_avg:.4f} "
            f"acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} prec={prec:.4f} f1={f1:.4f}"
        )

        scheduler.step(val_avg)

        if len(val_loader) > 0 and val_avg < best_val_loss:
            best_val_loss = val_avg
            ckpt = OUTPUTS / "ct_baseline_best.pt"
            torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
            print(f"  -> saved {ckpt}")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
