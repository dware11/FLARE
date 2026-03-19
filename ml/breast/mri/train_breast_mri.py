"""
Training script for the 3D breast MRI (DCE-MRI) classifier.

Usage:
    python ml/breast/mri/train_breast_mri.py --epochs 30 --batch-size 4 --lr 1e-4

Saves best checkpoint (by val AUC) to BREAST_MRI_CKPT.
"""
import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.config import BREAST_MRI_OUTPUTS, BREAST_MRI_CKPT
from ml.breast.mri.dataset_breast_mri import BreastMRIDataset
from ml.breast.mri.model_breast_mri import build_breast_mri_model


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets >= 0
        logits = logits[mask]
        targets = targets[mask].float()
        if logits.numel() == 0:
            return logits.sum() * 0.0
        cancer_logits = logits[:, 1] - logits[:, 0]
        bce = self.bce(cancer_logits, targets)
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


def compute_auc(scores: list, labels: list) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.0


def main(epochs: int = 30, batch_size: int = 4, lr: float = 1e-4) -> None:
    BREAST_MRI_OUTPUTS.mkdir(parents=True, exist_ok=True)

    train_ds = BreastMRIDataset(split="train", augment=True)
    val_ds   = BreastMRIDataset(split="val",   augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_breast_mri_model(in_channels=3, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training breast MRI classifier | device={device} | train={len(train_ds)} val={len(val_ds)}")

    best_auc = 0.0
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                probs = torch.softmax(model(x), dim=1)[:, 1].cpu().tolist()
                val_scores.extend(probs)
                val_labels.extend(y.tolist())

        val_auc = compute_auc(
            [s for s, l in zip(val_scores, val_labels) if l >= 0],
            [l for l in val_labels if l >= 0],
        )
        scheduler.step()
        print(f"Epoch {ep+1}/{epochs} train_loss={train_loss/max(len(train_loader),1):.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": val_auc},
                       BREAST_MRI_CKPT)
            print(f"  -> saved {BREAST_MRI_CKPT} (auc={val_auc:.4f})")

    print(f"Done. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--lr",         type=float, default=1e-4)
    args = ap.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
