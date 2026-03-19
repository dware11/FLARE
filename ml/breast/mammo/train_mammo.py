"""
Training script for the 3D DBT mammography classifier.

Usage:
    python ml/breast/mammo/train_mammo.py --epochs 30 --batch-size 4 --lr 1e-4

Saves best checkpoint (by val AUC) to BREAST_MAMMO_CKPT.
Requires: torch, scikit-learn (for AUC), BCS-DBT manifest at BREAST_MAMMO_MANIFEST.
"""
import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.config import BREAST_MAMMO_OUTPUTS, BREAST_MAMMO_CKPT
from ml.breast.mammo.dataset_mammo import MammoDataset
from ml.breast.mammo.model_mammo import build_mammo_model


class FocalLoss(nn.Module):
    """Binary focal loss — down-weights easy negatives, focuses on hard positives.

    Handles the ~2% cancer prevalence in BCS-DBT without manual class weighting.
    gamma=2 follows the original RetinaNet paper.
    """

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
        # Use cancer class logit for binary focal loss
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


def main(
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
) -> None:
    BREAST_MAMMO_OUTPUTS.mkdir(parents=True, exist_ok=True)

    train_ds = MammoDataset(split="train", augment=True)
    val_ds   = MammoDataset(split="val",   augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_mammo_model(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training DBT classifier | device={device} | train={len(train_ds)} val={len(val_ds)}")

    best_auc = 0.0
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                val_scores.extend(probs)
                val_labels.extend(y.tolist())

        val_auc = compute_auc(
            [s for s, l in zip(val_scores, val_labels) if l >= 0],
            [l for l in val_labels if l >= 0],
        )
        scheduler.step()

        print(
            f"Epoch {ep+1}/{epochs} "
            f"train_loss={train_loss/max(len(train_loader),1):.4f} "
            f"val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": val_auc},
                       BREAST_MAMMO_CKPT)
            print(f"  -> saved {BREAST_MAMMO_CKPT} (auc={val_auc:.4f})")

    print(f"Done. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--lr",         type=float, default=1e-4)
    args = ap.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
