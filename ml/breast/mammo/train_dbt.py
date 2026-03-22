"""Train DBT classifier; save best checkpoint by validation AUC."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ml.breast.mammo.config_dbt import (
    DBT_BEST_CKPT_PATH,
    DBT_DEFAULT_BATCH_SIZE,
    DBT_DEFAULT_EPOCHS,
    DBT_DEFAULT_LR,
    DBT_DEFAULT_WEIGHT_DECAY,
    DBT_TRAIN_ARTIFACT_ROOT,
    DBT_TRAIN_METRICS_PATH,
)
from ml.breast.mammo.dataset_dbt import DBTViewDataset
from ml.breast.mammo.model_dbt import build_dbt_model


def _run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    all_y: list[int] = []
    all_p1: list[float] = []
    correct = 0

    for x, y, _, _, _ in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        preds = torch.argmax(logits.detach(), dim=1)
        correct += int((preds == y).sum().item())
        all_y.extend(y.cpu().tolist())
        all_p1.extend(probs.cpu().tolist())

    avg_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)
    auc = 0.0
    if len(set(all_y)) > 1:
        auc = float(roc_auc_score(all_y, all_p1))
    return avg_loss, acc, auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DBT_DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DBT_DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DBT_DEFAULT_LR)
    ap.add_argument("--weight-decay", type=float, default=DBT_DEFAULT_WEIGHT_DECAY)
    args = ap.parse_args()

    DBT_TRAIN_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt_path = DBT_BEST_CKPT_PATH
    metrics_path = DBT_TRAIN_METRICS_PATH

    train_ds = DBTViewDataset(split="train")
    val_ds = DBTViewDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dbt_model(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_auc = -1.0
    history: list[dict] = []

    for ep in range(args.epochs):
        tr_loss, tr_acc, tr_auc = _run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        va_loss, va_acc, va_auc = _run_epoch(model, val_loader, device, criterion, None)
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_auc": tr_auc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_auc": va_auc,
            }
        )
        print(
            f"Epoch {ep + 1}/{args.epochs} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} tr_auc={tr_auc:.4f} | "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} va_auc={va_auc:.4f}"
        )
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": va_auc}, ckpt_path)
            print(f"  saved {ckpt_path}")

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"best_val_auc": best_auc, "history": history}, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
