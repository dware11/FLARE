"""Train DBT classifier; save best checkpoint by validation AUC."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

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
    all_pred: list[int] = []

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
        all_y.extend(y.cpu().tolist())
        all_p1.extend(probs.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(n, 1)
    acc = sum(1 for a, b in zip(all_y, all_pred) if a == b) / max(n, 1)

    auc = 0.0
    if len(set(all_y)) > 1:
        auc = float(roc_auc_score(all_y, all_p1))

    sens = float(recall_score(all_y, all_pred, zero_division=0))
    prec = float(precision_score(all_y, all_pred, zero_division=0))
    f1 = float(f1_score(all_y, all_pred, zero_division=0))
    cm = confusion_matrix(all_y, all_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / max(tn + fp, 1)

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "auc": auc,
        "sens": sens,
        "spec": spec,
        "prec": prec,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return metrics


def _build_weighted_sampler(dataset: DBTViewDataset) -> WeightedRandomSampler:
    labels = dataset.get_labels()
    counts = Counter(labels)
    class_weights = {c: 1.0 / cnt for c, cnt in counts.items()}
    sample_weights = [class_weights[y] for y in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


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

    train_labels = train_ds.get_labels()
    counts = Counter(train_labels)
    n_neg, n_pos = counts.get(0, 1), counts.get(1, 1)
    print(f"Train split: {len(train_ds)} samples | pos={n_pos} neg={n_neg}")

    sampler = _build_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dbt_model(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    class_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    best_auc = -1.0
    history: list[dict] = []

    for ep in range(args.epochs):
        tr = _run_epoch(model, train_loader, device, criterion, optimizer)
        va = _run_epoch(model, val_loader, device, criterion, None)
        row = {"epoch": ep + 1}
        for k, v in tr.items():
            row[f"train_{k}"] = v
        for k, v in va.items():
            row[f"val_{k}"] = v
        history.append(row)

        print(
            f"Epoch {ep + 1}/{args.epochs} | "
            f"tr_loss={tr['loss']:.4f} tr_auc={tr['auc']:.4f} | "
            f"va_loss={va['loss']:.4f} va_acc={va['acc']:.4f} va_auc={va['auc']:.4f} "
            f"sens={va['sens']:.4f} spec={va['spec']:.4f} f1={va['f1']:.4f}"
        )
        print(
            f"  CM [0=neg,1=pos]: TP={va['tp']} FP={va['fp']} TN={va['tn']} FN={va['fn']}"
        )
        if va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": va["auc"]}, ckpt_path)
            print(f"  saved {ckpt_path}")

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"best_val_auc": best_auc, "history": history}, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
