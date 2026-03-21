"""Train 3D breast MRI classifier; patient-level val split; save best AUC checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.breast.mri.config_mri import (
    BREAST_MRI_CKPT,
    BREAST_MRI_MANIFEST_PATH,
    BREAST_MRI_TRAIN_METRICS_PATH,
    MRI_DEFAULT_BATCH_SIZE,
    MRI_DEFAULT_EPOCHS,
    MRI_DEFAULT_LR,
    MRI_DEFAULT_SEED,
    MRI_DEFAULT_VAL_FRAC,
    MRI_DEFAULT_WEIGHT_DECAY,
    MRI_NUM_CHANNELS,
    ensure_mri_dirs,
)
from ml.breast.mri.dataset_mri import BreastMRIDataset
from ml.breast.mri.model_mri import build_mri_model


def run_epoch(model, loader, device, criterion, optim=None):
    train = optim is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    all_y: list[float] = []
    all_p: list[float] = []
    for x, y, _, _, _ in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        if train:
            optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optim.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        probs = torch.sigmoid(logits.detach())
        all_y.extend(y.view(-1).cpu().numpy().tolist())
        all_p.extend(probs.view(-1).cpu().numpy().tolist())
    auc = 0.0
    if len(set(all_y)) > 1:
        auc = float(roc_auc_score(all_y, all_p))
    return total_loss / max(n, 1), auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=BREAST_MRI_MANIFEST_PATH)
    ap.add_argument("--epochs", type=int, default=MRI_DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=MRI_DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=MRI_DEFAULT_LR)
    ap.add_argument("--weight-decay", type=float, default=MRI_DEFAULT_WEIGHT_DECAY)
    ap.add_argument("--val-frac", type=float, default=MRI_DEFAULT_VAL_FRAC)
    ap.add_argument("--seed", type=int, default=MRI_DEFAULT_SEED)
    ap.add_argument("--in-channels", type=int, default=MRI_NUM_CHANNELS)
    args = ap.parse_args()

    ensure_mri_dirs()
    BREAST_MRI_CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BreastMRIDataset(
        args.manifest, split="train", val_frac=args.val_frac, seed=args.seed, augment=True
    )
    val_ds = BreastMRIDataset(
        args.manifest, split="val", val_frac=args.val_frac, seed=args.seed, augment=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = build_mri_model(in_channels=args.in_channels).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    crit = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    history: list = []
    ckpt_path = BREAST_MRI_CKPT / "mri_best.pt"

    for ep in range(args.epochs):
        tr_loss, _ = run_epoch(model, train_loader, device, crit, opt)
        va_loss, va_auc = run_epoch(model, val_loader, device, crit, None)
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_auc": va_auc,
            }
        )
        print(
            f"Epoch {ep + 1} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_auc={va_auc:.4f}"
        )
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "val_auc": va_auc,
                    "in_channels": args.in_channels,
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path}")

    BREAST_MRI_TRAIN_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BREAST_MRI_TRAIN_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"best_val_auc": best_auc, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
