import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS
from ml.brain.ct.dataset_ct import CTDataset
from ml.brain.ct.model_ct import build_ct_model


def main(epochs: int = 10, batch_size: int = 4, lr: float = 1e-4):
    """Train ResNet18 on CT cache; saves best checkpoint to OUTPUTS/ct_baseline_best.pt."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    train_ds = CTDataset(split="train")
    val_ds = CTDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_ct_model()
    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device

    print("Training: train/val sizes {}, {} | device {} | epochs {} | batch_size {} | lr {}".format(
        len(train_ds), len(val_ds), device, epochs, batch_size, lr))
    print("Starting training...")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf") 

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() 

        model.eval() 
        val_loss = 0.0 

        with torch.no_grad(): 
            for x, y in val_loader: 
                x, y = x.to(device), y.to(device) 
                logits = model(x) 
                val_loss +=  criterion(logits, y).item() 
        train_avg = train_loss / len(train_loader) if train_loader else 0.0
        val_avg = val_loss / len(val_loader) if val_loader else 0.0
        print(f"Epoch {ep+1} / {epochs} train_loss={train_avg:.4f} val_loss={val_avg:.4f}")


        if val_loader and val_loss < best_val_loss: 
            best_val_loss = val_loss 
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
