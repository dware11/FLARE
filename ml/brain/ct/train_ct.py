import argparse
import json
import sys
import time
from pathlib import Path

# #region agent log — early env/path diagnostics (stderr -> Slurm .err)
def _stderr(msg: str, data=None):
    print(f"FLARE_DEBUG: {msg}", file=sys.stderr)
    if data is not None:
        print(f"FLARE_DEBUG: {data}", file=sys.stderr)
import os
_stderr("DATA_ROOT", os.environ.get("DATA_ROOT"))
# #endregion

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS
from src.config import CT_MANIFEST
# #region agent log
_stderr("CONFIG", f"OUTPUTS={OUTPUTS} exists={OUTPUTS.exists()} CT_MANIFEST={CT_MANIFEST} exists={CT_MANIFEST.exists()}")
# #endregion
from ml.brain.ct.dataset_ct import CTDataset
from ml.brain.ct.model_ct import build_ct_model

DEBUG_LOG = ROOT / "debug_pipeline.log"

def _agent_log(msg, data=None, hyp=""):
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"message": msg, "data": data or {}, "hypothesisId": hyp, "location": "train_ct", "timestamp": int(time.time() * 1000)}) + "\n")

def main(epochs: int = 10, batch_size: int = 4, lr: float = 1e-4):
    """Train ResNet18 on CT cache; saves best checkpoint to OUTPUTS/ct_baseline_best.pt."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    train_ds = CTDataset(split="train")
    val_ds = CTDataset(split="val")
    # #region agent log
    _agent_log("train_start", {"train_len": len(train_ds), "val_len": len(val_ds), "outputs_dir": str(OUTPUTS), "outputs_exists": OUTPUTS.exists()}, "H4")
    # #endregion
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_ct_model()
    model = model.cuda() if torch.cuda.is_available() else model
    device = next(model.parameters()).device
    # #region agent log
    _agent_log("train_device", {"device": str(device), "cuda_available": torch.cuda.is_available()}, "H5")
    # #endregion

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
            # #region agent log
            if ep == 0 and bi == 0:
                _agent_log("train_first_batch", {"x_shape": list(x.shape), "y_shape": list(y.shape)}, "H5")
            # #endregion
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
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--epochs", type=int, default=10)
        ap.add_argument("--batch-size", type=int, default=4)
        ap.add_argument("--lr", type=float, default=1e-4)
        args = ap.parse_args()
        main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    except Exception as e:
        import traceback
        print("FLARE_DEBUG: EXCEPTION", str(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
