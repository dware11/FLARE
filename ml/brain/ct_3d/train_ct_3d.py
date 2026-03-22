"""
Train a true 3D CT classifier.

Key features:
- dataset returns (C, D, H, W)
- DataLoader pads only within a batch using a custom collate_fn
- preserves original patient depth in preprocessing / manifest
- ignores label=-1 in loss and metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUTS
from ml.brain.ct_3d.dataset_ct_3d import CTDataset
from ml.brain.ct_3d.model_ct_3d import build_ct_model


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_ct_3d(batch):
    """
    Pads depth only to the max depth within the batch.
    Keeps the real patient depth in the dataset/manifest.
    """
    xs, ys, pids = zip(*batch)

    max_d = max(x.shape[1] for x in xs)
    padded_xs = []

    for x in xs:
        c, d, h, w = x.shape
        if d < max_d:
            pad_total = max_d - d
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front
            front = torch.zeros((c, pad_front, h, w), dtype=x.dtype)
            back = torch.zeros((c, pad_back, h, w), dtype=x.dtype)
            x = torch.cat([front, x, back], dim=1)
        padded_xs.append(x)

    X = torch.stack(padded_xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y, list(pids)


def compute_class_weights(train_ds: CTDataset, device: torch.device) -> torch.Tensor:
    labels = []
    for e in train_ds.entries:
        y = int(e.get("label", -1))
        if y in (0, 1):
            labels.append(y)

    if len(labels) == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    count_0 = sum(1 for y in labels if y == 0)
    count_1 = sum(1 for y in labels if y == 1)

    count_0 = max(count_0, 1)
    count_1 = max(count_1, 1)

    total = count_0 + count_1
    w0 = total / (2.0 * count_0)
    w1 = total / (2.0 * count_1)

    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0
    tp = tn = fp = fn = 0

    all_y = []
    all_preds = []
    all_probs_pos = []

    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            batch_loss = criterion(logits, y)
            val_loss += batch_loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            for i in range(y.size(0)):
                yi = y[i].item()
                pi = preds[i].item()

                if yi not in (0, 1):
                    continue

                total += 1
                all_y.append(yi)
                all_preds.append(pi)
                all_probs_pos.append(probs[i, 1].item())

                if pi == yi:
                    correct += 1
                if pi == 1 and yi == 1:
                    tp += 1
                elif pi == 0 and yi == 0:
                    tn += 1
                elif pi == 1 and yi == 0:
                    fp += 1
                else:
                    fn += 1

    avg_loss = val_loss / len(loader) if len(loader) > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    return {
        "loss": avg_loss,
        "acc": acc,
        "sens": sens,
        "spec": spec,
        "prec": prec,
        "f1": f1,
        "total_eval": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "all_y": all_y,
        "all_preds": all_preds,
        "all_probs_pos": all_probs_pos,
    }


def train(
    manifest_path: Optional[Path] = None,
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    seed: int = 42,
    out_path: Optional[Path] = None,
    pretrained: bool = False,
) -> None:
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = CTDataset(manifest_path=manifest_path, split="train", seed=seed)
    val_ds = CTDataset(manifest_path=manifest_path, split="val", seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_ct_3d,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_ct_3d,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    class_weights = compute_class_weights(train_ds, device)
    print(f"Class weights: {class_weights.tolist()}")

    model = build_ct_model(num_classes=2, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_path = out_path or (OUTPUTS / "ct_3d_best.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history_path = out_path.with_suffix(".history.json")
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X, y, _ in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_avg:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} "
            f"sens={val_metrics['sens']:.4f} "
            f"spec={val_metrics['spec']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"n_eval={val_metrics['total_eval']}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_avg,
                "val_loss": val_metrics["loss"],
                "acc": val_metrics["acc"],
                "sens": val_metrics["sens"],
                "spec": val_metrics["spec"],
                "prec": val_metrics["prec"],
                "f1": val_metrics["f1"],
                "n_eval": val_metrics["total_eval"],
                "tp": val_metrics["tp"],
                "tn": val_metrics["tn"],
                "fp": val_metrics["fp"],
                "fn": val_metrics["fn"],
            }
        )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                out_path,
            )
            print(f"Saved best checkpoint to {out_path}")

    print("Training complete.")
    print(f"Best checkpoint: {out_path}")
    print(f"History saved to: {history_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train 3D CT classifier")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    train(
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        out_path=args.out,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
