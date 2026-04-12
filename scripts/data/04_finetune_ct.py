#!/usr/bin/env python3
"""
Fine-tune CTSlicedSequenceModel on multimodal CT NPZ (k=1 slice).
Loads RSNA checkpoint like ml/brain/ct/infer.py _load_model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.brain.ct.model_ct import (  # noqa: E402
    build_ct_model,
    build_ct_sequence_model,
    is_sequence_ct_model,
    load_sequence_weights_compat,
    patient_logits_from_model,
)

DEFAULT_BASE_CKPT = Path("/scratch/bckk/flare/ct_brain/outputs/rsna_windowed_exp2/best.pt")
DEFAULT_OUT_DIR = Path("/scratch/bckk/flare/ct_brain/outputs/multimodal_finetune_ct")
DEFAULT_DATA = Path("/scratch/bckk/flare/multimodal_brain/preprocessed/CT")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(chw: torch.Tensor) -> torch.Tensor:
    """(3,H,W) normalized -> [0,1]."""
    return (chw * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)


def _norm(chw: torch.Tensor) -> torch.Tensor:
    return (chw - IMAGENET_MEAN.to(chw.device)) / IMAGENET_STD.to(chw.device)


def _augment_train(chw: torch.Tensor) -> torch.Tensor:
    """chw: (3,H,W) on CPU or CUDA."""
    import torchvision.transforms.functional as TF

    x = _denorm(chw)
    if random.random() < 0.5:
        x = TF.hflip(x)
    angle = random.uniform(-15.0, 15.0)
    x = TF.rotate(x, angle, fill=[0.0, 0.0, 0.0])
    b = 1.0 + random.uniform(-0.2, 0.2)
    c = 1.0 + random.uniform(-0.2, 0.2)
    x = TF.adjust_brightness(x, b)
    x = TF.adjust_contrast(x, c)
    x = x.clamp(0.0, 1.0)
    return _norm(x)


class CTNpzDataset(Dataset):
    def __init__(self, paths: List[Path], train: bool) -> None:
        self.paths = paths
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.paths[idx]
        data = np.load(p)
        img = torch.from_numpy(np.asarray(data["image"], dtype=np.float32))
        y = int(np.asarray(data["label"]).reshape(-1)[0])
        if self.train:
            img = _augment_train(img)
        x = img.unsqueeze(0)
        return x, torch.tensor(y, dtype=torch.long)


def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    xb = torch.stack(xs, dim=0)
    yb = torch.stack(ys, dim=0)
    return xb, yb


def _gather_npz_paths(root: Path, split: str) -> List[Path]:
    d = root / split
    if not d.is_dir():
        return []
    return sorted([p for p in d.glob("*.npz") if p.is_file()])


def _count_by_label(paths: List[Path]) -> Tuple[int, int]:
    n0 = n1 = 0
    for p in paths:
        try:
            data = np.load(p)
            lab = int(np.asarray(data["label"]).reshape(-1)[0])
            if lab == 0:
                n0 += 1
            elif lab == 1:
                n1 += 1
        except Exception:
            continue
    return n0, n1


def _load_raw_checkpoint_dict(checkpoint: Path) -> Dict[str, Any]:
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    return ckpt if isinstance(ckpt, dict) else {}


def load_model_from_checkpoint(checkpoint: Path, device: torch.device) -> Tuple[nn.Module, str, Dict[str, Any]]:
    """Match ml/brain/ct/infer.py _load_model (CPU load; caller sets train/eval)."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found {checkpoint}")
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_kind = ckpt.get("model_kind", "legacy") if isinstance(ckpt, dict) else "legacy"
    agg = ckpt.get("agg", "max") if isinstance(ckpt, dict) else "max"
    rnn_type = ckpt.get("rnn_type", "gru") if isinstance(ckpt, dict) else "gru"
    sequence_pool = ckpt.get("sequence_pool", "max") if isinstance(ckpt, dict) else "max"
    sequence_topk = ckpt.get("sequence_topk", 3) if isinstance(ckpt, dict) else 3
    rnn_dropout = ckpt.get("rnn_dropout", 0.25) if isinstance(ckpt, dict) else 0.25
    use_thickness = ckpt.get("use_thickness", False) if isinstance(ckpt, dict) else False

    meta: Dict[str, Any] = dict(ckpt) if isinstance(ckpt, dict) else {}

    if model_kind == "sequence":
        if isinstance(state, dict) and "head.weight" in state and "slice_head.weight" not in state:
            sequence_pool = "center"
        model = build_ct_sequence_model(
            rnn_type=rnn_type,
            sequence_pool=sequence_pool,
            topk=sequence_topk,
            rnn_dropout=rnn_dropout,
            use_thickness=use_thickness,
        )
        load_sequence_weights_compat(model, state)
    else:
        model = build_ct_model(pretrained=False)
        model.load_state_dict(state, strict=True)

    model = model.to(device)
    return model, agg, meta


def _freeze_early_densenet(model: nn.Module) -> None:
    if not is_sequence_ct_model(model):
        return
    feats = getattr(model.backbone, "features", None)
    if feats is None:
        return
    for name in ("denseblock1", "transition1", "denseblock2", "transition2"):
        mod = getattr(feats, name, None)
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad = False


def _safe_auc(y_true: List[int], y_score: List[float]) -> float:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _metrics_binary(y_true: np.ndarray, y_prob: np.ndarray, thresh: float) -> Tuple[float, float, float]:
    """Sensitivity (abnormal recall), specificity (normal recall), accuracy."""
    y_pred = (y_prob >= thresh).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / max(tp + tn + fp + fn, 1))
    return sens, spec, acc


def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    agg: str,
) -> Tuple[float, float, float, float, float, List[int], List[float]]:
    model.eval()
    if len(loader) == 0:
        return float("nan"), float("nan"), 0.0, 0.0, 0.0, [], []
    total_loss = 0.0
    n_batches = 0
    all_y: List[int] = []
    all_p: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            thick = None
            logits = patient_logits_from_model(model, xb, agg=agg, thickness=thick)
            loss = criterion(logits, yb)
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                n_batches += 1
            prob = torch.softmax(logits, dim=1)[:, 1]
            for i in range(yb.size(0)):
                all_y.append(int(yb[i].item()))
                all_p.append(float(prob[i].item()))
    avg_loss = total_loss / max(n_batches, 1)
    auc = _safe_auc(all_y, all_p)
    y_arr = np.asarray(all_y, dtype=np.int64)
    p_arr = np.asarray(all_p, dtype=np.float64)
    sens, spec, acc = _metrics_binary(y_arr, p_arr, 0.5)
    return avg_loss, auc, acc, sens, spec, all_y, all_p


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    agg: str,
) -> Tuple[float, float, List[int], List[float]]:
    model.train()
    if len(loader) == 0:
        return float("nan"), float("nan"), [], []
    total = 0.0
    n_batches = 0
    all_y: List[int] = []
    all_p: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = patient_logits_from_model(model, xb, agg=agg, thickness=None)
        loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.item())
        n_batches += 1
        prob = torch.softmax(logits, dim=1)[:, 1].detach()
        for i in range(yb.size(0)):
            all_y.append(int(yb[i].item()))
            all_p.append(float(prob[i].item()))
    avg = total / max(n_batches, 1)
    auc = _safe_auc(all_y, all_p)
    return avg, auc, all_y, all_p


def _json_sanitize(obj: Any) -> Any:
    """Replace nan/inf with None for strict JSON."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(x) for x in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune CTSlicedSequenceModel on multimodal CT NPZ.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CKPT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = _gather_npz_paths(args.data_root, "train")
    val_paths = _gather_npz_paths(args.data_root, "val")
    test_paths = _gather_npz_paths(args.data_root, "test")
    if not train_paths:
        raise FileNotFoundError(f"No train NPZ under {args.data_root / 'train'}")

    tr0, tr1 = _count_by_label(train_paths)
    va0, va1 = _count_by_label(val_paths)
    te0, te1 = _count_by_label(test_paths)

    train_ds = CTNpzDataset(train_paths, train=True)
    val_ds = CTNpzDataset(val_paths, train=False)
    test_ds = CTNpzDataset(test_paths, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    base_ckpt_raw = _load_raw_checkpoint_dict(args.base_checkpoint)

    model, agg, _base_meta = load_model_from_checkpoint(args.base_checkpoint, device)
    if not is_sequence_ct_model(model):
        raise RuntimeError("Base checkpoint must be a sequence (CTSlicedSequenceModel) checkpoint.")
    _freeze_early_densenet(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None

    print(
        "Epoch | TrLoss | TrAUC | VlLoss | VlAUC | VlAcc | VlSens | VlSpec",
        flush=True,
    )

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_auc, _, _ = _train_one_epoch(
            model, train_loader, device, optimizer, criterion, agg
        )
        vl_loss, vl_auc, vl_acc, vl_sens, vl_spec, _, _ = _eval_loader(
            model, val_loader, device, criterion, agg
        )

        history.append(
            {
                "epoch": ep,
                "tr_loss": tr_loss,
                "tr_auc": tr_auc,
                "vl_loss": vl_loss,
                "vl_auc": vl_auc,
                "vl_acc": vl_acc,
                "vl_sensitivity": vl_sens,
                "vl_specificity": vl_spec,
            }
        )

        vl_auc_s = f"{vl_auc:.4f}" if math.isfinite(vl_auc) else "  nan"
        vl_loss_s = f"{vl_loss:.4f}" if math.isfinite(vl_loss) else "  nan"
        print(
            f"{ep:5d} | {tr_loss:.4f} | {tr_auc:.4f} | {vl_loss_s} | "
            f"{vl_auc_s} | {vl_acc:.4f} | {vl_sens:.4f} | {vl_spec:.4f}",
            flush=True,
        )

        if is_sequence_ct_model(model):
            scheduler.step()

        improved = False
        if len(val_loader) > 0 and math.isfinite(vl_loss):
            if math.isfinite(vl_auc):
                if vl_auc > best_val_auc + 1e-6:
                    best_val_auc = vl_auc
                    best_val_loss = min(best_val_loss, vl_loss)
                    best_epoch = ep
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    improved = True
            else:
                if vl_loss < best_val_loss - 1e-6:
                    best_val_loss = vl_loss
                    best_epoch = ep
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    improved = True

        if len(val_loader) == 0:
            epochs_no_improve = 0
        elif improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if len(val_loader) > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {ep} (no val AUC improvement for {args.patience} epochs).", flush=True)
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)

    # --- Test evaluation ---
    if len(test_loader) == 0:
        print("[warn] No test NPZ; skipping test metrics.", flush=True)
        test_y, test_p = [], []
        test_auc = float("nan")
        y_arr = np.array([], dtype=np.int64)
        p_arr = np.array([], dtype=np.float64)
        test_sens = test_spec = test_acc = 0.0
    else:
        _, _, _, _, _, test_y, test_p = _eval_loader(model, test_loader, device, criterion, agg)
        test_auc = _safe_auc(test_y, test_p)
        y_arr = np.asarray(test_y, dtype=np.int64)
        p_arr = np.asarray(test_p, dtype=np.float64)
        test_sens, test_spec, test_acc = _metrics_binary(y_arr, p_arr, 0.5)
    print("\n=== TEST RESULTS (multimodal_finetune_ct) ===", flush=True)
    print(f"AUC:         {test_auc:.4f}" if math.isfinite(test_auc) else "AUC:         nan", flush=True)
    print(f"Accuracy:    {100.0 * test_acc:.1f}%", flush=True)
    print(f"Sensitivity: {100.0 * test_sens:.1f}%   (recall on abnormal class = TP / (TP+FN))", flush=True)
    print(f"Specificity: {100.0 * test_spec:.1f}%   (recall on normal class   = TN / (TN+FP))", flush=True)
    print("Threshold:   0.5 (default)", flush=True)

    sweep_thresh = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    threshold_sweep: List[Dict[str, Any]] = []
    print("\n=== THRESHOLD SWEEP ===", flush=True)
    print("Thresh | Sens  | Spec  | Acc", flush=True)
    if y_arr.size > 0:
        for t in sweep_thresh:
            s, sp, a = _metrics_binary(y_arr, p_arr, t)
            threshold_sweep.append(
                {"threshold": t, "sensitivity": s, "specificity": sp, "accuracy": a}
            )
            mark = "   ← default" if abs(t - 0.5) < 1e-6 else ""
            print(f"{t:.2f}   | {s:.4f} | {sp:.4f} | {a:.4f}{mark}", flush=True)

    best_ckpt_path = args.output_dir / "best_ct_finetune.pt"
    infer_meta_keys = (
        "model_kind",
        "agg",
        "rnn_type",
        "sequence_pool",
        "sequence_topk",
        "rnn_dropout",
        "use_thickness",
        "k_slices",
    )
    payload: Dict[str, Any] = {k: base_ckpt_raw[k] for k in infer_meta_keys if k in base_ckpt_raw}
    payload["model"] = best_state
    payload["model_kind"] = "sequence"
    payload["agg"] = agg
    payload["finetune_experiment"] = "multimodal_finetune_ct"
    payload["base_checkpoint"] = str(args.base_checkpoint.resolve())
    payload["best_val_auc"] = best_val_auc
    payload["best_epoch"] = best_epoch
    torch.save(payload, best_ckpt_path)
    print(f"\nSaved {best_ckpt_path}", flush=True)

    meta_out = {
        "experiment": "multimodal_finetune_ct",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_experiment": "rsna_windowed_exp2",
        "dataset": "murtozalikhon/brain-tumor-multimodal-image-ct-and-mri",
        "ct_train_normal": tr0,
        "ct_train_abnormal": tr1,
        "ct_val_normal": va0,
        "ct_val_abnormal": va1,
        "ct_test_normal": te0,
        "ct_test_abnormal": te1,
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "test_auc": float(test_auc) if math.isfinite(test_auc) else None,
        "test_acc": float(test_acc),
        "test_sensitivity": float(test_sens),
        "test_specificity": float(test_spec),
        "frozen_layers": ["denseblock1", "transition1", "denseblock2", "transition2"],
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs_trained": len(history),
    }
    with open(args.output_dir / "checkpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    results: Dict[str, Any] = {
        "experiment": "multimodal_finetune_ct",
        "base_experiment": "rsna_windowed_exp2",
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "test_auc": float(test_auc) if math.isfinite(test_auc) else None,
        "test_acc": float(test_acc),
        "test_sensitivity": float(test_sens),
        "test_specificity": float(test_spec),
        "threshold_sweep": threshold_sweep,
        "history": history,
        "test_y_true": test_y,
        "test_y_score": test_p,
    }
    with open(args.output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(results), f, indent=2)

    print(f"Wrote {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
