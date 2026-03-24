"""
Demo: Single patient (--patient-id/--npz) or batch from manifest (no id => batch).
Design: CAM flags (--cam-dir, --cam-limit, --cam-when-abnormal) toggle gradients
and overlay saving; inference stays lightweight when CAM is off.
"""

import argparse 
import csv 
import json 
import sys 
from pathlib import Path 
from typing import Optional, List, Dict 

import numpy as np 
import torch 

ROOT = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(ROOT)) 

from src.config import META, OUTPUTS
from ml.brain.ct.model_ct import (
    build_ct_model,
    build_ct_sequence_model,
    is_sequence_ct_model,
    patient_logits_from_model,
)
from ml.brain.ct.gradcam_ct import GradCAM

LABELS = ["normal", "abnormal"]

# Decision threshold for abnormal (class 1). Use 0.55 for validation-tuned boundary.
ABNORMAL_THRESHOLD = 0.55

SPLIT_SEED = 42 


def _save_overlay_png(overlay: np.ndarray, out_path: Path) -> bool:
    """Save RGB overlay (H,W,3) uint8 to PNG. Tries cv2 then matplotlib. Returns True if saved."""
    try:
        import cv2
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return True
    except ImportError:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.imsave(str(out_path), overlay)
        return True
    except ImportError:
        return False
TRAIN_FRAC = 0.7 
VAL_FRAC = 0.15 

def _load_patient_arr(path: Path) -> np.ndarray:
    """Load NPZ; return arr (k, C, H, W) float32. Handles (C, H,W) -> (1,C,H,W)"""
    data = np.load(path)
    arr = data["arr"]
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    return arr.astype(np.float32)


def _imagenet_normalize_volume(x: torch.Tensor) -> torch.Tensor:
    """Match CTDataset val/test: (k, C, H, W) with C=3."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std

def _load_manifest(manifest: Path) -> list: 
    with open(manifest, encoding="utf-8") as f:  
        return json.load(f) 

def _get_split_entries(entries: list, split: str): 
    """
    Deterministic split using fixed RNG seed. Matches train/val/test fractions. 
    """ 
    valid = [e for e in entries if e.get("label", -1) in (0, 1)] 
    if not valid: 
        valid = entries 

    if split == "all": 
        out = [] 
        for e in valid: 
            e_copy = dict(e) 
            e_copy.setdefault("split", "all") 
            out.append(e_copy) 
        return out 

    rng = np.random.default_rng(SPLIT_SEED) 
    n = len(valid) 
    inx = rng.permutation(n) 
    
    nt = int(n * TRAIN_FRAC) 
    nv = int(n * VAL_FRAC) 

    train_idx = set(inx[:nt]) 
    val_idx = set(inx[nt : nt + nv]) 
    test_idx = set(inx[nt + nv :])

    split_map = {"train": train_idx, "val": val_idx, "test": test_idx} 
    idx_set = split_map.get(split, val_idx) 

    out = [] 
    for i, e in enumerate(valid): 
        if i in idx_set: 
            e_copy = dict(e) 
            e_copy["split"] = split 
            out.append(e_copy) 

    return out 

def _apply_limit_and_random(entries: list, limit: Optional[int], random_n: Optional[int]) -> list: 
    """
    Limit: take first N deterministically 
    random_n: take N random entries deterministically 
    If both provided: random_n applied first, then limit. 
    """

    out = list(entries) 

    if random_n is not None: 
        rng = np.random.default_rng(SPLIT_SEED) 
        if random_n < len(out): 
            idx = rng.choice(len(out), size=random_n, replace=False) 
            out = [out[i] for i in idx] 

    if limit is not None: 
        out = out[:limit] 

    return out 

def _load_model(checkpoint: Path):
    """Load checkpoint; returns (model, model_kind, agg_for_legacy)."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found {checkpoint}")

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_kind = ckpt.get("model_kind", "legacy") if isinstance(ckpt, dict) else "legacy"
    agg = ckpt.get("agg", "max") if isinstance(ckpt, dict) else "max"
    rnn_type = ckpt.get("rnn_type", "lstm") if isinstance(ckpt, dict) else "lstm"

    if model_kind == "sequence":
        model = build_ct_sequence_model(rnn_type=rnn_type)
    else:
        model = build_ct_model()

    model.load_state_dict(state, strict=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, model_kind, agg

def _predict_one(
    model: torch.nn.Module,
    x: torch.Tensor,
    use_grad: bool,
    agg: str = "max",
) -> np.ndarray:
    """
    x: (1, k, C, H, W) after ImageNet normalization.
    Returns probs as numpy array shape (2,).
    """
    if use_grad:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = patient_logits_from_model(model, x, agg=agg)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().detach().numpy()
        return probs

    with torch.no_grad():
        logits = patient_logits_from_model(model, x, agg=agg)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs

def _gradcam_layer_name(model: torch.nn.Module) -> str:
    return "backbone.features.denseblock4" if is_sequence_ct_model(model) else "features.denseblock4"


def _run_single_inference(
    checkpoint: Path,
    patient_id: Optional[str] = None,
    npz_path: Optional[Path] = None,
    cam_dir: Optional[Path] = None,
) -> None:
    model, _model_kind, agg = _load_model(checkpoint)
    device = next(model.parameters()).device

    # Demo: loads NPZ from manifest lookup or direct --npz path (no DICOM here).
    if npz_path and npz_path.exists():
        path = npz_path
        inp_desc = f"npz {npz_path}"
        cam_id = npz_path.stem
    elif patient_id:
        manifest = META / "ct_processed_manifest.json"
        if not manifest.exists():
            print(f"Manifest not found: {manifest}")
            return
        entries = _load_manifest(manifest)
        path = None
        for e in entries:
            if e.get("patient_id") == patient_id:
                path = Path(e["path"])
                break
        if not path or not path.exists():
            print(f"Patient {patient_id} not found in cache/manifest")
            return
        inp_desc = f"patient_id {patient_id}"
        cam_id = patient_id
    else:
        print("Provide --patient-id or --npz for single patient mode, or leave both out for batch mode.")
        return

    print(f"Running inference on {inp_desc}")

    arr = _load_patient_arr(path)
    x_raw = torch.from_numpy(arr).float().to(device)
    x_all = _imagenet_normalize_volume(x_raw)
    x_batch = x_all.unsqueeze(0)
    use_grad = cam_dir is not None
    probs = _predict_one(model, x_batch, use_grad=use_grad, agg=agg)
    pred = 1 if float(probs[1]) >= ABNORMAL_THRESHOLD else 0
    conf = float(probs[pred])
    if cam_dir is not None:
        k = x_raw.shape[0]
        center_raw = x_raw[k // 2 : k // 2 + 1]
        if is_sequence_ct_model(model):
            x_cam = x_all[k // 2 : k // 2 + 1].unsqueeze(0)
        else:
            x_cam = x_all[k // 2 : k // 2 + 1]
        gradcam = GradCAM(model, _gradcam_layer_name(model))
        _, overlay = gradcam(x_cam, target_class=pred, input_for_overlay=center_raw)
        if overlay is not None:
            cam_dir = Path(cam_dir)
            cam_dir.mkdir(parents=True, exist_ok=True)
            out_path = cam_dir / f"{cam_id}.png"
            if _save_overlay_png(overlay, out_path):
                print(f"CAM overlay saved to {out_path}")
            else:
                print("Install opencv-python or matplotlib to save CAM overlay.")

    print(f"Prediction: {LABELS[pred]} (confidence={conf:.4f})")
    for i, lab in enumerate(LABELS):
        print(f" {lab}: {probs[i]:.4f}")

def _run_batch_inference(
    checkpoint: Path,
    manifest_path: Path,
    split: str = "val",
    limit: Optional[int] = None,
    random_n: Optional[int] = None,
    out_csv: Optional[Path] = None,
    cam_dir: Optional[Path] = None,
    cam_limit: Optional[int] = None,
    cam_when_abnormal: bool = False,
) -> list: 
    if not manifest_path.exists(): 
        print(f"Manifest not found: {manifest_path}") 
        return []
    
    model, _model_kind, agg = _load_model(checkpoint)
    device = next(model.parameters()).device

    entries = _load_manifest(manifest_path)
    entries = _get_split_entries(entries, split) 
    entries = _apply_limit_and_random(entries, limit=limit, random_n=random_n) 

    if not entries:
        print(f"No entries found for split={split}")
        return []

    # Design: loop over manifest entries one-by-one (batch_size=1; laptop-friendly).
    print(f"Running batch inference on {len(entries)} patients (split={split})...")
    results: List[Dict] = []
    cam_count = 0
    csv_has_cam = False 

    use_grad = cam_dir is not None
    gradcam = GradCAM(model, _gradcam_layer_name(model)) if cam_dir is not None else None

    for e in entries:
        path = Path(e["path"])
        pid = e.get("patient_id", "")

        if not path.exists():
            print(f" SKIP {pid}: npz not found at {path}")
            continue

        arr = _load_patient_arr(path)
        x_raw = torch.from_numpy(arr).float().to(device)
        x_all = _imagenet_normalize_volume(x_raw)
        x_batch = x_all.unsqueeze(0)
        probs = _predict_one(model, x_batch, use_grad=use_grad, agg=agg)
        pred = 1 if float(probs[1]) >= ABNORMAL_THRESHOLD else 0
        conf = float(probs[pred])

        row: Dict = {
            "patient_id": pid,
            "split": e.get("split", split),
            "true_label": int(e.get("label", -1)),
            "pred_label": pred,
            "confidence": conf,
            "p_normal": float(probs[0]),
            "p_abnormal": float(probs[1]),
        }

        # Note: optional CAM limit + abnormal-only filter (--cam-limit, --cam-when-abnormal).
        do_cam = (
            cam_dir is not None
            and (cam_limit is None or cam_count < cam_limit)
            and (not cam_when_abnormal or pred == 1)
        )

        if do_cam and gradcam is not None:
            k = x_raw.shape[0]
            center_raw = x_raw[k // 2 : k // 2 + 1]
            if is_sequence_ct_model(model):
                x_cam = x_all[k // 2 : k // 2 + 1].unsqueeze(0)
            else:
                x_cam = x_all[k // 2 : k // 2 + 1]
            _, overlay = gradcam(x_cam, target_class=pred, input_for_overlay=center_raw)
            if overlay is not None:
                cam_dir_p = Path(cam_dir)
                cam_dir_p.mkdir(parents=True, exist_ok=True)
                out_path = cam_dir_p / f"{pid}.png"
                if _save_overlay_png(overlay, out_path):
                    row["cam_path"] = str(out_path)
                    csv_has_cam = True
                    cam_count += 1
                else:
                    print("Install opencv-python or matplotlib to save CAM overlay.")

        results.append(row)

    num_pred = len(results)
    print(f"Predicted {num_pred} patients (split={split}).")

    labeled = [r for r in results if r["true_label"] in (0, 1)]
    if labeled:
        correct = sum(1 for r in labeled if r["true_label"] == r["pred_label"])
        acc = correct / len(labeled)
        print(f"Accuracy on labeled subset: {acc:.3f} ({correct}/{len(labeled)})")
    else:
        print("No ground-truth labels in manifest; accuracy not computed.")

    # Output: export predictions for analysis (CSV with optional cam_path column).
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "patient_id", "split", "true_label", "pred_label",
            "confidence", "p_normal", "p_abnormal",
        ]
        if csv_has_cam:
            fieldnames.append("cam_path")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"Wrote CSV to {out_csv}")

    return results


def main(
    patient_id: Optional[str] = None,
    npz_path: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    split: str = "val",
    limit: Optional[int] = None,
    random_n: Optional[int] = None,
    out_csv: Optional[Path] = None,
    cam_dir: Optional[Path] = None,
    cam_limit: Optional[int] = None,
    cam_when_abnormal: bool = False,
) -> None:
    checkpoint = checkpoint or OUTPUTS / "ct_baseline_best.pt"
    manifest_path = META / "ct_processed_manifest.json"

    if patient_id or (npz_path and npz_path.exists()):
        _run_single_inference(
            checkpoint=checkpoint,
            patient_id=patient_id,
            npz_path=npz_path,
            cam_dir=cam_dir,
        )
        return

    _run_batch_inference(
        checkpoint=checkpoint,
        manifest_path=manifest_path,
        split=split,
        limit=limit,
        random_n=random_n,
        out_csv=out_csv,
        cam_dir=cam_dir,
        cam_limit=cam_limit,
        cam_when_abnormal=cam_when_abnormal,
    )

def run_ct_for_patient(patient_id, checkpoint=None, cam_dir="backend/camo_outpus"):
    """Runs CT inference + GradCAM for one patient. Returns plain dict."""
    import numpy as np
    import torch
    from pathlib import Path
    from src.config import META, OUTPUTS

    if checkpoint is None:
        checkpoint = OUTPUTS / "ct_baseline_best.pt"
    manifest_path = META / "ct_processed_manifest.json"

    entries = _load_manifest(manifest_path)
    entry = next((e for e in entries if e.get("patient_id") == patient_id), None)
    if entry is None:
        return None

    arr_path = Path(entry["path"])
    if not arr_path.exists():
        return None

    model, _model_kind, agg = _load_model(checkpoint)
    device = next(model.parameters()).device

    arr = _load_patient_arr(arr_path)
    x_raw = torch.from_numpy(arr).float().to(device)
    x_all = _imagenet_normalize_volume(x_raw)
    x_batch = x_all.unsqueeze(0)
    use_grad = cam_dir is not None
    probs = _predict_one(model, x_batch, use_grad=use_grad, agg=agg)
    pred = 1 if float(probs[1]) >= ABNORMAL_THRESHOLD else 0

    cam_path = None
    if use_grad and cam_dir:
        gradcam = GradCAM(model, _gradcam_layer_name(model))
        k = x_raw.shape[0]
        center_raw = x_raw[k // 2 : k // 2 + 1]
        if is_sequence_ct_model(model):
            x_cam = x_all[k // 2 : k // 2 + 1].unsqueeze(0)
        else:
            x_cam = x_all[k // 2 : k // 2 + 1]
        _, overlay = gradcam(x_cam, target_class=pred, input_for_overlay=center_raw)
        if overlay is not None:
            cam_dir_p = Path(cam_dir)
            cam_dir_p.mkdir(parents=True, exist_ok=True)
            cam_path_obj = cam_dir_p / f"{patient_id}.png"
            if _save_overlay_png(overlay, cam_path_obj):
                cam_path = str(cam_path_obj)

    return {
        "patient_id": patient_id,
        "label": LABELS[pred],
        "confidence": float(probs[pred]),
        "p_normal": float(probs[0]),
        "p_abnormal": float(probs[1]),
        "cam_path": cam_path,
    }


if __name__ == "__main__":
    # CLI flags control demo behavior: single vs batch, checkpoint, CAM options.
    ap = argparse.ArgumentParser(description="Run CT inference (single patient or batch from manifest).")
    ap.add_argument("--patient-id", type=str, default=None)
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--random-n", type=int, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--cam-dir", type=Path, default=None, help="Save CAM overlays here (optional)")
    ap.add_argument("--cam-limit", type=int, default=None, help="Max CAMs in batch (optional)")
    ap.add_argument("--cam-when-abnormal", action="store_true", help="Only CAM for pred=abnormal in batch")
    args = ap.parse_args()

    main(
        patient_id=args.patient_id,
        npz_path=args.npz,
        checkpoint=args.checkpoint,
        split=args.split,
        limit=args.limit,
        random_n=args.random_n,
        out_csv=args.out_csv,
        cam_dir=args.cam_dir,
        cam_limit=args.cam_limit,
        cam_when_abnormal=args.cam_when_abnormal,
    )


