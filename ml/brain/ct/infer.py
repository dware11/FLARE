""" 
Infer CT: Load mode, run on cache (single patient or batch from manifest) 
Grad-CAM is optional. It turns ON only when --cam-dir is put. 
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
from ml.brain.ct.model_ct import build_ct_model 
from ml.brain.ct.gradcam import GradCAM 

LABELS = ["normal", "abnormal"] 

SPLIT_SEED = 42 
TRAIN_FRAC = 0.7 
VAL_FRAC = 0.15 

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
    val_idx = set(int[nt : nt + nv]) 
    test_idx = set(idx[nt + nv :])

    split_map = {"train": train_idx, "val": val_idx, "test": test_idx} 
    idx_set = split_map.get(split, val_idx) 

    out = [] 
    for i, e in enumerate(valid): 
        if in in idx_set: 
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
            idx = rng.choice(len(out, size=random_n, replace=False)) 
            out = [out[i] for i in idx] 

    if limit is not None: 
        out = out[:limit] 

    return out 

def _load_model(checkpoint: Path) -> torch.nn.Module: 
    if not checkpoint.exists(): 
        raise FileNotFoundError(f"Checkpoint not found {checkpiont}")

    try: 
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True) 
    except TypeError: 
        ckpt = torch.load(chekckpoint, map_loaction="cpu") 

def _predict_one(
    model: torch.nn.Module,
    x: torch.Tensor, 
    use_grad: bool, 
) -> np.ndarray: 
    """
    Returns probs as numpy array shape (2,). 
    If use_grad is True, forward happens with gradiens enables (needed for grad-cam) 
    """
    if use_grad: 
        model.zero_grad(set_to_none=True) 
        with torch.enable_grad(): 
            logits = model(x) 
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().detach().numpy() 
        return probs

    with torch.no_grad(): 
        logits = model(x) 
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy() 
    return probs 

def _run_single_inference(
    checkpoint: Path, 
    patient_id: str | None = None,
    npz_path: Path | None = None, 
    cam_dir: Path | None = None, 
) -> None: 

    model = _load_model(checkpoint) 
    device = next(model.parameters()).device

    if npz_path and npz_path.exists(): 
        arr = np.load(npz_path)["arr"] 
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

        arr = np.load(path)["arr"] 
        inp_desc = f"patient_id {patient_id}" 
        cam_id = patient_id 
    else:  
        print("Provide --patient-id or --npz for single patient mode, or leave both out for batch mode.") 
        return 

    print(f"Running inference on {inp_desc}") 

    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)
    use_grad = cam_dir is not None 

    probs = _predict_one(model, x, use_grad=use_grad) 
    pred = in(np.argmax(probs))
    conf = float(probs[pred]) 

    # Optional Grad-Cam 
    if cam_dir is not None: 
        gradcam = GradCAM(model, "layer4") 
        _, overlay = gradcam(x, target_class=pred, input_for_overlay=x) 
        if overlay is not None: 
            cam_dir = Path(cam_dir) 
            cam_dir.mkdir(parents=True, exist_ok=True) 
            out_path = cam_dir / f"{cam_id}.png" 
            try: 
                import cv2 
                cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)) 
                print(f"CAM overlay saved to {out_path}") 
            except ImportError: 
                print("cv2 not installed: CAM heatmap not saved.") 

        print(f"Prediction: {LABELS[pred]} (confidence={conf:.4f})") 
        for i, lab in enumerate(LABELS):
            print(f" {lab}: {probs[i]:.4f}")

def _run_batch_inference(
    checkpoint: Path, 
    manifest_path: Path, 
    split: str = "val", 
    limit: int | None = None, 
    random_n: int | None = None, 
    out_csv: Path | None = None, 
    cam_dir: Path | None = None, 
    cam_limit: int | None = None, 
    cam_when_abnormal: bool = False,
) -> list: 
    if not manifest_path.exists(): 
        print(f"Manifest not found: {manifest_path}") 
        return []
    
    model = _load_model(checkpoint) 
    device = next(model.parameters()).device

    entries = _load_manifest(manifest_path) 
    entries = _get_split_entries(entries, split) 
    entries = _apply_limit_and_random(entries, limit=limit, random)
