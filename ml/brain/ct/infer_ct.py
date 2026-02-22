"""
Infer CT: load model, run on cache (can be single patient or batch from the manifest)
"""

import argparse 
import csv 
import json 
import sys 
from pathlib import Path 

import numpy as np 
import torch 

ROOT = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(ROOT)) 

from src.config import CACHE_CT, META, OUTPUTS 
from ml.brain.ct.model_ct import build_ct_model 

LABELS = ["normal", "abnormal"]
 
# Same split as dataset_ct (seed 42, 0.7 / 0.15 / 0.15) 
SPLIT_SEED = 42 
TRAIN_FRAC =  0.7 
VAL_FRAC = 0.15

def _load_manifest(manifest: Path) -> list[dict]:
    """Load manifest JSON as list of dict entries."""
    with open(manifest, encoding="utf-8") as f:
        return json.load(f) 

def _get_split_entries(entries: list[dict], split: str): 
    """
    Filter entries by split using same scheme as CTDataset. 
    - valid entries: label in {0, 1} (if none, use all entries)
    - train/val/test: 70/15/15 split with fixed seed
    - all: all valid entries
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
    split_map = {"train" : train_idx, "val" : val_idx, "test": test_idx} 
    idx_set = split_map.get(split, val_idx) 

    out: list[dict] = [] 
    for i, e in enumerate(valid): 
        if i in idx_set: 
            e_copy = dict(e) 
            e_copy["split"] = split 
            out.append(e_copy) 
    return out


def _run_single_inference(
    checkpoint: Path,
    patient_id: str | None = None,
    npz_path: Path | None = None,
) -> None:
    """Run single inference for a given patient or NPZ file."""
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_ct_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = next(model.parameters()).device

    if npz_path and npz_path.exists():
        arr = np.load(npz_path)["arr"]
        inp_desc = f"npz {npz_path}"
    elif patient_id:
        manifest = META / "ct_processed_manifest.json"
        if not manifest.exists():
            print(f"Manifest not found: {manifest}")
            return

        entries = _load_manifest(manifest)
        path: Path | None = None
        for e in entries:
            if e.get("patient_id") == patient_id:
                path = Path(e["path"])
                break
        if not path or not path.exists():
            print(f"Patient {patient_id} not in cache/manifest")
            return
        arr = np.load(path)["arr"]
        inp_desc = f"patient_id {patient_id}"
    else:
        print("Provide --patient-id or --npz for single-patient mode, or leave out both for batch mode")
        return

    print(f"Running inference for {inp_desc}")
    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    print(f"Prediction: {LABELS[pred]} (confidence={conf:.4f})")
    for i, lab in enumerate(LABELS):
        print(f"  {lab}: {probs[i]:.4f}") 

def _run_batch_inference(
    checkpoint: Path, 
    manifest_path: Path, 
    split: str = "val", 
    limit: int | None = None, 
    random_n: int | None = None, 
    out_json: Path | None = None, 
    out_csv: Path | None = None,
) -> list[dict]:
    """
    Batch infer using manifest: 
    - loads ckpt + manifest one time 
    - Filters by split, then applies limit or random 
    - returns list of dicts: 
    {patient_id, split, true_label, pred_label, confidence, p_normal, p_abnormal}
    """
    if not checkpoint.exists(): 
        print(f"Checkpoint not found: {checkpoint}")
        return [] 

    if not manifest_path.exists(): 
        print(f"Manifest not found ")
        return [] 

    try: 
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True) 
    except TypeError: 
        ckpt = torch.load(checkpoint, map_location="cpu") 
    model = build_ct_model(pretrained=False) 
    model.load_state_dict(ckpt["model"]) 
    model.eval() 
    device = next(model.parameters()).device 

    entries = _load_manifest(manifest_path) 
    entries = _get_split_entries(entries, split) 

    if not entries: 
        print(f"No entries found for split={split}") 
        return [] 

    print(f"Running batch inference on {len(entries)} patients (split={split})...")

    results: list[dict] = []    
    for e in entries: 
        path = Path(e["path"]) 
        if not path.exists(): 
            print(f"  SKIP {e.get('patient_id', '')}: npz not found at {path}"
            continue 
        arr = np.load(path)["arr"] 
        x = torch.from_numpy(arr).float().unsqueeze(0).to(device)
        with torch.no_grad(): 
            logits = model(x) 
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) 
        true_label = int(e.get("label", -1)) 

        row = {
            "patient_id": e.get("patient_id", ""), 
            "split": e.get("split", split), 
            "true_label": true_label,
            "pred_label": pred, 
            "confidence": conf, 
            "p_normal": float(probs[0]),
            "p_abnormal": float(probs[1]),
        } 
        results.append(row)

    num_pred = len(results)
    print(f"Predicted {num_pred} patients (split={split}, limit={limit}, random_n={random_n}).")
    labeled = [r for r in results if r["true_label"] in (0, 1)]
    if labeled:
        correct = sum(1 for r in labeled if r["true_label"] == r["pred_label"])
        acc = correct / len(labeled)
        print(f"Accuracy on labeled subset: {acc:.3f} ({correct}/{len(labeled)})")
    else:
        print("No ground-truth labels in manifest; accuracy not computed.")

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "patient_id",
                    "split",
                    "true_label",
                    "pred_label",
                    "confidence",
                    "p_normal",
                    "p_abnormal",
                ],
            )
            w.writeheader()
            w.writerows(results)
        print(f"Wrote CSV to {out_csv}")

    return results 

def main(
    patient_id: str | None = None,
    npz_path: Path | None = None,
    checkpoint: Path | None = None,
    split: str = "val",
    limit: int | None = None,
    random_n: int | None = None,
    out_json: Path | None = None,
    out_csv: Path | None = None,
) -> None:
    """Dispatch single-patient vs batch mode."""
    checkpoint = checkpoint or OUTPUTS / "ct_baseline_best.pt"
    manifest_path = META / "ct_processed_manifest.json"

    # Single patient mode
    if patient_id or (npz_path and npz_path.exists()):
        _run_single_inference(
            checkpoint=checkpoint,
            patient_id=patient_id,
            npz_path=npz_path,
        )
        return

    _run_batch_inference(
        checkpoint=checkpoint,
        manifest_path=manifest_path,
        split=split,
        limit=limit,
        random_n=random_n,
        out_json=out_json,
        out_csv=out_csv,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run CT inference (single patient or batch from manifest).")
    ap.add_argument("--patient-id", type=str, default=None, help="Patient ID for single-patient mode")
    ap.add_argument("--npz", type=Path, default=None, help="Path to NPZ file for single-patient mode")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint (default: outputs/ct_baseline_best.pt)")
    ap.add_argument("--split", type=str, default="val", help="Split for batch mode: train/val/test/all")
    ap.add_argument("--limit", type=int, default=None, help="Max number of patients in batch mode")
    ap.add_argument("--random-n", type=int, default=None, help="Random subsample size in batch mode")
    ap.add_argument("--out-json", type=Path, default=None, help="Output JSON path for batch results")
    ap.add_argument("--out-csv", type=Path, default=None, help="Output CSV path for batch results")
    args = ap.parse_args()
    main(
        patient_id=args.patient_id,
        npz_path=args.npz,
        checkpoint=args.checkpoint,
        split=args.split,
        limit=args.limit,
        random_n=args.random_n,
        out_json=args.out_json,
        out_csv=args.out_csv,
    )
