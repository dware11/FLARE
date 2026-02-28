"""
Demo: Dataset reads cached NPZ only (not DICOM); manifest lists path + label.
"""
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import META

DEBUG_LOG = ROOT / "debug_pipeline.log"

def _agent_log(msg, data=None, hyp=""):
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"message": msg, "data": data or {}, "hypothesisId": hyp, "location": "dataset_ct", "timestamp": int(time.time() * 1000)}) + "\n")


def _resolve_manifest(manifest_path: Path | None) -> Path:
    """Return manifest path if given and exists, else default META/ct_processed_manifest.json, with fallback to manifests/."""
    if manifest_path and manifest_path.exists():
        return manifest_path
    default = META / "ct_processed_manifest.json"
    if default.exists():
        return default
    # Fallback: some setups write manifest under ct_brain/manifests/
    fallback = META.parent / "manifests" / "ct_processed_manifest.json"
    if fallback.exists():
        return fallback
    return default  # let open() raise FileNotFoundError with the canonical path 

class CTDataset(Dataset):
    """Dataset over cached CT npz from manifest; yields (C,H,W) tensor and 0/1 label."""
    def __init__(
        self, 
        manifest_path: Path | None = None, 
        split: Literal["train", "val", "test"] | None = None, 
        train_frac: float = 0.7, 
        val_frac: float = 0.15, 
        seed: int = 42, 
    ):
        manifest_path = _resolve_manifest(manifest_path)
        # #region agent log
        _agent_log("dataset_init_start", {"manifest_path": str(manifest_path), "manifest_exists": manifest_path.exists()}, "H4")
        # #endregion
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f) 
        valid = [e for e in entries if e.get("label", -1) in (0, 1)]
        if not valid:
            valid = entries

        # Design: fixed seed for reproducible train/val/test splits.
        rng = np.random.default_rng(seed)
        inx = rng.permutation(len(valid)) 
        n = len(valid) 
        nt = int(n * train_frac) 
        nv = int(n * val_frac) 
        train_idx, val_idx, test_idx = inx[:nt], inx[nt : nt + nv], inx[nt + nv :]
        split_map = {"train": set(train_idx), "val": set(val_idx), "test": set(test_idx)} 
        if split:
            self.entries = [valid[i] for i in range(n) if i in split_map[split]]
        else:
            self.entries = valid
        self.manifest_path = manifest_path
        # #region agent log
        _agent_log("dataset_init", {"manifest_path": str(manifest_path), "manifest_exists": manifest_path.exists(), "entries_total": len(entries), "valid_labeled": len(valid), "split": split or "all", "self_entries": len(self.entries)}, "H4")
        # #endregion 

    def __len__(self) -> int:
        """Return number of entries in this split."""
        return len(self.entries) 
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Demo: load cached NPZ "arr" (3,256,256); no DICOM access.
        e = self.entries[idx]
        # #region agent log
        if idx == 0:
            _agent_log("dataset_getitem_first", {"path": e.get("path"), "path_exists": Path(e["path"]).exists() if e.get("path") else False}, "H4")
        # #endregion
        arr = np.load(e["path"])["arr"]
        x = torch.from_numpy(arr).float()
        y = int(e.get("label", 0))
        # #region agent log
        if idx == 0:
            _agent_log("dataset_getitem_shape", {"arr_shape": list(arr.shape), "x_shape": list(x.shape)}, "H5")
        # #endregion
        return x, y 

    