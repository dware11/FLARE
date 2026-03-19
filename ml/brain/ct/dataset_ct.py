"""
Demo: Dataset reads cached NPZ only (not DICOM); manifest lists path + label.
"""
import json
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import META

DEFAULT_MANIFEST = META / "ct_processed_manifest.json"

DEBUG_LOG = ROOT / "debug_pipeline.log"

def _agent_log(msg, data=None, hyp=""):
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"message": msg, "data": data or {}, "hypothesisId": hyp, "location": "dataset_ct", "timestamp": int(time.time() * 1000)}) + "\n")


def _resolve_manifest(manifest_path: Optional[Path]) -> Path:
    # If caller didn't provide a manifest, use the default one written by preprocessing
    if manifest_path is None:
        return DEFAULT_MANIFEST
    return Path(manifest_path)

class CTDataset(Dataset):
    """Dataset over cached CT npz from manifest; yields (C,H,W) tensor and 0/1 label."""
    def __init__(
        self, 
        manifest_path: Optional[Path] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
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
        # Use all entries with existing NPZ path (label can be 0, 1, or -1; loss uses ignore_index=-1).
        valid = [e for e in entries if Path(e.get("path", "")).exists()]
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
        self.split = split or "all" 
        # #region agent log
        _agent_log("dataset_init", {"manifest_path": str(manifest_path), "manifest_exists": manifest_path.exists(), "entries_total": len(entries), "valid_labeled": len(valid), "split": split or "all", "self_entries": len(self.entries)}, "H4")
        # #endregion 
        # #region agent log
        try:
            import json as _json  # local import to avoid polluting module namespace
            from time import time as _time
            from pathlib import Path as _Path

            with open("debug-a04713.log", "a", encoding="utf-8") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "a04713",
                            "runId": "pre-fix",
                            "hypothesisId": "H1",
                            "location": "dataset_ct.py:CTDataset.__init__",
                            "message": "CTDataset init split sizes",
                            "data": {
                                "manifest_path": str(manifest_path),
                                "manifest_exists": _Path(manifest_path).exists(),
                                "entries_total": len(entries),
                                "split": split or "all",
                                "self_entries": len(self.entries),
                            },
                            "timestamp": int(_time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

    def __len__(self) -> int:
        """Return number of entries in this split."""
        return len(self.entries) 
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        # Demo: load cached NPZ "arr" (3,256,256); no DICOM access
        e = self.entries[idx]
        path = Path(e["path"])
        data = np.load(path)
        arr = data["arr"] 
        if arr.ndim == 3: 
            arr = arr[np.newaxis, ...] 
        x = torch.from_numpy(arr).float()
        # Apply ImageNet normalization so pretrained DenseNet121 features align.
        # x shape: (k, 3, H, W) — normalize each (3, H, W) slice independently.
        x = torch.stack([TF.normalize(x[s], _IMAGENET_MEAN, _IMAGENET_STD)
                         for s in range(x.shape[0])])
        y = int(e.get("label", -1))
        patient_id = e.get("patient_id", path.parent.name)
        if getattr(self, "split", "all") == "train":
            for s in range(x.shape[0]): 
                if torch.rand(1).item() < 0.5: 
                    scale = 0.9 + 0.2 * torch.rand(1).item() 
                    x[s] = (x[s] * scale).clamp(0.0, 1.0) 
                if torch.rand(1).item() < 0.5: 
                    max_shift = 8
                    dh = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                    dw = int(torch.randint(-max_shift, max_shift + 1, (1,)).item()) 
                    x[s] = torch.roll(x[s], shifts=(dh, dw), dims=(1,2))
                if torch.rand(1).item() < 0.5: 
                    x[s] = (x[s] + 0.01 * torch.randn_like(x[s])).clamp(0.0, 1.0) 
        return x, y, patient_id

    