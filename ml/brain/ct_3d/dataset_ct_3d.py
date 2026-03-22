"""
3D CT dataset using processed NPZ volumes from a manifest.

This version is for the ct_3d pipeline:
- reads from a separate 3D manifest
- returns tensors shaped (C, D, H, W)
- normalizes depth so batches can stack cleanly
- keeps label=-1 cases in the dataset, but training can ignore them
"""

import json
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.config import META

DEFAULT_MANIFEST = META / "ct_3d_processed_manifest.json"
DEBUG_LOG = ROOT / "debug_pipeline.log"


def _agent_log(msg, data=None, hyp=""):
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "message": msg,
                    "data": data or {},
                    "hypothesisId": hyp,
                    "location": "dataset_ct_3d",
                    "timestamp": int(time.time() * 1000),
                }
            )
            + "\n"
        )


def _resolve_manifest(manifest_path: Optional[Path]) -> Path:
    if manifest_path is None:
        return DEFAULT_MANIFEST
    return Path(manifest_path)


def _fix_depth(x: torch.Tensor, target_depth: int = 7) -> torch.Tensor:
    """
    x is (C, D, H, W).
    Pad or center-crop depth to target_depth so DataLoader can stack batches.
    """
    c, d, h, w = x.shape

    if d == target_depth:
        return x

    if d < target_depth:
        pad_total = target_depth - d
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front

        pad_front_tensor = torch.zeros((c, pad_front, h, w), dtype=x.dtype)
        pad_back_tensor = torch.zeros((c, pad_back, h, w), dtype=x.dtype)
        return torch.cat([pad_front_tensor, x, pad_back_tensor], dim=1)

    # d > target_depth -> center crop
    start = (d - target_depth) // 2
    end = start + target_depth
    return x[:, start:end, :, :]


class CTDataset(Dataset):
    """Dataset over cached 3D CT NPZ files; yields (C, D, H, W), label, patient_id."""

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        target_depth: int = 7,
    ):
        manifest_path = _resolve_manifest(manifest_path)
        self.target_depth = target_depth

        _agent_log(
            "dataset_init_start",
            {
                "manifest_path": str(manifest_path),
                "manifest_exists": manifest_path.exists(),
                "target_depth": target_depth,
            },
            "H4",
        )

        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)

        valid = [e for e in entries if Path(e.get("path", "")).exists()]
        if not valid:
            valid = entries

        rng = np.random.default_rng(seed)
        inx = rng.permutation(len(valid))
        n = len(valid)
        nt = int(n * train_frac)
        nv = int(n * val_frac)

        train_idx = inx[:nt]
        val_idx = inx[nt : nt + nv]
        test_idx = inx[nt + nv :]

        split_map = {
            "train": set(train_idx),
            "val": set(val_idx),
            "test": set(test_idx),
        }

        if split:
            self.entries = [valid[i] for i in range(n) if i in split_map[split]]
        else:
            self.entries = valid

        self.manifest_path = manifest_path
        self.split = split or "all"

        _agent_log(
            "dataset_init_done",
            {
                "manifest_path": str(manifest_path),
                "manifest_exists": manifest_path.exists(),
                "entries_total": len(entries),
                "entries_existing_path": len(valid),
                "split": self.split,
                "self_entries": len(self.entries),
                "target_depth": target_depth,
            },
            "H4",
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        e = self.entries[idx]
        path = Path(e["path"])
        data = np.load(path)
        arr = data["arr"]

        # Convert to (C, D, H, W)
        if arr.ndim == 3:
            # assume (D, H, W)
            arr = arr[np.newaxis, ...]  # -> (1, D, H, W)
        elif arr.ndim == 4:
            # could already be (C, D, H, W) or (D, C, H, W)
            if arr.shape[0] in (1, 3):
                # already channel-first
                pass
            elif arr.shape[1] in (1, 3):
                # likely depth-first -> move channel first
                arr = np.transpose(arr, (1, 0, 2, 3))
            else:
                raise ValueError(f"Unexpected 4D CT array shape {arr.shape} at {path}")
        else:
            raise ValueError(f"Unexpected CT array shape {arr.shape} at {path}")

        x = torch.from_numpy(arr).float()
   

        y = int(e.get("label", -1))
        patient_id = e.get("patient_id", path.parent.name)

        # Simple 3D augmentation for training
        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                scale = 0.9 + 0.2 * torch.rand(1).item()
                x = (x * scale).clamp(0.0, 1.0)

            if torch.rand(1).item() < 0.5:
                max_shift = 4
                dd = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                dh = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                dw = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                x = torch.roll(x, shifts=(dd, dh, dw), dims=(1, 2, 3))

            if torch.rand(1).item() < 0.5:
                x = (x + 0.01 * torch.randn_like(x)).clamp(0.0, 1.0)

        # Make channel count compatible with the current 3D model
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1, 1)

        # Normalize
        if x.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
        elif x.shape[0] == 1:
            mean = torch.tensor([0.5], dtype=torch.float32).view(1, 1, 1, 1)
            std = torch.tensor([0.5], dtype=torch.float32).view(1, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected channel count {x.shape[0]} for {path}")

        x = (x - mean) / std

        return x, y, patient_id




























