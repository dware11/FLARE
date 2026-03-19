"""
BCS-DBT dataset loader for 3D mammography training.

Expected manifest format (JSON list):
[
  {
    "patient_id": "DBT-P00001",
    "view": "LCC",               // LCC | RCC | LMLO | RMLO
    "path": "/path/to/cache/DBT-P00001_LCC.npz",
    "label": 0                   // 0=normal/benign, 1=cancer
  },
  ...
]

Each NPZ contains:
  "arr": float32 (32, 256, 256) preprocessed DBT slice stack

At inference, predictions for all 4 views of the same patient are averaged.
"""
import json
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.config import BREAST_MAMMO_MANIFEST


class MammoDataset(Dataset):
    """Dataset over cached DBT NPZ files from a BCS-DBT manifest.

    Yields: (tensor (1, 32, 256, 256), label int, patient_id str, view str)
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        path = Path(manifest_path) if manifest_path else BREAST_MAMMO_MANIFEST
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)

        valid = [e for e in entries if Path(e.get("path", "")).exists()]
        if not valid:
            valid = entries

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(valid))
        n = len(valid)
        nt = int(n * train_frac)
        nv = int(n * val_frac)
        split_map = {
            "train": set(idx[:nt]),
            "val":   set(idx[nt: nt + nv]),
            "test":  set(idx[nt + nv:]),
        }
        if split:
            self.entries = [valid[i] for i in range(n) if i in split_map[split]]
        else:
            self.entries = valid

        self.split = split or "all"
        self.augment = augment and (self.split == "train")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        e = self.entries[idx]
        data = np.load(e["path"])
        arr = data["arr"].astype(np.float32)   # (32, 256, 256)
        x = torch.from_numpy(arr).unsqueeze(0)  # (1, 32, 256, 256)

        if self.augment:
            x = self._augment(x)

        label = int(e.get("label", -1))
        return x, label, e.get("patient_id", ""), e.get("view", "")

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        """Lightweight 3D augmentation: LR flip, intensity jitter."""
        if torch.rand(1).item() < 0.5:
            x = x.flip(dims=[3])                  # flip W (left-right)
        if torch.rand(1).item() < 0.5:
            scale = 0.9 + 0.2 * torch.rand(1).item()
            x = (x * scale).clamp(0.0, 1.0)
        if torch.rand(1).item() < 0.3:
            x = (x + 0.01 * torch.randn_like(x)).clamp(0.0, 1.0)
        return x
