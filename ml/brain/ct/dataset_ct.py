"""
Dataset: reads cached NPZ slices from manifest; applies train-time augmentations.
"""
import json
import math
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import META

DEFAULT_MANIFEST = META / "ct_processed_manifest.json"


def _resolve_manifest(manifest_path: Optional[Path]) -> Path:
    if manifest_path is None:
        return DEFAULT_MANIFEST
    return Path(manifest_path)


def _random_erasing(x: torch.Tensor, p: float = 0.3, scale_lo: float = 0.02, scale_hi: float = 0.15) -> torch.Tensor:
    """Zero-out a random rectangular patch (cutout-style) on a (C, H, W) tensor."""
    if torch.rand(1).item() > p:
        return x
    C, H, W = x.shape
    area = H * W
    target_area = area * (scale_lo + (scale_hi - scale_lo) * torch.rand(1).item())
    aspect = 0.3 + 2.4 * torch.rand(1).item()
    rh = int(round(math.sqrt(target_area * aspect)))
    rw = int(round(math.sqrt(target_area / aspect)))
    rh, rw = min(rh, H), min(rw, W)
    top = torch.randint(0, H - rh + 1, (1,)).item()
    left = torch.randint(0, W - rw + 1, (1,)).item()
    x[:, top:top + rh, left:left + rw] = 0.0
    return x


class CTDataset(Dataset):
    """Dataset over cached CT npz from manifest; yields (k, C, H, W) tensor and 0/1 label."""
    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        manifest_path = _resolve_manifest(manifest_path)
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
        train_idx, val_idx, test_idx = inx[:nt], inx[nt : nt + nv], inx[nt + nv :]
        split_map = {"train": set(train_idx), "val": set(val_idx), "test": set(test_idx)}
        if split:
            self.entries = [valid[i] for i in range(n) if i in split_map[split]]
        else:
            self.entries = valid
        self.manifest_path = manifest_path
        self.split = split or "all"

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        e = self.entries[idx]
        path = Path(e["path"])
        data = np.load(path)
        arr = data["arr"]
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        x = torch.from_numpy(arr).float()
        y = int(e.get("label", -1))
        patient_id = e.get("patient_id", path.parent.name)

        if self.split == "train":
            x = self._augment(x)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x, y, patient_id

    # ------------------------------------------------------------------
    # Augmentation pipeline (train only)
    # ------------------------------------------------------------------
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations consistently across all k slices in a patient volume.

        Spatial transforms (flip, rotation, translation) use the SAME random
        params across slices so anatomy stays aligned.  Intensity transforms
        (brightness, contrast, noise, erasing) are per-slice for variety.
        """
        k = x.shape[0]

        # --- shared spatial params ---
        do_hflip = torch.rand(1).item() < 0.5
        do_vflip = torch.rand(1).item() < 0.1
        angle = (torch.rand(1).item() - 0.5) * 30.0 if torch.rand(1).item() < 0.5 else 0.0
        max_shift = 10
        dh = int(torch.randint(-max_shift, max_shift + 1, (1,)).item()) if torch.rand(1).item() < 0.4 else 0
        dw = int(torch.randint(-max_shift, max_shift + 1, (1,)).item()) if torch.rand(1).item() < 0.4 else 0

        for s in range(k):
            sl = x[s]  # (C, H, W)

            # -- spatial (shared) --
            if do_hflip:
                sl = torch.flip(sl, dims=[2])
            if do_vflip:
                sl = torch.flip(sl, dims=[1])
            if angle != 0.0:
                sl = TF.rotate(sl, angle)
            if dh != 0 or dw != 0:
                sl = torch.roll(sl, shifts=(dh, dw), dims=(1, 2))

            # -- intensity (per-slice) --
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1).item()
                sl = (sl * scale).clamp(0.0, 1.0)
            if torch.rand(1).item() < 0.4:
                gamma = 0.7 + 0.6 * torch.rand(1).item()
                sl = sl.clamp(1e-8, 1.0).pow(gamma)
            if torch.rand(1).item() < 0.5:
                sl = (sl + 0.02 * torch.randn_like(sl)).clamp(0.0, 1.0)
            sl = _random_erasing(sl, p=0.25)

            x[s] = sl
        return x
