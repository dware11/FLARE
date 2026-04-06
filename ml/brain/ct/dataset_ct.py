"""
Dataset: reads cached NPZ slices from manifest; applies train-time augmentations.
"""
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.config import META

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = META / "ct_processed_manifest.json"

# Match training / Wang-style thickness fusion: mm scaled to ~[0, 3] for stability.
_SLICE_THICKNESS_DIV_MM = 10.0
_SLICE_THICKNESS_CAP = 3.0


def _resolve_manifest(manifest_path: Optional[Path]) -> Path:
    if manifest_path is None:
        return DEFAULT_MANIFEST
    return Path(manifest_path)


def ct_batch_collate(batch: List[Tuple[torch.Tensor, int, str, torch.Tensor]]) -> Dict[str, Any]:
    """Stack batch into a dict for fusion-friendly training (patient_id always present)."""
    xs, ys, pids, ths = zip(*batch)
    return {
        "x": torch.stack(xs, dim=0),
        "y": torch.tensor(ys, dtype=torch.long),
        "patient_id": list(pids),
        "thickness": torch.stack(ths, dim=0),
    }


def unpack_ct_batch(
    batch: Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
    """Normalize DataLoader batch: dict (preferred) or legacy 4-tuple."""
    if isinstance(batch, dict):
        return batch["x"], batch["y"], batch["patient_id"], batch["thickness"]
    X, y, pid, thick = batch
    pids = [str(p) if not isinstance(p, str) else p for p in pid]
    return X, y, pids, thick


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
    x[:, top : top + rh, left : left + rw] = 0.0
    return x


def _filter_corrupt_npz_entries(entries: List[dict]) -> List[dict]:
    """Drop manifest rows whose NPZ cannot be read (logs warning)."""
    good: List[dict] = []
    for e in entries:
        p = Path(e.get("path", ""))
        if not p.is_file():
            continue
        try:
            with np.load(p) as data:
                _ = data["arr"]
        except Exception as ex:
            logger.warning("Skipping corrupt or unreadable NPZ %s: %s", p, ex)
            continue
        good.append(e)
    return good


class CTDataset(Dataset):
    """Dataset over cached CT npz from manifest; yields volume, label, id, and optional thickness."""

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        expected_k_slices: Optional[int] = None,
        skip_manifest_split: bool = False,
        manifest_split_filter: Optional[Literal["train", "val", "test"]] = None,
    ):
        manifest_path = _resolve_manifest(manifest_path)
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)
        valid = [e for e in entries if Path(e.get("path", "")).exists()]
        if not valid:
            valid = entries

        # Supervised train/val/test: only binary labels (skip manifest label -1).
        if split is not None:
            valid = [e for e in valid if int(e.get("label", -1)) in (0, 1)]

        if split is not None and len(valid) == 0:
            raise ValueError(
                f"No labeled samples (label 0/1) with existing NPZ paths in {manifest_path}. "
                "Check DATA_ROOT / meta path, --labels in preprocess_ct, and id alignment to reads."
            )

        rng = np.random.default_rng(seed)

        # CQ500 / combined manifest: subset by explicit "split" field or deterministic 70/15/15 on this file.
        if skip_manifest_split and manifest_split_filter is not None:
            labeled = [e for e in valid if e.get("split") == manifest_split_filter]
            if labeled:
                valid = labeled
            else:
                n = len(valid)
                if n == 0:
                    pass
                else:
                    inx = rng.permutation(n)
                    nt = int(n * train_frac)
                    nv = int(n * val_frac)
                    train_idx = set(inx[:nt])
                    val_idx = set(inx[nt : nt + nv])
                    test_idx = set(inx[nt + nv :])
                    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
                    idx_set = split_map[manifest_split_filter]
                    valid = [valid[i] for i in range(n) if i in idx_set]

        if split is not None and len(valid) == 0:
            raise ValueError(
                f"No samples left after manifest_split_filter={manifest_split_filter!r} in {manifest_path}."
            )

        if split is None:
            self.entries = valid
        elif skip_manifest_split:
            inx = rng.permutation(len(valid))
            self.entries = [valid[i] for i in inx]
        else:
            inx = rng.permutation(len(valid))
            n = len(valid)
            nt = int(n * train_frac)
            nv = int(n * val_frac)
            train_idx, val_idx, test_idx = inx[:nt], inx[nt : nt + nv], inx[nt + nv :]
            split_map = {"train": set(train_idx), "val": set(val_idx), "test": set(test_idx)}
            self.entries = [valid[i] for i in range(n) if i in split_map[split]]

        self.entries = _filter_corrupt_npz_entries(self.entries)
        if split is not None and len(self.entries) == 0:
            raise ValueError(
                f"No usable samples after filtering corrupt NPZ in {manifest_path}. "
                "Check cache paths and NPZ integrity."
            )

        self.manifest_path = manifest_path
        self.split = split or "all"
        self.expected_k_slices = expected_k_slices
        self._k_warned_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, torch.Tensor]:
        e = self.entries[idx]
        path = Path(e["path"])
        try:
            data = np.load(path)
            arr = data["arr"]
        except Exception as ex:
            logger.warning("NPZ read failed at %s: %s", path, ex)
            raise RuntimeError(f"Corrupt NPZ (should have been filtered at init): {path}") from ex

        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        if self.expected_k_slices is not None:
            k_got = int(arr.shape[0])
            if k_got != self.expected_k_slices:
                pkey = str(path.resolve())
                if pkey not in self._k_warned_paths:
                    self._k_warned_paths.add(pkey)
                    print(
                        f"WARNING: NPZ K-slices mismatch (expected {self.expected_k_slices}, got {k_got}) path={path}",
                        flush=True,
                    )
        x = torch.from_numpy(arr).float()
        y = int(e.get("label", -1))
        patient_id = e.get("patient_id", path.parent.name)

        if "slice_thickness_mm" in data.files:
            tmm = float(np.asarray(data["slice_thickness_mm"]).reshape(-1)[0])
        else:
            tmm = 0.0
        tnorm = min(max(tmm, 0.0) / _SLICE_THICKNESS_DIV_MM, _SLICE_THICKNESS_CAP)
        thickness = torch.tensor([tnorm], dtype=torch.float32)

        if self.split == "train":
            x = self._augment(x)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x, y, patient_id, thickness

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

        do_hflip = torch.rand(1).item() < 0.35
        do_vflip = torch.rand(1).item() < 0.08
        angle = (torch.rand(1).item() - 0.5) * 16.0 if torch.rand(1).item() < 0.45 else 0.0
        max_shift = 8
        dh = int(torch.randint(-max_shift, max_shift + 1, (1,)).item()) if torch.rand(1).item() < 0.35 else 0
        dw = int(torch.randint(-max_shift, max_shift + 1, (1,)).item()) if torch.rand(1).item() < 0.35 else 0
        do_mild_scale = torch.rand(1).item() < 0.35
        scale = 0.92 + 0.14 * torch.rand(1).item() if do_mild_scale else 1.0
        _, _, H, W = x.shape

        for s in range(k):
            sl = x[s]

            if do_hflip:
                sl = torch.flip(sl, dims=[2])
            if do_vflip:
                sl = torch.flip(sl, dims=[1])
            if angle != 0.0:
                sl = TF.rotate(sl, angle)
            if dh != 0 or dw != 0:
                sl = torch.roll(sl, shifts=(dh, dw), dims=(1, 2))
            if do_mild_scale and scale != 1.0:
                nh = max(1, int(round(H * scale)))
                nw = max(1, int(round(W * scale)))
                sl = TF.resize(sl, [nh, nw])
                sl = TF.center_crop(sl, [H, W])

            if torch.rand(1).item() < 0.5:
                scale_i = 0.8 + 0.4 * torch.rand(1).item()
                sl = (sl * scale_i).clamp(0.0, 1.0)
            if torch.rand(1).item() < 0.4:
                gamma = 0.7 + 0.6 * torch.rand(1).item()
                sl = sl.clamp(1e-8, 1.0).pow(gamma)
            if torch.rand(1).item() < 0.5:
                sl = (sl + 0.02 * torch.randn_like(sl)).clamp(0.0, 1.0)
            sl = _random_erasing(sl, p=0.25)

            x[s] = sl
        return x
