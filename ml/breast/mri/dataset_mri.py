"""PyTorch Dataset reading breast MRI NPZ cache via CSV manifest."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config_mri import BREAST_MRI_MANIFEST_PATH, MRI_DEFAULT_SEED, MRI_DEFAULT_VAL_FRAC


def _load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _patient_train_val_split(
    entries: List[Dict[str, str]], val_frac: float, seed: int
) -> Tuple[set, set]:
    patients = sorted({e["patient_id"] for e in entries})
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_frac))) if len(patients) > 1 else 0
    val_set = set(patients[:n_val])
    train_set = set(patients[n_val:])
    return train_set, val_set


def _decode_np_scalar(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if hasattr(val, "item"):
        v = val.item()
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return str(v)
    return str(val)


class BreastMRIDataset(Dataset):
    """
    Manifest CSV is source of truth. Skips label -1 and non-success rows.
    """

    def __init__(
        self,
        manifest_path: Path = BREAST_MRI_MANIFEST_PATH,
        split: str = "train",
        val_frac: float = MRI_DEFAULT_VAL_FRAC,
        seed: int = MRI_DEFAULT_SEED,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.augment = augment and split == "train"
        raw = _load_manifest_rows(self.manifest_path)
        valid: List[Dict[str, str]] = []
        for e in raw:
            if e.get("status") != "success":
                continue
            p = e.get("npz_path", "")
            if not p or not Path(p).is_file():
                continue
            try:
                y = int(float(e.get("label", -1)))
            except ValueError:
                continue
            if y not in (0, 1):
                continue
            valid.append(e)

        tr_pat, va_pat = _patient_train_val_split(valid, val_frac, seed)
        if split == "train":
            self.entries = [e for e in valid if e["patient_id"] in tr_pat]
        elif split == "val":
            self.entries = [e for e in valid if e["patient_id"] in va_pat]
        elif split == "all":
            self.entries = valid
        else:
            raise ValueError("split must be train|val|all")

    def __len__(self) -> int:
        return len(self.entries)

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-2])
        if torch.rand(1).item() < 0.3:
            x = x + torch.randn_like(x) * 0.02
        if torch.rand(1).item() < 0.3:
            s = 0.9 + 0.2 * torch.rand(1).item()
            x = torch.clamp(x * s, 0.0, 1.0)
        return x

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        data = np.load(e["npz_path"], allow_pickle=True)
        img = torch.from_numpy(data["image"].astype(np.float32))
        y = int(np.asarray(data["label"]).reshape(-1)[0])
        exam_id = _decode_np_scalar(data["exam_id"])
        patient_id = _decode_np_scalar(data["patient_id"])
        pn = data.get("phase_names", None)
        if pn is not None:
            # Avoid numpy unicode arrays in meta — default_collate cannot batch them.
            pn = np.asarray(pn, dtype=str).tolist()
        meta: Dict[str, Any] = {
            "npz_path": e["npz_path"],
            "raw_label": _decode_np_scalar(data.get("raw_label", "")),
            "phase_names": pn,
        }
        if self.augment:
            img = self._aug(img)
        return img, torch.tensor(y, dtype=torch.float32), exam_id, patient_id, meta
