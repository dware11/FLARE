"""
EA1141 paired loader — returns (mammo_path, mri_path, label) per patient.

Both modalities must exist in their respective cache directories. Patients
missing either modality are excluded so that fusion metrics are computed
on the same cohort.

Manifest format (JSON list, one entry per patient):
[
  {
    "patient_id": "EA1141-001",
    "mammo_path": "/path/to/breast_mammo/cache/EA1141-001_LCC.npz",
    "mri_path":   "/path/to/breast_mri/cache/EA1141-001_mri.npz",
    "label": 1
  },
  ...
]

Usage:
    from ml.breast.fusion.dataset_paired import PairedDataset
    ds = PairedDataset(manifest_path="...")
    mammo_npz, mri_npz, label, pid = ds[0]
"""
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    """Paired mammography + MRI dataset for EA1141 fusion evaluation."""

    def __init__(self, manifest_path: str | Path) -> None:
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)

        self.entries = [
            e for e in entries
            if Path(e.get("mammo_path", "")).exists()
            and Path(e.get("mri_path", "")).exists()
        ]
        if not self.entries:
            raise FileNotFoundError(
                f"No paired entries found in {manifest_path}. "
                "Run preprocess_mammo.py and preprocess_breast_mri.py first."
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        e = self.entries[idx]
        mammo = torch.from_numpy(np.load(e["mammo_path"])["arr"].astype(np.float32))
        mri   = torch.from_numpy(np.load(e["mri_path"])["arr"].astype(np.float32))
        label = int(e.get("label", -1))
        pid   = e.get("patient_id", "")
        return mammo, mri, label, pid
