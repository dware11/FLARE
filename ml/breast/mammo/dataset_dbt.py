"""Dataset for DBT NPZ cache entries from JSON manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config_dbt import (
    DBT_MANIFEST_PATH,
    DBT_TARGET_SHAPE,
    DBT_TRAIN_FRACTION,
    DBT_VAL_FRACTION,
)


class DBTViewDataset(Dataset):
    """
    Yields:
      - image: FloatTensor (1, D, H, W)
      - label: int tensor (0/1)
      - exam_id: str
      - patient_id: str
      - view: str
    """

    def __init__(
        self,
        manifest_path: Path = DBT_MANIFEST_PATH,
        split: str = "train",
        train_frac: float = DBT_TRAIN_FRACTION,
        val_frac: float = DBT_VAL_FRACTION,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            self.entries: List[dict] = []
            return

        with self.manifest_path.open(encoding="utf-8") as f:
            entries = json.load(f)

        valid = []
        for e in entries:
            p = Path(e.get("path", "") or e.get("npz_path", ""))
            y = int(e.get("label", -1))
            if p.exists() and y in (0, 1):
                e["path"] = str(p)
                valid.append(e)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(valid))
        n = len(valid)
        nt = int(n * train_frac)
        nv = int(n * val_frac)
        split_map = {
            "train": set(idx[:nt]),
            "val": set(idx[nt : nt + nv]),
            "test": set(idx[nt + nv :]),
            "all": set(range(n)),
        }
        if split not in split_map:
            raise ValueError(f"Unknown split: '{split}'")

        wanted = split_map[split]
        self.entries = [valid[i] for i in range(n) if i in wanted]

    def __len__(self) -> int:
        return len(self.entries)

    def get_labels(self) -> List[int]:
        return [int(e.get("label", -1)) for e in self.entries]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str, str]:
        e = self.entries[idx]
        data = np.load(Path(e["path"]), allow_pickle=True)

        vol = data["image"].astype(np.float32)
        y = int(np.asarray(data["label"]).reshape(-1)[0])
        exam_id = str(np.asarray(data.get("exam_id", "")).reshape(-1)[0])
        patient_id = str(np.asarray(data.get("patient_id", "")).reshape(-1)[0])
        view = str(np.asarray(data.get("view", "")).reshape(-1)[0])

        c, d_t, h_t, w_t = DBT_TARGET_SHAPE
        if vol.shape != (c, d_t, h_t, w_t):
            raise ValueError(
                f"Unexpected DBT shape {vol.shape}, expected {(c, d_t, h_t, w_t)}"
            )

        x = torch.from_numpy(vol)
        label = torch.tensor(y, dtype=torch.long)
        return x, label, exam_id, patient_id, view
