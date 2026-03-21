"""Strict binary MRI labels: 1 / 0 / -1 (ignore). Tune sets to your EA1141 CSV."""

from __future__ import annotations

import re
from typing import Any, Mapping, Tuple


def _canon(s: str) -> str:
    s = (s or "").strip().lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).replace(" ", "_")
    return s


EA1141_POSITIVE = {
    "malignant",
    "malignat",
    "cancer",
    "biopsy_proven_cancer",
    "invasive",
    "dcis",
    "positive",
    "postitive",
}
EA1141_NEGATIVE = {"benign", "normal", "negative", "biopsy_proven_benign", "no_cancer"}
EA1141_IGNORE = {
    "suspicious",
    "actionable",
    "unknown",
    "unknwon",
    "indeterminate",
    "inteterminate",
    "pending",
}

_LABEL_KEYS = (
    "raw_label",
    "label",
    "pathology",
    "diagnosis",
    "diagonsis",
    "final_diagnosis",
    "final_dignosis",
    "clinical_label",
)


def map_ea1141_label(raw_label: str) -> int:
    n = _canon(raw_label)
    if not n:
        return -1
    if n in EA1141_POSITIVE:
        return 1
    if n in EA1141_NEGATIVE:
        return 0
    if n in EA1141_IGNORE:
        return -1
    if any(x in n for x in ("malignant", "cancer", "carcinoma")):
        if "benign" in n or "bengin" in n or "normal" in n:
            return -1
        return 1
    if any(x in n for x in ("benign", "normal", "negative")):
        return 0
    return -1


def map_mri_label_from_row(row: Mapping[str, Any]) -> Tuple[int, str]:
    raw = ""
    for k in _LABEL_KEYS:
        if k in row and row[k] is not None:
            raw = str(row[k]).strip()
            break
    if not raw:
        low = {str(k).lower(): k for k in row}
        for k in _LABEL_KEYS:
            if k in low and row[low[k]] is not None:
                raw = str(row[low[k]]).strip()
                break
    return map_ea1141_label(raw), raw


def is_valid_training_label(y: int) -> bool:
    return y in (0, 1)
