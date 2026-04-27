"""
5-class brain MRI + CT late fusion (Huda known-good weighting path).

- MRI abnormal score  =  P(glioma) + P(meningioma) + P(pituitary)  (softmax outputs).
- Do NOT use 1 − max class confidence; benign tumor subtypes are still "abnormal" for triage.
- pred_label in the MRI result dict is left to predict_mri for clinician display; fusion stays binary
  (abnormal vs normal) from fusion_score.

When no 5-class ``probabilities`` list is available (e.g. BraTS NPZ path), we derive a binary
tumor-laden score from ``segmentation.tumor_voxels_total`` only.
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

# Must match training / eval: eval_fusion T2 5-class split
MRI_NORMAL_CLASSES = frozenset({"normal", "no_tumor"})
MRI_TUMOR_CLASSES = frozenset({"glioma", "meningioma", "pituitary"})

CT_WEIGHT = 0.4
MRI_WEIGHT = 0.6
FUSION_THRESHOLD = 0.5


def _norm_label(label: str) -> str:
    return (label or "").strip().lower()


def mri_abnormal_probability(mri_result: MutableMapping[str, Any] | Mapping[str, Any] | None) -> float | None:
    """
    Return probability mass on tumor subtypes: sum of softmax probs for the three tumor classes.
    """
    if not mri_result or not isinstance(mri_result, dict):
        return None
    if mri_result.get("error"):
        return None

    probs = mri_result.get("probabilities")
    if isinstance(probs, list) and len(probs) > 0:
        total = 0.0
        saw_known = False
        for item in probs:
            if not isinstance(item, dict):
                continue
            lab = _norm_label(str(item.get("label", "")))
            if lab in MRI_TUMOR_CLASSES | MRI_NORMAL_CLASSES:
                saw_known = True
            if lab in MRI_TUMOR_CLASSES:
                try:
                    total += float(item.get("value", 0.0))
                except (TypeError, ValueError):
                    continue
        if saw_known:
            return max(0.0, min(1.0, float(total)))

    # BraTS: no 5-class vector — use volume presence
    seg = mri_result.get("segmentation")
    if isinstance(seg, dict) and "tumor_voxels_total" in seg:
        try:
            tv = int(seg.get("tumor_voxels_total") or 0)
        except (TypeError, ValueError):
            return None
        return 1.0 if tv > 0 else 0.0

    pl = _norm_label(str(mri_result.get("pred_label", "")))
    if pl == "no_detected_tumor":
        return 0.0
    if pl in MRI_NORMAL_CLASSES:
        return 0.0
    if pl in MRI_TUMOR_CLASSES:
        # 5-class path should have listed probabilities; avoid inventing 1.0
        return None

    return None
