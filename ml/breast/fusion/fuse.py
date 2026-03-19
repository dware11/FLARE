"""
Late fusion strategies for mammography (DBT) + MRI predictions.

All strategies take P_mammo ∈ [0,1] and P_mri ∈ [0,1] and return P_fused ∈ [0,1].

Three strategies:
  1. weighted_avg  — α × P_mammo + (1-α) × P_mri  (no training needed)
  2. learned_linear — sigmoid(w1·P_mammo + w2·P_mri + b)  (fit on val labels)
  3. confidence_weighted — weight by per-sample confidence (|p - 0.5|)

Usage:
    from ml.breast.fusion.fuse import weighted_avg, learned_linear, confidence_weighted

    p_fused = weighted_avg(p_mammo, p_mri, alpha=0.4)
    p_fused = confidence_weighted(p_mammo, p_mri)
"""
from __future__ import annotations

import math
import numpy as np


def weighted_avg(
    p_mammo: float | np.ndarray,
    p_mri:   float | np.ndarray,
    alpha:   float = 0.5,
) -> float | np.ndarray:
    """Weighted average fusion.

    Args:
        p_mammo: mammography cancer probability [0,1]
        p_mri:   MRI cancer probability [0,1]
        alpha:   weight for mammography (1-alpha for MRI); default 0.5
    """
    return alpha * np.asarray(p_mammo) + (1.0 - alpha) * np.asarray(p_mri)


def confidence_weighted(
    p_mammo: float | np.ndarray,
    p_mri:   float | np.ndarray,
) -> float | np.ndarray:
    """Confidence-weighted fusion: weight by certainty (distance from 0.5).

    Higher-confidence modality (farther from 0.5) gets more weight.
    """
    p_mammo = np.asarray(p_mammo, dtype=float)
    p_mri   = np.asarray(p_mri,   dtype=float)
    conf_mammo = np.abs(p_mammo - 0.5)
    conf_mri   = np.abs(p_mri   - 0.5)
    total = conf_mammo + conf_mri + 1e-8
    return (conf_mammo * p_mammo + conf_mri * p_mri) / total


class LearnedLinearFusion:
    """Logistic regression fusion: sigmoid(w1·P_mammo + w2·P_mri + b).

    Fit on validation labels using gradient descent (no sklearn required).
    Learned weights are stored as plain floats for easy serialization.
    """

    def __init__(self) -> None:
        self.w1: float = 1.0
        self.w2: float = 1.0
        self.b:  float = 0.0

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(
        self,
        p_mammo: np.ndarray,
        p_mri:   np.ndarray,
        labels:  np.ndarray,
        lr: float = 0.1,
        epochs: int = 500,
    ) -> None:
        """Fit w1, w2, b via binary cross-entropy gradient descent."""
        p_mammo = np.asarray(p_mammo, dtype=float)
        p_mri   = np.asarray(p_mri,   dtype=float)
        labels  = np.asarray(labels,  dtype=float)
        mask = labels >= 0
        p_mammo, p_mri, labels = p_mammo[mask], p_mri[mask], labels[mask]

        w1, w2, b = self.w1, self.w2, self.b
        for _ in range(epochs):
            logit = w1 * p_mammo + w2 * p_mri + b
            pred  = self._sigmoid(logit)
            err   = pred - labels
            grad_w1 = (err * p_mammo).mean()
            grad_w2 = (err * p_mri).mean()
            grad_b  = err.mean()
            w1 -= lr * grad_w1
            w2 -= lr * grad_w2
            b  -= lr * grad_b

        self.w1, self.w2, self.b = float(w1), float(w2), float(b)

    def predict(
        self,
        p_mammo: float | np.ndarray,
        p_mri:   float | np.ndarray,
    ) -> float | np.ndarray:
        logit = self.w1 * np.asarray(p_mammo) + self.w2 * np.asarray(p_mri) + self.b
        return self._sigmoid(logit)

    def __repr__(self) -> str:
        return f"LearnedLinearFusion(w1={self.w1:.4f}, w2={self.w2:.4f}, b={self.b:.4f})"
