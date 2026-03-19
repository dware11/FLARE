"""
Preprocessing transforms for DBT volumes.

Pipeline per DICOM stack (one view):
  1. Load DICOM series → sort by InstanceNumber → (D, H, W) uint16
  2. Normalize to [0, 1] (12–16 bit pixel values)
  3. Breast ROI crop via Otsu threshold (removes black background)
  4. CLAHE per slice (clip_limit=2.0, tile_grid=8×8) — requires opencv-python
  5. Trilinear resample → (D_out, H_out, W_out)
  6. Return float32 numpy array

No channel replication — DBT is single channel throughout.
"""
from __future__ import annotations

import numpy as np


def normalize_uint(arr: np.ndarray) -> np.ndarray:
    """Normalize integer pixel array to float32 [0, 1]."""
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def otsu_crop(arr: np.ndarray) -> np.ndarray:
    """Crop breast ROI using Otsu threshold on the MIP (max intensity projection).

    Args:
        arr: float32 (D, H, W) in [0, 1]
    Returns:
        Cropped float32 array — may have different H, W but same D.
    """
    mip = arr.max(axis=0)  # (H, W)
    thresh = _otsu_threshold(mip)
    mask = mip > thresh
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return arr[:, r0:r1, c0:c1]


def _otsu_threshold(img: np.ndarray, n_bins: int = 256) -> float:
    """Compute Otsu threshold without scipy/sklearn dependency."""
    hist, bin_edges = np.histogram(img, bins=n_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight_b = np.cumsum(hist) / total
    weight_f = 1.0 - weight_b
    mean_b = np.cumsum(hist * bin_centers) / (np.cumsum(hist) + 1e-12)
    mean_total = (hist * bin_centers).sum() / total
    mean_f = np.where(weight_f > 1e-12,
                      (mean_total - weight_b * mean_b) / (weight_f + 1e-12),
                      mean_b)
    variance = weight_b * weight_f * (mean_b - mean_f) ** 2
    return float(bin_centers[np.argmax(variance)])


def clahe_volume(arr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE to each slice of a (D, H, W) float32 [0,1] volume.

    Falls back to identity if opencv-python is not installed.
    """
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                 tileGridSize=(tile_grid, tile_grid))
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            sl_uint8 = (arr[i] * 255).astype(np.uint8)
            out[i] = clahe.apply(sl_uint8).astype(np.float32) / 255.0
        return out
    except ImportError:
        return arr


def trilinear_resample(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resample (D, H, W) float32 array to target_shape using trilinear interpolation.

    Uses torch.nn.functional.interpolate under the hood — avoids scipy dependency.
    """
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def preprocess_dbt_volume(
    arr: np.ndarray,
    target_shape: tuple[int, int, int] = (32, 256, 256),
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> np.ndarray:
    """Full preprocessing pipeline for a raw DBT volume.

    Args:
        arr: (D, H, W) integer or float pixel array (raw DICOM pixel values)
        target_shape: output (D, H, W)
        clip_limit: CLAHE clip limit
        tile_grid: CLAHE tile grid size

    Returns:
        float32 (D, H, W) in [0, 1], resampled to target_shape
    """
    arr = normalize_uint(arr)
    arr = otsu_crop(arr)
    arr = clahe_volume(arr, clip_limit=clip_limit, tile_grid=tile_grid)
    arr = trilinear_resample(arr, target_shape)
    return arr.astype(np.float32)
