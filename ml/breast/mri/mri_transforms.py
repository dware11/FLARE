"""Pure numpy / scipy / pydicom volume helpers for breast DCE-MRI preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pydicom

try:
    from scipy import ndimage
except ImportError:
    ndimage = None  # type: ignore


def percentile_normalize(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(a, (lo_pct, hi_pct))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    a = np.clip(a, lo, hi)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def load_dce_series(dicom_dir: Path) -> np.ndarray:
    """Load one DICOM series from a directory into (D, H, W) float32."""
    files = sorted(dicom_dir.glob("*.dcm")) + sorted(dicom_dir.glob("*.DCM"))
    if not files:
        raise FileNotFoundError(f"No DICOM in {dicom_dir}")
    slices: list[tuple[int, np.ndarray]] = []
    for p in files:
        ds = pydicom.dcmread(p)
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
        img = ds.pixel_array.astype(np.float32)
        slices.append((inst, img))
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=0)


def resample_volume_3d(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Trilinear resize (D, H, W) -> target_shape."""
    if ndimage is None:
        raise ImportError("scipy is required for resample_volume_3d")
    d, h, w = arr.shape
    td, th, tw = target_shape
    zoom = (td / d, th / h, tw / w)
    return ndimage.zoom(arr.astype(np.float32), zoom, order=1).astype(np.float32)


def _align_to_reference(moving: np.ndarray, ref_shape: Tuple[int, int, int]) -> np.ndarray:
    if moving.shape == ref_shape:
        return moving.astype(np.float32)
    return resample_volume_3d(moving, ref_shape)


def compute_foreground_bbox(
    arr_norm_01: np.ndarray,
    threshold: float,
    padding_frac: float,
    min_foreground_fraction: float = 0.02,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Binary mask arr > threshold -> padded bbox (z0, z1, y0, y1, x0, x1) half-open intervals.
    """
    m = arr_norm_01 > threshold
    if m.mean() < min_foreground_fraction:
        return None
    coords = np.argwhere(m)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    dz, dy, dx = arr_norm_01.shape
    pz = int(round(padding_frac * dz))
    py = int(round(padding_frac * dy))
    px = int(round(padding_frac * dx))
    z0, y0, x0 = max(0, z0 - pz), max(0, y0 - py), max(0, x0 - px)
    z1, y1, x1 = min(dz, z1 + pz), min(dy, y1 + py), min(dx, x1 + px)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return None
    return (z0, z1, y0, y1, x0, x1)


def crop_volume(arr: np.ndarray, bbox: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    return arr[z0:z1, y0:y1, x0:x1].astype(np.float32)


def safe_center_crop(arr: np.ndarray, frac: float = 0.75) -> np.ndarray:
    d, h, w = arr.shape

    def span(n: int) -> Tuple[int, int]:
        m = max(1, int(round(n * frac)))
        s = (n - m) // 2
        return s, s + m

    z0, z1 = span(d)
    y0, y1 = span(h)
    x0, x1 = span(w)
    return arr[z0:z1, y0:y1, x0:x1].astype(np.float32)


def preprocess_dce_volume(
    pre_arr: np.ndarray,
    post1_arr: np.ndarray,
    post2_arr: np.ndarray,
    target_shape: Tuple[int, int, int] = (32, 128, 128),
    add_subtraction: bool = False,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    roi_threshold: float = 0.08,
    roi_padding_frac: float = 0.08,
    roi_reference: str = "post1",
    center_crop_frac: float = 0.75,
    roi_min_frac: float = 0.02,
) -> Tuple[np.ndarray, str, str]:
    """
    Returns:
        vol (C, D, H, W), crop_bbox_str, roi_method ('threshold_bbox' | 'center_fallback')
    """
    ref_shape = post1_arr.shape
    pre_a = _align_to_reference(pre_arr, ref_shape)
    p1_a = post1_arr.astype(np.float32)
    p2_a = _align_to_reference(post2_arr, ref_shape)

    pre_n = percentile_normalize(pre_a, lo_pct, hi_pct)
    p1_n = percentile_normalize(p1_a, lo_pct, hi_pct)
    p2_n = percentile_normalize(p2_a, lo_pct, hi_pct)

    if roi_reference == "channel_max":
        ref = np.maximum(np.maximum(pre_n, p1_n), p2_n)
    else:
        ref = p1_n

    bbox = compute_foreground_bbox(ref, roi_threshold, roi_padding_frac, roi_min_frac)
    roi_method = "threshold_bbox"
    if bbox is None:
        roi_method = "center_fallback"
        pre_c = safe_center_crop(pre_n, center_crop_frac)
        p1_c = safe_center_crop(p1_n, center_crop_frac)
        p2_c = safe_center_crop(p2_n, center_crop_frac)
    else:
        pre_c = crop_volume(pre_n, bbox)
        p1_c = crop_volume(p1_n, bbox)
        p2_c = crop_volume(p2_n, bbox)

    pre_r = resample_volume_3d(pre_c, target_shape)
    p1_r = resample_volume_3d(p1_c, target_shape)
    p2_r = resample_volume_3d(p2_c, target_shape)

    chans = [pre_r, p1_r, p2_r]
    if add_subtraction:
        sub = p1_r - pre_r
        sub = np.clip(sub, 0.0, 1.0).astype(np.float32)
        chans.append(sub)
    vol = np.stack(chans, axis=0)
    bbox_str = "none" if bbox is None else "|".join(str(x) for x in bbox)
    return vol.astype(np.float32), bbox_str, roi_method
