"""
tumor_aware_crop.py
====================
Crop a raw BraTS volume down to exactly 128 x 128 x 128
while keeping the entire tumor + surrounding brain context.

Two paths, chosen automatically per-case:

  Path A  —  "clean ROI"  (most cases)
      The tumor bounding box + margin fits inside 128^3.
      We place a 128^3 window centred on the tumor centroid,
      clamp it to the volume edges, and zero-pad any side that
      overflows.  Result: ZERO information loss, no interpolation.

  Path B  —  "resize fallback"  (giant tumors only)
      The tumor + margin exceeds 128 on at least one axis.
      We cut a tight ROI around the tumor + margin (which may be
      larger than 128 on some axes), then zoom it down to 128^3.
      Images  ->  trilinear   (order=1).
      Seg     ->  nearest-nbr (order=0) so labels stay as clean
                  integers  --  no blurring, no new label values.

Why two paths and not always zoom?
  Zooming (even trilinear) slightly blurs spatial detail.  For the
  majority of cases where the tumor fits comfortably, there is no
  reason to pay that cost.  Path A is lossless.  Path B only kicks
  in when the volume is physically too big to fit without it.

Margin logic
  margin = clamp( max(20, 10 % of bbox extent per axis), 15, 30 )
  Small tumors  ->  20 voxels of healthy-brain context each side.
  Giant tumors  ->  capped at 30 so the ROI does not balloon.

Usage
  from tumor_aware_crop import tumor_aware_crop

  image_out, seg_out, info = tumor_aware_crop(image, seg)
      image : (4, H, W, D)  float32   already normalised
      seg   : (H, W, D)     int64
      image_out : (4, 128, 128, 128)  float32
      seg_out   : (128, 128, 128)     int64   labels still {0,1,2,3}
      info      : dict  (path taken, margins, etc.)
"""

import numpy as np
from scipy import ndimage

TARGET = np.array([128, 128, 128])


# ─────────────────────────────────────────────────────────────────────
# public entry point
# ─────────────────────────────────────────────────────────────────────

def tumor_aware_crop(image: np.ndarray, seg: np.ndarray):
    """
    Main entry point.  Dispatches to Path A or Path B automatically.

    Returns
    -------
    image_out : (4, 128, 128, 128) float32
    seg_out   : (128, 128, 128)    int64    labels {0,1,2,3}
    info      : dict with diagnostics
    """
    vol_shape = np.array(image.shape[1:])   # (H, W, D)

    # 1. tumor bounding box
    bbox_min, bbox_max = _tumor_bbox(seg)

    # 2. margin per axis
    bbox_extent = bbox_max - bbox_min + 1          # inclusive size
    margin      = _compute_margin(bbox_extent)

    # 3. ROI = bbox + margin on every side
    roi_min = bbox_min - margin
    roi_max = bbox_max + margin + 1                # exclusive end

    roi_extent = roi_max - roi_min

    # 4. pick path
    if np.all(roi_extent <= TARGET):
        image_out, seg_out = _path_a(image, seg, roi_min, roi_max, vol_shape)
        path_label = 'A_clean_ROI'
    else:
        image_out, seg_out = _path_b(image, seg, roi_min, roi_max, vol_shape)
        path_label = 'B_resize_fallback'

    info = {
        'path':             path_label,
        'tumor_extents':    tuple(bbox_extent.tolist()),
        'margin':           tuple(margin.tolist()),
        'roi_extent':       tuple(roi_extent.tolist()),
        'labels':           np.unique(seg_out).tolist(),
        'tumor_preserved':  int((seg_out > 0).sum()),
    }

    return image_out.astype(np.float32), seg_out.astype(np.int64), info


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

def _tumor_bbox(seg):
    """
    (min_coords, max_coords) of the tumor bounding box.
    Falls back to volume centre if no tumor voxels exist.
    """
    coords = np.argwhere(seg > 0)
    if len(coords) == 0:
        center = np.array(seg.shape) // 2
        return center, center
    return coords.min(axis=0), coords.max(axis=0)


def _compute_margin(bbox_extent):
    """
    Per-axis margin.
      base  = max(20, round(10 % of extent))
      final = clamp(base, 15, 30)
    """
    base   = np.maximum(20, np.round(0.10 * bbox_extent).astype(int))
    return np.clip(base, 15, 30)


# ─────────────────────────────────────────────────────────────────────
# Path A  —  crop + pad   (lossless)
# ─────────────────────────────────────────────────────────────────────

def _path_a(image, seg, roi_min, roi_max, vol_shape):
    """
    Centre a 128^3 window on the tumor centroid.
    Clamp to volume edges.  Zero-pad any overflow side.

    Why centre on centroid, not ROI centre?
      The ROI wraps the tumor tightly.  Centring the larger 128^3 window
      on the centroid distributes the leftover space evenly, keeping the
      healthy-brain context balanced on all sides.
    """
    centroid = (roi_min + roi_max) // 2

    half    = TARGET // 2
    win_min = centroid - half
    win_max = win_min + TARGET

    # clamp
    win_min_c = np.maximum(win_min, 0)
    win_max_c = np.minimum(win_max, vol_shape)

    # padding needed
    pad_before = win_min_c - win_min                                    # >= 0
    pad_after  = TARGET - (win_max_c - win_min_c) - pad_before         # >= 0

    # slice
    s_img = (slice(None),) + tuple(slice(int(win_min_c[i]), int(win_max_c[i])) for i in range(3))
    s_seg =                  tuple(slice(int(win_min_c[i]), int(win_max_c[i])) for i in range(3))

    img_crop = image[s_img]
    seg_crop = seg[s_seg]

    # pad
    p_img = [(0, 0)] + [(int(pad_before[i]), int(pad_after[i])) for i in range(3)]
    p_seg =            [(int(pad_before[i]), int(pad_after[i])) for i in range(3)]

    return (np.pad(img_crop, p_img, mode='constant', constant_values=0),
            np.pad(seg_crop, p_seg, mode='constant', constant_values=0))


# ─────────────────────────────────────────────────────────────────────
# Path B  —  crop ROI then zoom to 128^3
# ─────────────────────────────────────────────────────────────────────

def _path_b(image, seg, roi_min, roi_max, vol_shape):
    """
    Cut the tightest ROI that contains bbox + margin, then zoom
    down to 128^3.

    image  ->  trilinear  (order=1)          good for intensities
    seg    ->  nearest-nbr (order=0)         labels stay clean integers
    """
    # clamp ROI to volume
    roi_min_c = np.maximum(roi_min, 0)
    roi_max_c = np.minimum(roi_max, vol_shape)

    # slice out
    s_img = (slice(None),) + tuple(slice(int(roi_min_c[i]), int(roi_max_c[i])) for i in range(3))
    s_seg =                  tuple(slice(int(roi_min_c[i]), int(roi_max_c[i])) for i in range(3))

    img_roi = image[s_img]
    seg_roi = seg[s_seg]

    roi_shape    = np.array(img_roi.shape[1:])
    zoom_factors = TARGET / roi_shape

    # zoom image channel by channel
    img_out = np.stack([
        ndimage.zoom(img_roi[c], zoom_factors, order=1, mode='constant', cval=0)
        for c in range(img_roi.shape[0])
    ], axis=0)

    # zoom seg  —  nearest neighbour keeps labels clean
    seg_out = ndimage.zoom(seg_roi, zoom_factors, order=0, mode='constant', cval=0)

    # ndimage.zoom can land on 127 or 129 due to floating-point rounding
    img_out, seg_out = _force_target_size(img_out, seg_out)

    return img_out, seg_out


def _force_target_size(image, seg):
    """
    After zoom, each axis can be off by +/-1.
    Trim (centre-crop) or pad (centre-pad) to exactly TARGET.
    """
    for ax in range(3):
        cur = image.shape[ax + 1]       # +1  ->  skip channel dim
        tgt = int(TARGET[ax])

        if cur == tgt:
            continue

        if cur > tgt:
            # centre-trim
            trim  = cur - tgt
            start = trim // 2

            slc_i = [slice(None)] * 4
            slc_i[ax + 1] = slice(start, start + tgt)
            image = image[tuple(slc_i)]

            slc_s = [slice(None)] * 3
            slc_s[ax] = slice(start, start + tgt)
            seg   = seg[tuple(slc_s)]

        else:   # cur < tgt
            # centre-pad
            pad_total  = tgt - cur
            pad_before = pad_total // 2
            pad_after  = pad_total - pad_before

            p_i = [(0, 0)] * 4
            p_i[ax + 1] = (pad_before, pad_after)
            image = np.pad(image, p_i, mode='constant', constant_values=0)

            p_s = [(0, 0)] * 3
            p_s[ax] = (pad_before, pad_after)
            seg = np.pad(seg, p_s, mode='constant', constant_values=0)

    return image, seg