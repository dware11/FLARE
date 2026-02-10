"""
This module contains the transforms for the CT dataset.
"""

import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False 

def apply_window_hu(hu: np.ndarray, center: float, width: float) -> np.ndarray: 
    # Applied common brain window
    # center = 40, width = 80 => range [0,80]
    low = center - (width / 2 )
    high = center + (width / 2) 
    hu = np.clip(hu, low, high) 
    return hu 

def normalize_to_0_1(x: np.ndarray) -> np.ndarray: 
    x = x.astype(np.float32) 
    minv = x.min() 
    maxv = x.max() 
    if maxv - minv < 1e-6: 
        return np.zeros_like(x, dtype=np.float32)
    return (x - minv) / (maxv - minv) 

def resize_2d(x: np.ndarray, size=(256, 256)) -> np.ndarray:
    h, w = size[0], size[1]
    if _HAS_CV2:
        return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy().astype(np.float32) 


def dicom_to_hu(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Convert DICOM pixel array to Hounsfield units."""
    return pixel_array.astype(np.float32) * float(slope) + float(intercept)


def ct_to_tensor(
    hu: np.ndarray,
    size: tuple[int, int] = (256, 256),
    center: float = 40,
    width: float = 80,
) -> torch.Tensor:
    """Window HU, normalize to [0,1], resize, and return as (1, H, W) tensor."""
    x = apply_window_hu(hu, center=center, width=width)
    x = normalize_to_0_1(x)
    x = resize_2d(x, size=size)
    return torch.from_numpy(x).unsqueeze(0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT))

    from src.config import RAW

    try:
        import pydicom
    except ImportError:
        print("Install pydicom: pip install pydicom")
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        sys.exit(1)

    # Find first patient with CT_SELECTED DICOMs
    if not RAW.exists():
        print(f"Data root not found: {RAW}")
        print("Extract CQ500 data first (e.g. python scripts/extract_cq500.py)")
        sys.exit(1)

    patient_dirs = sorted(p for p in RAW.iterdir() if p.is_dir())
    dicom_path = None
    patient_id = None
    for pdir in patient_dirs:
        ct_dir = pdir / "CT_SELECTED"
        if not ct_dir.is_dir():
            continue
        dicoms = list(ct_dir.glob("*.dcm")) + list(ct_dir.glob("*.dicom"))
        if dicoms:
            dicom_path = dicoms[0]
            patient_id = pdir.name
            break

    if dicom_path is None:
        print("No patient with DICOMs in raw/<id>/CT_SELECTED found.")
        sys.exit(1)

    print(f"Loading: {patient_id} / {dicom_path.name}")
    dcm = pydicom.dcmread(str(dicom_path))
    slope = getattr(dcm, "RescaleSlope", 1.0)
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    hu = dicom_to_hu(dcm.pixel_array, slope, intercept)
    tensor = ct_to_tensor(hu)

    # Show normalized slice
    img = tensor.squeeze(0).numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(f"Patient {patient_id} — brain window")
    plt.axis("off")
    plt.tight_layout()
    plt.show()