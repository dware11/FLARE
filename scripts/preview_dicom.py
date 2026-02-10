from __future__ import annotations

import numpy as np 
import torch 

try: 
    import cv2
    _HAS_CV2 = True 
except ImportError: 
    _HAS_CV2 = False 

# Core Math for CT Transforms

def dicom_to_hu(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray: 
    # HU = pixel_value * rescale_slope + rescale_intercept
    # Gives consistent Hounsfield Units across different scanners and acquisition settings
    return pixel_array.astype(np.float32) * float(slope) + float(intercept) 

def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray: 
    # Windowing for the brain window: typically center=40, width=80 for brain CTs
    low = center - width / 2 
    high = center + width / 2 
    return np.clip(hu, low, high) 

def normalize_0_1(x: np.ndarray) -> np.ndarray: 
    # Normalizes array to [0, 1] range
    x = x.astype(np.float32) 
    mn = float(x.min())
    mx = float(x.max()) 
    if mx - mn < 1e-6:  
        return np.zeros_like(x, dtype=np.float32) 
    return (x-mn) / (mx - mn) 
    
def resize_2d(x: np.ndarray, size: tuple[int, int]) -> np.ndarray: 
    # Resizes 2D image to (H, W) 
    h, w = size 
    if _HAS_CV2:
        return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    
    # Fallback to PyTorch if cv2 is not available
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0) 
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False) 
    return t.squeeze(0).squeeze(0).numpy().astype(np.float32) 

# Public Transform APIs 

# Common CT windows will tweak 
BRAIN_WINDOW = (40, 80) 
SUBDURAL_WINDOW = (80, 2800)
BONE_WINDOW = (600, 2800) 

def hu_to_tensor_single_window(
        hu: np.ndarray, 
        size: tuple[int, int] = (256, 256), 
        center: float = 40, 
        width: float = 80, ) -> torch.Tensor: 
    
    x = apply_window(hu, center=center, width=width) 
    x = normalize_0_1(x) 
    x = resize_2d(x, size=size) 
    return torch.from_numpy(x).unsqueeze(0) # (1, H, W) for single channel

def hu_to_tensor_multi_window(
        hu: np.ndarray, 
        size: tuple[int, int] = (256, 256),
        windows: tuple[tuple[float, float], ...] = (BRAIN_WINDOW, SUBDURAL_WINDOW, BONE_WINDOW)    
        ) -> torch.Tensor: 
    
    chans = [] 
    for (c, w) in windows: 
        x = apply_window(hu, center=c, width=w) 
        x = normalize_0_1(x) 
        x = resize_2d(x, size=size) 
        chans.append(x) 

    stacked = np.stack(chans, axis=0).astype(np.float32)  # (C, H, W) for multi-channel input
    return torch.from_numpy(stacked)