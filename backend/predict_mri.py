"""
FLARE - MRI Prediction Pipeline
File: backend/predict_mri.py

Usage in app.py:
    from predict_mri import predict_mri
    result = predict_mri(file_path, modality, patient_id)

Modalities:
    brain_mri   → PNG/JPG → EfficientNetB0 classify → MiT-B3 segment
    brain_brats → NPZ     → 3D U-Net glioma sub-regions

Output (brain_mri):
    {
        "pred_label":      "pituitary",
        "result_class":    "Benign",
        "confidence":      0.9681,
        "probabilities":   [{"label": "pituitary", "value": 0.9681}, ...],
        "input_image_url": "/static/uploads/MRI_001_input.jpg",
        "segmentation": {
            "original_url":   "/static/masks/MRI_001_original.png",
            "mask_url":       "/static/masks/MRI_001_mask.png",
            "overlay_url":    "/static/masks/MRI_001_overlay.png",
            "tumor_area_pct": 1.46,
            "tumor_area_px":  3816,
            "tumor_area_mm2": 954.0,
            "model":          "MiT-B3 UNet"
        },
        "gradcam_url":   null,
        "gradcam_ready": false
    }

Output (brain_brats):
    {
        "pred_label":      "edema",
        "result_class":    "Malignant",
        "confidence":      0.8142,
        "probabilities":   null,
        "input_image_url": "/static/masks/MRI_001_t1c.png",
        "segmentation": {
            "original_url":       "/static/masks/MRI_001_t1c.png",
            "overlay_url":        "/static/masks/MRI_001_overlay.png",
            "mask_urls": {
                "necrotic_core":   "/static/masks/MRI_001_necrotic_core_mask.png",
                "edema":           "/static/masks/MRI_001_edema_mask.png",
                "enhancing_tumor": "/static/masks/MRI_001_enhancing_tumor_mask.png"
            },
            "volumes":     {"necrotic_core": 49708, ...},
            "volumes_mm3": {"necrotic_core": 49708.0, ...},
            "tumor_voxels_total": 251063,
            "slice_index":  72,
            "model":        "3D U-Net (BraTS)"
        },
        "gradcam_url":   null,
        "gradcam_ready": false
    }
"""

import os
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")


# PATHS
# ==============================================================================
CLS_CHECKPOINT   = "/scratch/bckk/flare/mri_brain/classification/outputs/output/best_model.pth"
SEG_CHECKPOINT   = "/scratch/bckk/flare/mri_brain/segmentation_brisc/outputs/swin_unet/best_model_swin.pth"
BRATS_CHECKPOINT = "/scratch/bckk/flare/projects/FLARE/ml/brain/mri/checkpoints/best_seg_model.pth"

MASK_OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "static", "masks")
UPLOAD_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(MASK_OUTPUT_DIR,   exist_ok=True)
os.makedirs(UPLOAD_OUTPUT_DIR, exist_ok=True)

# BRISC: 512x512 images, ~0.5mm pixel spacing
PIXEL_SPACING_MM  = 0.5
# BraTS: 1mm isotropic voxels
VOXEL_MM3         = 1.0


# LABEL MAPS
# ==============================================================================
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

LABEL_TO_RESULT_CLASS = {
    "glioma":     "Malignant",
    "meningioma": "Benign",
    "pituitary":  "Benign",
    "no_tumor":   "Normal",
}

# Tumor overlay colors (BGR for OpenCV)
TUMOR_COLORS_BGR = {
    "glioma":     (0,   80, 255),
    "meningioma": (255, 200,   0),
    "pituitary":  (0,  255, 200),
}

BRATS_LABELS = {
    1: "necrotic_core",
    2: "edema",
    3: "enhancing_tumor",
}

BRATS_COLORS_BGR = {
    1: (255, 100,   0),   # blue — necrotic core
    2: (  0, 180,   0),   # darker green — edema
    3: (  0, 140, 255),   # orange — enhancing tumor
}

# DEVICE + MODEL CACHE — load once, reuse across requests
# ==============================================================================
_device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cls_model   = None
_seg_model   = None
_brats_model = None


# MODEL LOADING
# ==============================================================================
def _load_cls_model():
    global _cls_model
    if _cls_model is not None:
        return _cls_model
    model = models.efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(p=0.15),
        nn.Linear(256, 4),
    )
    ckpt  = torch.load(CLS_CHECKPOINT, map_location=_device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval().to(_device)
    _cls_model = model
    return model


def _load_seg_model():
    global _seg_model
    if _seg_model is not None:
        return _seg_model
    model = smp.Unet(
        encoder_name="mit_b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    ckpt  = torch.load(SEG_CHECKPOINT, map_location=_device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval().to(_device)
    _seg_model = model
    return model


def _load_brats_model():
    global _brats_model
    if _brats_model is not None:
        return _brats_model

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
            )
        def forward(self, x): return self.block(x)

    class SegmentationUNet(nn.Module):
        def __init__(self, in_channels=4, out_channels=4):
            super().__init__()
            self.enc1 = ConvBlock(in_channels, 32);  self.pool1 = nn.MaxPool3d(2)
            self.enc2 = ConvBlock(32, 64);            self.pool2 = nn.MaxPool3d(2)
            self.enc3 = ConvBlock(64, 128);           self.pool3 = nn.MaxPool3d(2)
            self.enc4 = ConvBlock(128, 256);          self.pool4 = nn.MaxPool3d(2)
            self.bottleneck = ConvBlock(256, 512)
            self.up4  = nn.ConvTranspose3d(512, 256, 2, stride=2)
            self.dec4 = ConvBlock(512, 256)
            self.up3  = nn.ConvTranspose3d(256, 128, 2, stride=2)
            self.dec3 = ConvBlock(256, 128)
            self.up2  = nn.ConvTranspose3d(128, 64, 2, stride=2)
            self.dec2 = ConvBlock(128, 64)
            self.up1  = nn.ConvTranspose3d(64, 32, 2, stride=2)
            self.dec1 = ConvBlock(64, 32)
            self.seg_head = nn.Conv3d(32, out_channels, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
            bn = self.bottleneck(self.pool4(e4))
            d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.seg_head(d1)

    model = SegmentationUNet()
    ckpt  = torch.load(BRATS_CHECKPOINT, map_location=_device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval().to(_device)
    _brats_model = model
    return model


# PREPROCESSING — must match training exactly
# ==============================================================================
def _preprocess_for_classification(image_path: str):
    """PNG/JPG → tensor for EfficientNetB0. Matches BRISC training exactly."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img            = img.astype(np.float32)
    orig_h, orig_w = img.shape
    if img.max() <= 1.0:
        img = img * 255.0
    img_3ch   = np.stack([img, img, img], axis=0)
    tensor    = torch.tensor(img_3ch, dtype=torch.float32)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor  = transform(tensor).unsqueeze(0)
    display = (img / 255.0).clip(0, 1)
    return tensor, display, (orig_h, orig_w)


def _preprocess_for_segmentation(image_path: str):
    """PNG/JPG → tensor for MiT-B3. Matches BRISC seg training exactly."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    tensor = tensor.repeat(1, 3, 1, 1)
    mean   = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std    = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


# OUTPUT HELPERS
# ==============================================================================
def _sorted_probabilities(probs_array: np.ndarray) -> list:
    """Probabilities sorted largest to smallest for React bar chart."""
    return sorted(
        [{"label": CLASS_NAMES[i], "value": round(float(p), 4)}
         for i, p in enumerate(probs_array)],
        key=lambda x: x["value"],
        reverse=True
    )


def _save_input_image(image_path: str, patient_id: str) -> str:
    """Copy uploaded image to static/uploads for React display."""
    ext      = Path(image_path).suffix
    filename = f"{patient_id}_input{ext}"
    shutil.copy2(image_path, os.path.join(UPLOAD_OUTPUT_DIR, filename))
    return f"/static/uploads/{filename}"


def _save_brisc_outputs(mask_np: np.ndarray, display: np.ndarray,
                         pred_label: str, patient_id: str) -> dict:
    """
    Save 3 outputs React needs:
      original.png — clean grayscale MRI
      mask.png     — pure white/black binary (React overlays freely)
      overlay.png  — pre-blended MRI + colored tumor region
    """
    color  = TUMOR_COLORS_BGR.get(pred_label, (0, 255, 255))
    mri_u8 = (display * 255).astype(np.uint8)

    # 1. Original
    cv2.imwrite(
        os.path.join(MASK_OUTPUT_DIR, f"{patient_id}_original.png"),
        mri_u8
    )

    # 2. Binary mask
    cv2.imwrite(
        os.path.join(MASK_OUTPUT_DIR, f"{patient_id}_mask.png"),
        (mask_np > 0.5).astype(np.uint8) * 255
    )

    # 3. Colored overlay
    mri_bgr      = cv2.cvtColor(mri_u8, cv2.COLOR_GRAY2BGR)
    colored      = np.zeros_like(mri_bgr)
    colored[mask_np > 0.5] = color
    blended      = cv2.addWeighted(mri_bgr, 0.65, colored, 0.35, 0)
    cv2.imwrite(
        os.path.join(MASK_OUTPUT_DIR, f"{patient_id}_overlay.png"),
        blended
    )

    return {
        "original_url": f"/static/masks/{patient_id}_original.png",
        "mask_url":     f"/static/masks/{patient_id}_mask.png",
        "overlay_url":  f"/static/masks/{patient_id}_overlay.png",
    }

def _save_brats_outputs(pred_3d: np.ndarray, image_4ch: np.ndarray,
                         patient_id: str) -> dict:
    """
    Save BraTS outputs:
      t1c.png                       — T1c MRI slice (upscaled 512x512)
      overlay.png                   — T1c + all class colors combined
      {class}_mask.png per class    — pure binary (React overlays freely)
    Returns URLs + volumes in voxels and mm³.
    """
    slice_idx  = int((pred_3d > 0).sum(axis=(0, 1)).argmax())
    pred_slice = pred_3d[:, :, slice_idx].T

    # T1c display
    t1c      = image_4ch[1, :, :, slice_idx].T  # ← add .T
    p2, p98  = np.percentile(t1c, 2), np.percentile(t1c, 98)
    t1c_u8   = (np.clip((t1c - p2) / (p98 - p2 + 1e-8), 0, 1) * 255).astype(np.uint8)
    t1c_bgr  = cv2.cvtColor(t1c_u8, cv2.COLOR_GRAY2BGR)
    t1c_512 = cv2.resize(t1c_bgr, (512, 512), interpolation=cv2.INTER_CUBIC)


    cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, f"{patient_id}_t1c.png"), t1c_512)

    mask_urls   = {}
    volumes     = {}
    volumes_mm3 = {}
    combined    = t1c_512.copy()

    for cls in [1, 2, 3]:
        label    = BRATS_LABELS[cls]
        color    = BRATS_COLORS_BGR[cls]
        cls_mask = (pred_slice == cls).astype(np.uint8)

        # Volume
        voxels          = int((pred_3d == cls).sum())
        volumes[label]  = voxels
        volumes_mm3[label] = round(voxels * VOXEL_MM3, 1)

        # Binary mask — upscaled
        mask_512  = cv2.resize(cls_mask * 255, (512, 512),
                               interpolation=cv2.INTER_NEAREST)
        mask_path = os.path.join(MASK_OUTPUT_DIR, f"{patient_id}_{label}_mask.png")
        cv2.imwrite(mask_path, mask_512)
        mask_urls[label] = f"/static/masks/{patient_id}_{label}_mask.png"

        # Add to combined overlay
        colored = np.zeros_like(t1c_512)
        colored[mask_512 > 0] = color
        combined = cv2.addWeighted(combined, 1.0, colored, 0.4, 0)

    cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, f"{patient_id}_overlay.png"), combined)

    return {
        "original_url": f"/static/masks/{patient_id}_t1c.png",
        "overlay_url":  f"/static/masks/{patient_id}_overlay.png",
        "mask_urls":    mask_urls,
        "volumes":      volumes,
        "volumes_mm3":  volumes_mm3,
        "slice_index":  slice_idx,
    }


# PIPELINES
# ==============================================================================
def _predict_brisc(image_path: str, patient_id: str) -> dict:
    """BRISC: classify → branch → segment if tumor found."""

    # Classification
    cls_model                         = _load_cls_model()
    tensor, display, (orig_h, orig_w) = _preprocess_for_classification(image_path)

    with torch.no_grad():
        probs = F.softmax(cls_model(tensor.to(_device)), dim=1).cpu().squeeze().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    input_url  = _save_input_image(image_path, patient_id)

    result = {
        "pred_label":      pred_label,
        "result_class":    LABEL_TO_RESULT_CLASS[pred_label],
        "confidence":      round(confidence, 4),
        "probabilities":   _sorted_probabilities(probs),
        "input_image_url": input_url,
        "segmentation":    None,
        "gradcam_url":     None,
        "gradcam_ready":   False,
    }

    # Branch
    if pred_label == "no_tumor":
        result["segmentation"] = {
            "status":         "No tumor detected — clean scan",
            "original_url":   input_url,
            "mask_url":       None,
            "overlay_url":    None,
            "tumor_area_pct": 0.0,
            "tumor_area_px":  0,
            "tumor_area_mm2": 0.0,
            "model":          "MiT-B3 UNet",
        }
    else:
        try:
            seg_tensor = _preprocess_for_segmentation(image_path).to(_device)

            with torch.no_grad():
                mask_np = (torch.sigmoid(
                    _load_seg_model()(seg_tensor)
                ) > 0.5).float().cpu().squeeze().numpy()

            mask_resized    = cv2.resize(mask_np.astype(np.float32),
                                         (orig_w, orig_h),
                                         interpolation=cv2.INTER_NEAREST)
            display_resized = cv2.resize(display.astype(np.float32),
                                         (orig_w, orig_h))

            tumor_px       = int((mask_resized > 0.5).sum())
            seg_urls       = _save_brisc_outputs(
                mask_resized, display_resized, pred_label, patient_id
            )

            result["segmentation"] = {
                **seg_urls,
                "tumor_area_pct": round(tumor_px / mask_resized.size * 100, 2),
                "tumor_area_px":  tumor_px,
                "tumor_area_mm2": round(tumor_px * (PIXEL_SPACING_MM ** 2), 1),
                "model":          "MiT-B3 UNet",
            }
        except Exception as e:
            result["segmentation"] = {"error": str(e)}

    return result


def _predict_brats(npz_path: str, patient_id: str) -> dict:
    """BraTS: 3D U-Net glioma sub-region segmentation."""
    data   = np.load(npz_path, allow_pickle=True)
    image  = data["image"].astype(np.float32)

    if image.shape != (4, 128, 128, 128):
        raise ValueError(f"Expected (4,128,128,128), got {image.shape}")

    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(_device)

    with torch.no_grad():
        probs   = torch.softmax(_load_brats_model()(tensor), dim=1)
        pred_3d = probs.argmax(dim=1).squeeze().cpu().numpy()

    total_tumor  = int((pred_3d > 0).sum())
    dominant_cls = max([1, 2, 3], key=lambda c: int((pred_3d == c).sum())) \
                   if total_tumor > 0 else 2
    dom_mask     = (pred_3d == dominant_cls)
    dom_conf     = probs[:, dominant_cls].squeeze().cpu().numpy()
    confidence   = float(dom_conf[dom_mask].mean()) if dom_mask.any() else 0.0

    seg          = _save_brats_outputs(pred_3d, image, patient_id)

    return {
        "pred_label":      BRATS_LABELS[dominant_cls] if total_tumor > 0 else "no_detected_tumor",
        "result_class":    "Malignant",
        "confidence":      round(confidence, 4),
        "probabilities":   None,
        "input_image_url": seg["original_url"],
        "segmentation": {
            "original_url":       seg["original_url"],
            "overlay_url":        seg["overlay_url"],
            "mask_urls":          seg["mask_urls"],
            "volumes":            seg["volumes"],
            "volumes_mm3":        seg["volumes_mm3"],
            "tumor_voxels_total": total_tumor,
            "slice_index":        seg["slice_index"],
            "model":              "3D U-Net (BraTS)",
        },
        "gradcam_url":   None,
        "gradcam_ready": False,
    }


# MAIN ENTRY POINT — Flask calls this
# ==============================================================================
def predict_mri(file_path: str,
                modality:   str = "brain_mri",
                patient_id: str = "MRI_000") -> dict:
    """
    Args:
        file_path:  PNG/JPG for brain_mri | NPZ for brain_brats
        modality:   "brain_mri" or "brain_brats"
        patient_id: used for output filenames

    Returns:
        dict — full prediction result
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    try:
        if modality == "brain_brats":
            return _predict_brats(file_path, patient_id)
        return _predict_brisc(file_path, patient_id)
    except Exception as e:
        return {"error": str(e), "pred_label": None, "modality": modality}