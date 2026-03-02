"""
Demo script for a single CT patient:
- Save a "raw" middle-slice brain-windowed image from DICOM
- Save the preprocessed cache slice used by the model
- Run inference + Grad-CAM on that slice and save the overlay
- Print prediction scores

Intended for live demos to walk through the full workflow.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RAW_CT, META, OUTPUTS  # noqa: E402
from ml.brain.ct.ct_transforms import dicom_to_hu, ct_to_tensor  # noqa: E402
from scripts.preprocess_ct import get_middle_dicom_path  # noqa: E402
from ml.brain.ct.gradcam_ct import GradCAM  # noqa: E402
from ml.brain.ct.infer import _load_manifest, _load_model, _predict_one  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_gray_image(img: np.ndarray, out_path: Path) -> None:
    """
    Save a single-channel float32 image in [0,1] to disk as PNG.
    Tries cv2 first, then matplotlib as a fallback.
    """
    img = np.clip(img, 0.0, 1.0)
    img_uint8 = (img * 255.0).astype(np.uint8)

    try:
        import cv2  # type: ignore

        cv2.imwrite(str(out_path), img_uint8)
    except ImportError:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.imsave(str(out_path), img_uint8, cmap="gray")
        except ImportError:
            raise RuntimeError(
                f"Could not save image to {out_path}: install either opencv-python or matplotlib."
            )


def _save_rgb_image(img: np.ndarray, out_path: Path) -> None:
    """
    Save an RGB uint8 image to disk as PNG.
    Tries cv2 first, then matplotlib as a fallback.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    try:
        import cv2  # type: ignore

        # overlay is RGB; cv2 expects BGR
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except ImportError:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.imsave(str(out_path), img)
        except ImportError:
            raise RuntimeError(
                f"Could not save image to {out_path}: install either opencv-python or matplotlib."
            )


def _find_npz_for_patient(patient_id: str) -> Path:
    """Look up the cached npz path for a patient_id using the manifest."""
    manifest_path = META / "ct_processed_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries = _load_manifest(manifest_path)
    for e in entries:
        if e.get("patient_id") == patient_id:
            p = Path(e["path"])
            if p.exists():
                return p
    raise FileNotFoundError(f"No cache npz found for patient_id={patient_id} in {manifest_path}")


def save_raw_view(patient_id: str, out_dir: Path) -> Path:
    """
    Load the middle DICOM slice for a patient, convert to HU and a single-window
    brain view, and save a normalized grayscale PNG.
    """
    if not RAW_CT.exists():
        raise FileNotFoundError(
            f"RAW_CT root not found: {RAW_CT}. Check DATA_ROOT in src/config.py."
        )

    patient_dir = RAW_CT / patient_id
    ct_dir = patient_dir / "CT_SELECTED"
    if not ct_dir.is_dir():
        raise FileNotFoundError(f"CT_SELECTED folder not found for patient {patient_id} at {ct_dir}")

    try:
        import pydicom  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install pydicom to load raw DICOMs (pip install pydicom).") from e

    dicom_path = get_middle_dicom_path(ct_dir)
    dcm = pydicom.dcmread(str(dicom_path))
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    hu = dicom_to_hu(dcm.pixel_array, slope, intercept)

    # Use the same single-window transform as the rest of the pipeline
    tensor = ct_to_tensor(hu)  # (1,H,W)
    img = tensor.squeeze(0).numpy()

    out_path = out_dir / f"{patient_id}_raw_brain_window.png"
    _save_gray_image(img, out_path)
    print(f"[RAW] Saved brain-windowed middle slice to {out_path}")
    return out_path


def save_preprocessed_view(npz_path: Path, patient_id: str, out_dir: Path) -> Path:
    """
    Load the cached multi-window npz and save the first channel (brain window)
    as a grayscale PNG for inspection.
    """
    arr = np.load(npz_path)["arr"]  # (C,H,W)
    if arr.ndim != 3:
        raise ValueError(f"Expected cached arr to be (C,H,W), got shape {arr.shape}")

    # Channel 0 corresponds to the brain window in hu_to_multiwindow
    img = arr[0].astype(np.float32)
    # arr is already normalized to [0,1] in preprocessing, but re-clamp for safety
    img = np.clip(img, 0.0, 1.0)

    out_path = out_dir / f"{patient_id}_preprocessed_brain_channel.png"
    _save_gray_image(img, out_path)
    print(f"[PREPROCESSED] Saved cached brain channel to {out_path}")
    return out_path


def run_gradcam_and_scores(
    checkpoint: Path,
    npz_path: Path,
    patient_id: str,
    out_dir: Path,
) -> Tuple[Optional[Path], int, float, np.ndarray]:
    """
    Run model inference with gradients enabled on the cached slice, compute
    Grad-CAM on layer4, and save the overlay PNG. Returns (cam_path, pred, conf, probs).
    """
    model = _load_model(checkpoint)
    device = next(model.parameters()).device

    arr = np.load(npz_path)["arr"]
    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)

    probs = _predict_one(model, x, use_grad=True)
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    gradcam = GradCAM(model, "layer4")
    _, overlay = gradcam(x, target_class=pred, input_for_overlay=x)

    cam_path: Optional[Path] = None
    if overlay is not None:
        cam_path = out_dir / f"{patient_id}_gradcam_overlay.png"
        _save_rgb_image(overlay, cam_path)
        print(f"[GRAD-CAM] Saved overlay to {cam_path}")
    else:
        print("[GRAD-CAM] Overlay was None (no gradients/activations captured).")

    print(f"[SCORES] Prediction class index={pred} confidence={conf:.4f}")
    print(f"[SCORES] Probabilities: {probs}")
    return cam_path, pred, conf, probs


def main(patient_id: str, checkpoint: Optional[Path] = None, out_dir: Optional[Path] = None) -> None:
    checkpoint = checkpoint or (OUTPUTS / "ct_baseline_best.pt")
    out_dir = out_dir or (OUTPUTS / "demo" / patient_id)
    _ensure_dir(out_dir)

    print(f"=== Demo for patient {patient_id} ===")
    print(f"Using checkpoint: {checkpoint}")
    print(f"Output directory: {out_dir}")

    # 1) Raw middle slice view
    try:
        save_raw_view(patient_id, out_dir)
    except Exception as e:
        print(f"[WARN] Could not generate RAW view: {e}")

    # 2) Preprocessed cache view
    try:
        npz_path = _find_npz_for_patient(patient_id)
    except Exception as e:
        print(f"[ERROR] Could not find cached npz for patient {patient_id}: {e}")
        return

    save_preprocessed_view(npz_path, patient_id, out_dir)

    # 3) Grad-CAM overlay + scores
    try:
        run_gradcam_and_scores(checkpoint, npz_path, patient_id, out_dir)
    except Exception as e:
        print(f"[ERROR] Failed to run Grad-CAM inference: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Demo: raw → preprocessed → Grad-CAM overlay + scores for a single CT patient."
    )
    ap.add_argument("--patient-id", type=str, required=True, help="Patient ID (e.g. CQ500-CT-102)")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint path (defaults to OUTPUTS/ct_baseline_best.pt)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write PNGs (defaults to outputs/demo/<patient_id>)",
    )
    args = ap.parse_args()

    main(patient_id=args.patient_id, checkpoint=args.checkpoint, out_dir=args.out_dir)

