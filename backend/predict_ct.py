"""
FLARE - CT Prediction Wrapper

Safe adapter around existing CT inference + DICOM ZIP preprocessing.
This keeps app.py focused on request/response flow while preserving
current model behavior, thresholds, and checkpoint defaults.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


CT_K_SLICES_DEFAULT = 21
DICOM_EXTS = {".dcm", ".dicom"}
CT_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _is_ct_image(path_or_name: str) -> bool:
    return Path(path_or_name.lower()).suffix in CT_IMAGE_EXTS


def _is_ct_npz(path_or_name: str) -> bool:
    return Path(path_or_name.lower()).suffix == ".npz"


def _is_ct_zip(path_or_name: str) -> bool:
    return Path(path_or_name.lower()).suffix == ".zip"


def _default_ct_checkpoint() -> str:
    return os.environ.get(
        "FLARE_CT_CHECKPOINT",
        # OLD: /scratch/bckk/flare/ct_brain/outputs/rsna_windowed_exp2/best.pt
        "/scratch/bckk/flare/ct_brain/outputs/multimodal_finetune_ct/best_ct_finetune.pt",
    )


def _preprocess_dicom_zip_to_npz(
    zip_path: str,
    patient_id: str,
    k_slices: int = CT_K_SLICES_DEFAULT,
) -> str:
    """Extract ZIP DICOM study and convert to model-ready NPZ."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Uploaded file is not a valid ZIP archive")

    extract_dir = tempfile.mkdtemp(prefix="flare_dicom_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if not basename:
                    continue
                if basename.startswith(".") or basename.startswith("__"):
                    continue
                zf.extract(member, extract_dir)

        dcm_files: list[Path] = []
        for root, _dirs, files in os.walk(extract_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in DICOM_EXTS or (ext == "" and not fname.startswith(".")):
                    dcm_files.append(Path(root) / fname)

        if not dcm_files:
            raise ValueError(
                "ZIP contains no DICOM files (.dcm/.dicom). "
                "Ensure the ZIP contains a flat or nested folder of DICOM slices."
            )

        try:
            import pydicom

            pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        except Exception:
            raise ValueError(
                f"First file in ZIP ({dcm_files[0].name}) is not a valid DICOM. "
                "Ensure the ZIP contains standard DICOM CT slices."
            )

        from scripts.preprocess_ct import save_ct_volume_npz_from_dicoms

        all_dcms_flat = sorted(dcm_files, key=lambda p: p.name)
        try:
            from scripts.preprocess_ct import get_instance_number

            all_dcms_flat.sort(key=lambda p: (get_instance_number(p), p.name))
        except Exception:
            pass

        upload_dir = Path(__file__).resolve().parent / "static" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        npz_out = upload_dir / f"{patient_id}_ct_from_dicom.npz"
        save_ct_volume_npz_from_dicoms(all_dcms_flat, npz_out, k_slices)
        return str(npz_out)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def _prepare_ct_input(file_path: str, patient_id: str) -> tuple[str, str]:
    """
    Returns:
        (inference_path, input_format)
    input_format is one of: image | npz | dicom_zip
    """
    lower = file_path.lower()
    if _is_ct_zip(lower):
        return _preprocess_dicom_zip_to_npz(file_path, patient_id), "dicom_zip"
    if _is_ct_npz(lower):
        return file_path, "npz"
    if _is_ct_image(lower):
        return file_path, "image"
    raise ValueError(
        "CT inference requires a .npz volume, .zip DICOM study, or .jpg/.jpeg/.png image"
    )


def predict_ct(
    file_path: str,
    patient_id: str | None = None,
    allow_image: bool = True,
    checkpoint: str | None = None,
    threshold: float = 0.488,
    cam_dir: str | None = None,
) -> dict[str, Any]:
    """
    Safe adapter for CT inference using existing implementation.

    Preserves current defaults:
      - checkpoint env var/default path
      - threshold=0.488
      - image/npz/zip behavior
    """
    if patient_id is None:
        patient_id = Path(file_path).stem

    inference_path, input_format = _prepare_ct_input(file_path, patient_id)
    if input_format == "image" and not allow_image:
        raise ValueError("CT image input is not enabled for this endpoint")

    from ml.brain.ct.infer import run_ct_from_image, run_ct_from_npz

    ct_checkpoint = checkpoint or _default_ct_checkpoint()
    if input_format == "image":
        result = run_ct_from_image(
            inference_path,
            checkpoint=ct_checkpoint,
            cam_dir=cam_dir,
            threshold=threshold,
        )
    else:
        result = run_ct_from_npz(
            inference_path,
            checkpoint=ct_checkpoint,
            cam_dir=cam_dir,
            threshold=threshold,
        )

    if result is None:
        return {}

    out = dict(result)
    out["input_format"] = input_format
    out["inference_path"] = inference_path
    return out

