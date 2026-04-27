"""
Demo: Single patient (--patient-id/--npz) or batch from manifest (no id => batch).
Design: CAM flags (--cam-dir, --cam-limit, --cam-when-abnormal) toggle gradients
and overlay saving; inference stays lightweight when CAM is off.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path 
from typing import Any, Optional, List, Dict, Tuple, Union 

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(ROOT)) 

from src.config import META, OUTPUTS
from ml.brain.ct.threshold_util import resolve_abnormal_threshold
from ml.brain.ct.model_ct import (
    build_ct_model,
    build_ct_sequence_model,
    is_sequence_ct_model,
    load_sequence_weights_compat,
    patient_logits_from_model,
)
from ml.brain.ct.gradcam_ct import GradCAM, normalize_ct_raw_volume_for_viz

LABELS = ["normal", "abnormal"]

SPLIT_SEED = 42

# Warn once per NPZ path when K differs from checkpoint / expected preprocess depth.
_K_WARNED_PATHS: set = set()

# Display-only: rotate/flip saved CT Grad-CAM PNGs to match UI orientation (not model input / logits / raw CAM).
_CT_CAM_ORIENTS = frozenset(
    {
        "none",
        "rot90_cw",
        "rot90_ccw",
        "rot180",
        "flip_lr",
        "flip_ud",
        "transpose",
        "transpose_flip_lr",
    }
)
_CAM_ORIENT_INVALID_WARNED: set = set()

_FLARE_CT_CAM_ENV = "FLARE_CT_CAM_DISPLAY_ORIENTATION"


def resolve_flare_ct_cam_display_orientation() -> str:
    """
    Read FLARE_CT_CAM_DISPLAY_ORIENTATION (default: none).
    Invalid values log once and map to 'none' (no display transform on disk).
    """
    raw = os.environ.get(_FLARE_CT_CAM_ENV, "none")
    o = (raw or "none").strip().lower() or "none"
    if o in _CT_CAM_ORIENTS:
        return o
    if o not in _CAM_ORIENT_INVALID_WARNED:
        _CAM_ORIENT_INVALID_WARNED.add(o)
        print(
            f"WARNING: {_FLARE_CT_CAM_ENV}={raw!r} is not allowed; use one of {sorted(_CT_CAM_ORIENTS)}. Using 'none'.",
            flush=True,
        )
    return "none"


def apply_ct_cam_display_orientation(image_or_array, orientation: str):
    """
    Display-only transform for saved Grad-CAM visualization arrays/images.
    Supports uint8 (H, W, C) with C=3 or 4, or a PIL Image (RGB / RGBA).
    For orientation 'none', returns the input without copying when possible.
    """
    o = (orientation or "none").strip().lower() or "none"
    if o not in _CT_CAM_ORIENTS:
        o = "none"
    if o == "none":
        try:
            from PIL import Image  # type: ignore

            if isinstance(image_or_array, Image.Image):
                return image_or_array
        except Exception:
            pass
        return image_or_array

    try:
        from PIL import Image  # type: ignore

        if isinstance(image_or_array, Image.Image):
            arr = np.array(image_or_array)
            out = _apply_ct_cam_spatial_to_hwc(arr, o)
            return Image.fromarray(out, mode=image_or_array.mode)
    except Exception:
        pass
    arr = np.ascontiguousarray(np.asarray(image_or_array))
    return _apply_ct_cam_spatial_to_hwc(arr, o)


def _apply_ct_cam_spatial_to_hwc(x: np.ndarray, o: str) -> np.ndarray:
    if x.ndim != 3 or x.shape[2] not in (3, 4):
        return x
    if o == "rot90_cw":
        return np.rot90(x, k=-1, axes=(0, 1))
    if o == "rot90_ccw":
        return np.rot90(x, k=1, axes=(0, 1))
    if o == "rot180":
        return np.rot90(x, k=2, axes=(0, 1))
    if o == "flip_lr":
        return np.ascontiguousarray(np.fliplr(x))
    if o == "flip_ud":
        return np.ascontiguousarray(np.flipud(x))
    if o == "transpose":
        return np.ascontiguousarray(np.transpose(x, (1, 0, 2)))
    if o == "transpose_flip_lr":
        t = np.transpose(x, (1, 0, 2))
        return np.ascontiguousarray(np.fliplr(t))
    return x


def _save_overlay_png(overlay: np.ndarray, out_path: Path) -> bool:
    """Save RGB overlay (H,W,3) uint8 to PNG. Tries cv2 then matplotlib. Returns True if saved."""
    try:
        import cv2
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return True
    except ImportError:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.imsave(str(out_path), overlay)
        return True
    except ImportError:
        return False
TRAIN_FRAC = 0.7 
VAL_FRAC = 0.15 

def _load_patient_arr(path: Path, expected_k: Optional[int] = None) -> np.ndarray:
    """Load NPZ; return arr (k, C, H, W) float32. Handles (C, H,W) -> (1,C,H,W)"""
    data = np.load(path)
    arr = data["arr"]
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    arr = arr.astype(np.float32)
    if expected_k is not None and arr.shape[0] != expected_k:
        key = str(path.resolve())
        if key not in _K_WARNED_PATHS:
            _K_WARNED_PATHS.add(key)
            print(
                f"WARNING: NPZ K-slices mismatch (expected {expected_k}, got {arr.shape[0]}) path={path}",
                flush=True,
            )
    return arr


def _thickness_from_npz(path: Path, device: torch.device) -> torch.Tensor:
    """Normalized thickness (B=1, 1); matches CTDataset scaling."""
    data = np.load(path)
    if "slice_thickness_mm" in data.files:
        tmm = float(np.asarray(data["slice_thickness_mm"]).reshape(-1)[0])
    else:
        tmm = 0.0
    tn = min(max(tmm, 0.0) / 10.0, 3.0)
    return torch.tensor([[tn]], dtype=torch.float32, device=device)


def _imagenet_normalize_volume(x: torch.Tensor) -> torch.Tensor:
    """Match CTDataset val/test: (k, C, H, W) with C=3."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _denormalize_volume(x_norm: torch.Tensor) -> torch.Tensor:
    """Inverse ImageNet norm for GradCAM overlay display; x_norm (k, C, H, W)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_norm.device, dtype=x_norm.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x_norm.device, dtype=x_norm.dtype).view(1, 3, 1, 1)
    return (x_norm * std + mean).clamp(0.0, 1.0)


def _unpack_gradcam_out(
    res: Union[Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any, Dict]],
) -> Tuple[torch.Tensor, Any, Dict]:
    if isinstance(res, tuple) and len(res) == 3:
        h, o, m = res
        return h, o, m if isinstance(m, dict) else {}
    if isinstance(res, tuple) and len(res) == 2:
        return res[0], res[1], {}
    raise TypeError("GradCAM must return a 2- or 3-tuple")


def _run_ct_core(
    model: torch.nn.Module,
    agg: str,
    eval_threshold: float,
    x_batch: torch.Tensor,
    x_raw_volume: torch.Tensor,
    x_norm_volume: torch.Tensor,
    thick: torch.Tensor,
    cam_dir: Optional[Path],
    cam_stem: str,
) -> Dict:
    """
    x_batch: (1, k, C, H, W) ImageNet-normalized.
    x_raw_volume: (k, C, H, W) ~[0,1] for overlay.
    x_norm_volume: (k, C, H, W) normalized (same as x_batch[0]).
    """
    use_grad = cam_dir is not None
    probs = _predict_one(model, x_batch, use_grad=use_grad, agg=agg, thickness=thick)
    pred = 1 if float(probs[1]) >= eval_threshold else 0

    cam_path = None
    cam_display_slice_index: Optional[int] = None
    cam_center_slice_index: Optional[int] = None
    cam_selection_method: Optional[str] = None
    cam_error: Optional[str] = None
    cam_display_orientation: str = resolve_flare_ct_cam_display_orientation()
    if use_grad and cam_dir:
        cam_dir_p = Path(cam_dir)
        ge = _gradcam_save_files(
            model,
            x_batch,
            x_raw_volume,
            x_norm_volume,
            thick,
            pred,
            cam_dir_p,
            cam_stem,
        )
        cam_path = ge.get("cam_path")
        cam_display_slice_index = ge.get("cam_display_slice_index")
        cam_center_slice_index = ge.get("cam_center_slice_index")
        cam_selection_method = ge.get("cam_selection_method")
        cam_error = ge.get("cam_error")
        if ge.get("cam_display_orientation") is not None:
            cam_display_orientation = str(ge["cam_display_orientation"])

    return {
        "label": LABELS[pred],
        "confidence": float(probs[pred]),
        "p_normal": float(probs[0]),
        "p_abnormal": float(probs[1]),
        "cam_path": cam_path,
        "cam_display_slice_index": cam_display_slice_index,
        "cam_center_slice_index": cam_center_slice_index,
        "cam_selection_method": cam_selection_method,
        "cam_error": cam_error,
        "cam_display_orientation": cam_display_orientation,
    }

def _load_manifest(manifest: Path) -> list: 
    with open(manifest, encoding="utf-8") as f:  
        return json.load(f) 

def _get_split_entries(entries: list, split: str): 
    """
    Deterministic split using fixed RNG seed. Matches train/val/test fractions. 
    """ 
    valid = [e for e in entries if e.get("label", -1) in (0, 1)] 
    if not valid: 
        valid = entries 

    if split == "all": 
        out = [] 
        for e in valid: 
            e_copy = dict(e) 
            e_copy.setdefault("split", "all") 
            out.append(e_copy) 
        return out 

    rng = np.random.default_rng(SPLIT_SEED) 
    n = len(valid) 
    inx = rng.permutation(n) 
    
    nt = int(n * TRAIN_FRAC) 
    nv = int(n * VAL_FRAC) 

    train_idx = set(inx[:nt]) 
    val_idx = set(inx[nt : nt + nv]) 
    test_idx = set(inx[nt + nv :])

    split_map = {"train": train_idx, "val": val_idx, "test": test_idx} 
    idx_set = split_map.get(split, val_idx) 

    out = [] 
    for i, e in enumerate(valid): 
        if i in idx_set: 
            e_copy = dict(e) 
            e_copy["split"] = split 
            out.append(e_copy) 

    return out 

def _apply_limit_and_random(entries: list, limit: Optional[int], random_n: Optional[int]) -> list: 
    """
    Limit: take first N deterministically 
    random_n: take N random entries deterministically 
    If both provided: random_n applied first, then limit. 
    """

    out = list(entries) 

    if random_n is not None: 
        rng = np.random.default_rng(SPLIT_SEED) 
        if random_n < len(out): 
            idx = rng.choice(len(out), size=random_n, replace=False) 
            out = [out[i] for i in idx] 

    if limit is not None: 
        out = out[:limit] 

    return out 

def _load_model(checkpoint: Path):
    """Load checkpoint; returns (model, model_kind, agg, k_slices_or_none, ckpt_meta_dict)."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found {checkpoint}")

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_kind = ckpt.get("model_kind", "legacy") if isinstance(ckpt, dict) else "legacy"
    agg = ckpt.get("agg", "max") if isinstance(ckpt, dict) else "max"
    rnn_type = ckpt.get("rnn_type", "gru") if isinstance(ckpt, dict) else "gru"
    sequence_pool = ckpt.get("sequence_pool", "max") if isinstance(ckpt, dict) else "max"
    sequence_topk = ckpt.get("sequence_topk", 3) if isinstance(ckpt, dict) else 3
    rnn_dropout = ckpt.get("rnn_dropout", 0.25) if isinstance(ckpt, dict) else 0.25
    use_thickness = ckpt.get("use_thickness", False) if isinstance(ckpt, dict) else False

    if model_kind == "sequence":
        if "head.weight" in state and "slice_head.weight" not in state:
            sequence_pool = "center"
        model = build_ct_sequence_model(
            rnn_type=rnn_type,
            sequence_pool=sequence_pool,
            topk=sequence_topk,
            rnn_dropout=rnn_dropout,
            use_thickness=use_thickness,
        )
        load_sequence_weights_compat(model, state)
    else:
        model = build_ct_model()
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            incomp = model.load_state_dict(state, strict=False)
            print(
                "WARNING: CT checkpoint loaded with strict=False; "
                f"missing_keys={incomp.missing_keys} unexpected_keys={incomp.unexpected_keys}",
                flush=True,
            )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    ckpt_meta: Dict = {}
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt_meta = ckpt
    k_slices_meta = ckpt_meta.get("k_slices")
    return model, model_kind, agg, k_slices_meta, ckpt_meta

def _predict_one(
    model: torch.nn.Module,
    x: torch.Tensor,
    use_grad: bool,
    agg: str = "max",
    thickness: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    x: (1, k, C, H, W) after ImageNet normalization.
    thickness: (1, 1) optional; used when model was trained with use_thickness.
    Returns probs as numpy array shape (2,).
    """
    if use_grad:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = patient_logits_from_model(model, x, agg=agg, thickness=thickness)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().detach().numpy()
        return probs

    with torch.no_grad():
        logits = patient_logits_from_model(model, x, agg=agg, thickness=thickness)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs

def _gradcam_layer_name(model: torch.nn.Module) -> str:
    """
    Default: denseblock3 (higher spatial res than the last block; often clearer localization).
    Override: FLARE_GRADCAM_LAYER=backbone.features.denseblock4
    """
    override = os.environ.get("FLARE_GRADCAM_LAYER", "").strip()
    if override:
        return override
    base = "backbone.features" if is_sequence_ct_model(model) else "features"
    return f"{base}.denseblock3"


def _save_jet_heatmap_png(heatmap_2d: torch.Tensor, out_path: Path, cam_orientation: str = "none") -> bool:
    """Save a 2D [0,1] heatmap as a JET PNG (debug/demo). Display orientation is display-only."""
    try:
        import cv2

        h = torch.clamp(heatmap_2d.detach().cpu(), 0, 1).numpy()
        h = (h * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        bgr = apply_ct_cam_display_orientation(bgr, cam_orientation)
        return bool(cv2.imwrite(str(out_path), bgr))
    except Exception:
        return False


def _save_center_preview_rgb(center_raw: torch.Tensor, out_path: Path, cam_orientation: str = "none") -> bool:
    """Grayscale center slice as RGB, same robust normalization as the overlay underlay."""
    g = center_raw[0, 0].float().cpu().numpy()
    if g.max() > 1.5:
        g = np.clip(g / 255.0, 0.0, 1.0)
    else:
        g = np.clip(g, 0.0, 1.0)
    p_lo, p_hi = float(np.percentile(g, 2.0)), float(np.percentile(g, 98.0))
    if p_hi - p_lo > 1e-6:
        g = np.clip((g - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0)
    rgb = (np.stack([g, g, g], axis=-1) * 255.0).astype(np.uint8)
    rgb = apply_ct_cam_display_orientation(rgb, cam_orientation)
    return _save_overlay_png(rgb, out_path)


def _gradcam_save_files(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    x_raw_volume: torch.Tensor,
    x_norm_volume: torch.Tensor,
    thick: torch.Tensor,
    pred: int,
    cam_dir: Path,
    cam_stem: str,
) -> Dict:
    """
    Run Grad-CAM: sequence models use the full (1,k,*,*,*) input + thickness, matching
    the study-level forward. The saved overlay (attention, not segmentation) uses the
    slice selected by per-slice saliency (top-5% mean) when the hook is (k,C,h,w);
    else falls back to center-slice (k//2) as before.
    Writes {stem}.png (overlay, API cam_path), and optionally {stem}_heatmap.png,
    {stem}_preview.png, {stem}_center.png, {stem}_best.png, {stem}_top3.png for demos.
    """
    cam_orn = resolve_flare_ct_cam_display_orientation()
    try:
        x_raw_viz = normalize_ct_raw_volume_for_viz(x_raw_volume, like=x_batch)
    except Exception as e:
        print(f"WARNING: CT Grad-CAM skipped (raw volume shape for overlay): {e}", flush=True)
        return {
            "cam_path": None,
            "cam_display_slice_index": None,
            "cam_center_slice_index": None,
            "cam_selection_method": None,
            "cam_error": str(e),
            "cam_display_orientation": cam_orn,
        }
    try:
        gradcam = GradCAM(model, _gradcam_layer_name(model))
        k = int(x_raw_viz.shape[0])
        k_mid = k // 2
        center_raw = x_raw_viz[k_mid : k_mid + 1]
        is_seq = is_sequence_ct_model(model)
        if is_seq:
            gout = gradcam(
                x_batch,
                target_class=pred,
                input_for_overlay=center_raw,
                thickness=thick,
                x_raw_volume=x_raw_viz,
            )
        else:
            x_cam = x_norm_volume[k_mid : k_mid + 1]
            gout = gradcam(
                x_cam,
                target_class=pred,
                input_for_overlay=center_raw,
                thickness=None,
            )
        heatmap, overlay, meta = _unpack_gradcam_out(gout)
        out: Dict = {
            "cam_path": None,
            "cam_display_slice_index": k_mid,
            "cam_center_slice_index": k_mid,
            "cam_selection_method": None,
        }
        if meta.get("cam_display_slice_index") is not None:
            out["cam_display_slice_index"] = meta["cam_display_slice_index"]
        if meta.get("cam_center_slice_index") is not None:
            out["cam_center_slice_index"] = meta["cam_center_slice_index"]
        if meta.get("cam_selection_method"):
            out["cam_selection_method"] = meta["cam_selection_method"]

        if overlay is None:
            out["cam_display_orientation"] = cam_orn
            return out
        cam_dir.mkdir(parents=True, exist_ok=True)
        out_path = cam_dir / f"{cam_stem}.png"
        overlay_save = apply_ct_cam_display_orientation(np.asarray(overlay), cam_orn)
        if not _save_overlay_png(overlay_save, out_path):
            out["cam_display_orientation"] = cam_orn
            return out
        out["cam_path"] = str(out_path)
        out["cam_display_orientation"] = cam_orn
        print(f"[CT CAM] display orientation = {cam_orn}", flush=True)
        if heatmap is not None:
            _save_jet_heatmap_png(heatmap, cam_dir / f"{cam_stem}_heatmap.png", cam_orientation=cam_orn)
        _save_center_preview_rgb(center_raw, cam_dir / f"{cam_stem}_preview.png", cam_orientation=cam_orn)

        hms = meta.get("heatmaps_per_slice")
        sc_list = meta.get("top5_mean_per_slice")
        if hms is not None and sc_list is not None and len(hms) == k and len(sc_list) == k:
            o_center = gradcam._overlay(hms[k_mid], x_raw_viz[k_mid : k_mid + 1])
            _save_overlay_png(
                apply_ct_cam_display_orientation(np.asarray(o_center), cam_orn),
                cam_dir / f"{cam_stem}_center.png",
            )
            _save_overlay_png(
                apply_ct_cam_display_orientation(np.asarray(overlay), cam_orn),
                cam_dir / f"{cam_stem}_best.png",
            )
            order = sorted(range(k), key=lambda j: -float(sc_list[j]))[:3]
            try:
                tiles = [gradcam._overlay(hms[j], x_raw_viz[j : j + 1]) for j in order]
                if len(tiles) == 1:
                    top3_arr = tiles[0]
                else:
                    top3_arr = np.hstack(tiles)
                _save_overlay_png(
                    apply_ct_cam_display_orientation(np.asarray(top3_arr), cam_orn),
                    cam_dir / f"{cam_stem}_top3.png",
                )
            except Exception:
                pass

        return out
    except Exception as e:
        print(f"WARNING: CT Grad-CAM failed (inference was successful): {e}", flush=True)
        return {
            "cam_path": None,
            "cam_display_slice_index": None,
            "cam_center_slice_index": None,
            "cam_selection_method": None,
            "cam_error": str(e),
            "cam_display_orientation": cam_orn,
        }


def _run_single_inference(
    checkpoint: Path,
    patient_id: Optional[str] = None,
    npz_path: Optional[Path] = None,
    cam_dir: Optional[Path] = None,
    threshold_cli: Optional[float] = None,
) -> None:
    model, _model_kind, agg, k_slices_meta, ckpt_meta = _load_model(checkpoint)
    eval_threshold = resolve_abnormal_threshold(cli=threshold_cli, default=0.5, ckpt=ckpt_meta)
    print(
        f"[threshold] Binary decision uses P(abnormal)>={eval_threshold:.4f} "
        f"(CLI > env ABNORMAL_THRESHOLD > checkpoint eval_threshold > default)",
        flush=True,
    )
    expected_k = int(k_slices_meta) if k_slices_meta is not None else None
    device = next(model.parameters()).device

    # Demo: loads NPZ from manifest lookup or direct --npz path (no DICOM here).
    if npz_path and npz_path.exists():
        path = npz_path
        inp_desc = f"npz {npz_path}"
        cam_id = npz_path.stem
    elif patient_id:
        manifest = META / "ct_processed_manifest.json"
        if not manifest.exists():
            print(f"Manifest not found: {manifest}")
            return
        entries = _load_manifest(manifest)
        path = None
        for e in entries:
            if e.get("patient_id") == patient_id:
                path = Path(e["path"])
                break
        if not path or not path.exists():
            print(f"Patient {patient_id} not found in cache/manifest")
            return
        inp_desc = f"patient_id {patient_id}"
        cam_id = patient_id
    else:
        print("Provide --patient-id or --npz for single patient mode, or leave both out for batch mode.")
        return

    print(f"Running inference on {inp_desc}")

    arr = _load_patient_arr(path, expected_k=expected_k)
    x_raw = torch.from_numpy(arr).float().to(device)
    x_all = _imagenet_normalize_volume(x_raw)
    x_batch = x_all.unsqueeze(0)
    thick = _thickness_from_npz(path, device)
    use_grad = cam_dir is not None
    probs = _predict_one(model, x_batch, use_grad=use_grad, agg=agg, thickness=thick)
    pred = 1 if float(probs[1]) >= eval_threshold else 0
    conf = float(probs[pred])
    if cam_dir is not None:
        ge = _gradcam_save_files(
            model,
            x_batch,
            x_raw,
            x_all,
            thick,
            pred,
            Path(cam_dir),
            cam_id,
        )
        if ge.get("cam_path"):
            print(f"CAM overlay saved to {ge['cam_path']}")
        else:
            print("Install opencv-python or matplotlib to save CAM overlay.")

    print(f"Prediction: {LABELS[pred]} (confidence={conf:.4f})")
    for i, lab in enumerate(LABELS):
        print(f" {lab}: {probs[i]:.4f}")

def _run_batch_inference(
    checkpoint: Path,
    manifest_path: Path,
    split: str = "val",
    limit: Optional[int] = None,
    random_n: Optional[int] = None,
    out_csv: Optional[Path] = None,
    cam_dir: Optional[Path] = None,
    cam_limit: Optional[int] = None,
    cam_when_abnormal: bool = False,
    threshold_cli: Optional[float] = None,
) -> list:
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return []

    model, _model_kind, agg, k_slices_meta, ckpt_meta = _load_model(checkpoint)
    eval_threshold = resolve_abnormal_threshold(cli=threshold_cli, default=0.5, ckpt=ckpt_meta)
    print(
        f"[threshold] Binary decision uses P(abnormal)>={eval_threshold:.4f} "
        f"(CLI > env ABNORMAL_THRESHOLD > checkpoint eval_threshold > default)",
        flush=True,
    )
    expected_k = int(k_slices_meta) if k_slices_meta is not None else None
    device = next(model.parameters()).device

    entries = _load_manifest(manifest_path)
    entries = _get_split_entries(entries, split) 
    entries = _apply_limit_and_random(entries, limit=limit, random_n=random_n) 

    if not entries:
        print(f"No entries found for split={split}")
        return []

    # Design: loop over manifest entries one-by-one (batch_size=1; laptop-friendly).
    print(f"Running batch inference on {len(entries)} patients (split={split})...")
    results: List[Dict] = []
    cam_count = 0
    csv_has_cam = False 

    use_grad = cam_dir is not None

    for e in entries:
        path = Path(e["path"])
        pid = e.get("patient_id", "")

        if not path.exists():
            print(f" SKIP {pid}: npz not found at {path}")
            continue

        arr = _load_patient_arr(path, expected_k=expected_k)
        x_raw = torch.from_numpy(arr).float().to(device)
        x_all = _imagenet_normalize_volume(x_raw)
        x_batch = x_all.unsqueeze(0)
        thick = _thickness_from_npz(path, device)
        probs = _predict_one(model, x_batch, use_grad=use_grad, agg=agg, thickness=thick)
        pred = 1 if float(probs[1]) >= eval_threshold else 0
        conf = float(probs[pred])

        row: Dict = {
            "patient_id": pid,
            "split": e.get("split", split),
            "true_label": int(e.get("label", -1)),
            "pred_label": pred,
            "confidence": conf,
            "p_normal": float(probs[0]),
            "p_abnormal": float(probs[1]),
        }

        # Note: optional CAM limit + abnormal-only filter (--cam-limit, --cam-when-abnormal).
        do_cam = (
            cam_dir is not None
            and (cam_limit is None or cam_count < cam_limit)
            and (not cam_when_abnormal or pred == 1)
        )

        if do_cam and cam_dir is not None:
            ge = _gradcam_save_files(
                model,
                x_batch,
                x_raw,
                x_all,
                thick,
                pred,
                Path(cam_dir),
                pid,
            )
            cp = ge.get("cam_path")
            if cp:
                row["cam_path"] = cp
                csv_has_cam = True
                cam_count += 1
            else:
                print("Install opencv-python or matplotlib to save CAM overlay.")

        results.append(row)

    num_pred = len(results)
    print(f"Predicted {num_pred} patients (split={split}).")

    labeled = [r for r in results if r["true_label"] in (0, 1)]
    if labeled:
        correct = sum(1 for r in labeled if r["true_label"] == r["pred_label"])
        acc = correct / len(labeled)
        print(f"Accuracy on labeled subset: {acc:.3f} ({correct}/{len(labeled)})")
        y_true = [int(r["true_label"]) for r in labeled]
        y_pred = [int(r["pred_label"]) for r in labeled]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print(
            "Confusion matrix (true=rows, pred=cols) [0=normal, 1=abnormal]:\n"
            f"{cm}"
        )
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"Sensitivity (abnormal): {sens:.4f}  Specificity: {spec:.4f}")
    else:
        print("No ground-truth labels in manifest; confusion matrix / accuracy not computed.")

    # Output: export predictions for analysis (CSV with optional cam_path column).
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "patient_id", "split", "true_label", "pred_label",
            "confidence", "p_normal", "p_abnormal",
        ]
        if csv_has_cam:
            fieldnames.append("cam_path")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"Wrote CSV to {out_csv}")

    return results


def main(
    patient_id: Optional[str] = None,
    npz_path: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    split: str = "val",
    limit: Optional[int] = None,
    random_n: Optional[int] = None,
    out_csv: Optional[Path] = None,
    cam_dir: Optional[Path] = None,
    cam_limit: Optional[int] = None,
    cam_when_abnormal: bool = False,
    threshold_cli: Optional[float] = None,
) -> None:
    checkpoint = checkpoint or OUTPUTS / "ct_baseline_best.pt"
    manifest_path = META / "ct_processed_manifest.json"

    if patient_id or (npz_path and npz_path.exists()):
        _run_single_inference(
            checkpoint=checkpoint,
            patient_id=patient_id,
            npz_path=npz_path,
            cam_dir=cam_dir,
            threshold_cli=threshold_cli,
        )
        return

    _run_batch_inference(
        checkpoint=checkpoint,
        manifest_path=manifest_path,
        split=split,
        limit=limit,
        random_n=random_n,
        out_csv=out_csv,
        cam_dir=cam_dir,
        cam_limit=cam_limit,
        cam_when_abnormal=cam_when_abnormal,
        threshold_cli=threshold_cli,
    )

def run_ct_for_patient(patient_id, checkpoint=None, cam_dir="backend/camo_outpus", threshold: Optional[float] = None):
    """Runs CT inference + GradCAM for one patient. Returns plain dict.
    Threshold: argument > env ABNORMAL_THRESHOLD > checkpoint eval_threshold > 0.5.
    """
    import numpy as np
    import torch
    from pathlib import Path
    from src.config import META, OUTPUTS

    if checkpoint is None:
        checkpoint = OUTPUTS / "ct_baseline_best.pt"
    manifest_path = META / "ct_processed_manifest.json"

    entries = _load_manifest(manifest_path)
    entry = next((e for e in entries if e.get("patient_id") == patient_id), None)
    if entry is None:
        return None

    arr_path = Path(entry["path"])
    if not arr_path.exists():
        return None

    model, _model_kind, agg, k_slices_meta, ckpt_meta = _load_model(checkpoint)
    eval_threshold = resolve_abnormal_threshold(cli=threshold, default=0.5, ckpt=ckpt_meta)
    expected_k = int(k_slices_meta) if k_slices_meta is not None else None
    device = next(model.parameters()).device

    arr = _load_patient_arr(arr_path, expected_k=expected_k)
    x_raw = torch.from_numpy(arr).float().to(device)
    x_all = _imagenet_normalize_volume(x_raw)
    x_batch = x_all.unsqueeze(0)
    thick = _thickness_from_npz(arr_path, device)
    use_grad = cam_dir is not None
    probs = _predict_one(model, x_batch, use_grad=use_grad, agg=agg, thickness=thick)
    pred = 1 if float(probs[1]) >= eval_threshold else 0

    cam_path = None
    cam_display_slice_index: Optional[int] = None
    cam_center_slice_index: Optional[int] = None
    cam_selection_method: Optional[str] = None
    cam_error: Optional[str] = None
    cam_display_orientation: str = resolve_flare_ct_cam_display_orientation()
    if use_grad and cam_dir:
        ge = _gradcam_save_files(
            model,
            x_batch,
            x_raw,
            x_all,
            thick,
            pred,
            Path(cam_dir),
            patient_id,
        )
        cam_path = ge.get("cam_path")
        cam_display_slice_index = ge.get("cam_display_slice_index")
        cam_center_slice_index = ge.get("cam_center_slice_index")
        cam_selection_method = ge.get("cam_selection_method")
        cam_error = ge.get("cam_error")
        if ge.get("cam_display_orientation") is not None:
            cam_display_orientation = str(ge["cam_display_orientation"])

    return {
        "patient_id": patient_id,
        "label": LABELS[pred],
        "confidence": float(probs[pred]),
        "p_normal": float(probs[0]),
        "p_abnormal": float(probs[1]),
        "cam_path": cam_path,
        "cam_display_slice_index": cam_display_slice_index,
        "cam_center_slice_index": cam_center_slice_index,
        "cam_selection_method": cam_selection_method,
        "cam_error": cam_error,
        "cam_display_orientation": cam_display_orientation,
    }


def run_ct_from_npz(npz_path, checkpoint=None, cam_dir=None, threshold=None):
    """Run CT inference directly from an NPZ file path. Returns same dict shape as run_ct_for_patient.

    Supports:
    - RSNA volumetric NPZ with key ``arr`` (k, C, H, W) raw → ImageNet norm applied here.
    - Single-slice NPZ with key ``image`` (3, H, W) already ImageNet-normalized (see scripts/data/03_preprocess_ct.py).
    """
    from src.config import OUTPUTS

    if checkpoint is None:
        checkpoint = OUTPUTS / "ct_baseline_best.pt"
    else:
        checkpoint = Path(checkpoint)

    arr_path = Path(npz_path)
    if not arr_path.exists():
        return None

    model, _model_kind, agg, k_slices_meta, ckpt_meta = _load_model(checkpoint)
    eval_threshold = resolve_abnormal_threshold(cli=threshold, default=0.5, ckpt=ckpt_meta)
    expected_k = int(k_slices_meta) if k_slices_meta is not None else None
    device = next(model.parameters()).device

    data = np.load(arr_path)
    if "arr" in data.files:
        arr = _load_patient_arr(arr_path, expected_k=expected_k)
        x_raw = torch.from_numpy(arr).float().to(device)
        x_all = _imagenet_normalize_volume(x_raw)
        patient_id = arr_path.parent.name
    elif "image" in data.files:
        img = torch.from_numpy(np.asarray(data["image"], dtype=np.float32)).to(device)
        if img.dim() != 3:
            raise ValueError(f"NPZ 'image' must be (3,H,W), got {tuple(img.shape)}")
        x_all = img.unsqueeze(0)
        x_raw = _denormalize_volume(x_all)
        patient_id = arr_path.stem
    else:
        raise ValueError(
            "NPZ must contain 'arr' (RSNA volume) or 'image' (single-slice normalized); "
            f"got keys: {list(data.files)}"
        )

    x_batch = x_all.unsqueeze(0)
    thick = _thickness_from_npz(arr_path, device)
    out = _run_ct_core(
        model,
        agg,
        eval_threshold,
        x_batch,
        x_raw,
        x_all,
        thick,
        Path(cam_dir) if cam_dir is not None else None,
        arr_path.stem,
    )
    out["patient_id"] = patient_id
    return out


def run_ct_from_image(
    image_path,
    checkpoint=None,
    device=None,
    cam_dir=None,
    threshold=None,
):
    """
    Run CT inference on a single 2D JPEG or PNG.
    PNG/JPG preprocessing mirrors backend/predict_mri.py::_preprocess_for_classification:
        grayscale → 3-channel stack → Resize(224, 224) → ImageNet Normalize.
    Returns the same dict shape as run_ct_from_npz.
    """
    import cv2
    from torchvision import transforms

    from src.config import OUTPUTS

    image_path = Path(image_path)
    if not image_path.exists():
        return None

    ckpt_path = Path(checkpoint) if checkpoint is not None else OUTPUTS / "ct_baseline_best.pt"

    model, _model_kind, agg, _k_slices_meta, ckpt_meta = _load_model(ckpt_path)
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        model = model.to(dev)
    eval_threshold = resolve_abnormal_threshold(cli=threshold, default=0.5, ckpt=ckpt_meta)
    model_device = next(model.parameters()).device

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read CT image: {image_path}")
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    img_3ch = np.stack([img, img, img], axis=0)
    tensor = torch.tensor(img_3ch, dtype=torch.float32)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x_all = transform(tensor).unsqueeze(0).to(model_device)
    x_raw = _denormalize_volume(x_all)
    x_batch = x_all.unsqueeze(0)
    thick = torch.zeros(1, 1, dtype=torch.float32, device=model_device)

    stem = image_path.stem
    out = _run_ct_core(
        model,
        agg,
        eval_threshold,
        x_batch,
        x_raw,
        x_all,
        thick,
        Path(cam_dir) if cam_dir is not None else None,
        stem,
    )
    out["patient_id"] = stem
    return out


if __name__ == "__main__":
    # CLI flags control demo behavior: single vs batch, checkpoint, CAM options.
    ap = argparse.ArgumentParser(description="Run CT inference (single patient or batch from manifest).")
    ap.add_argument("--patient-id", type=str, default=None)
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--random-n", type=int, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--cam-dir", type=Path, default=None, help="Save CAM overlays here (optional)")
    ap.add_argument("--cam-limit", type=int, default=None, help="Max CAMs in batch (optional)")
    ap.add_argument("--cam-when-abnormal", action="store_true", help="Only CAM for pred=abnormal in batch")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="P(abnormal) cutoff; else env ABNORMAL_THRESHOLD, else checkpoint eval_threshold, else 0.5",
    )
    args = ap.parse_args()

    main(
        patient_id=args.patient_id,
        npz_path=args.npz,
        checkpoint=args.checkpoint,
        split=args.split,
        limit=args.limit,
        random_n=args.random_n,
        out_csv=args.out_csv,
        cam_dir=args.cam_dir,
        cam_limit=args.cam_limit,
        cam_when_abnormal=args.cam_when_abnormal,
        threshold_cli=args.threshold,
    )


