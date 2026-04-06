"""
CT explainability for DenseNet / sequence models: GradCAM (frontend), GradCAM++, ScoreCAM, LayerCAM.
Import: ``from ml.brain.ct.ct_explainability import explain_prediction`` (no Flask).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.brain.ct.gradcam_ct import GradCAM, _get_layer  # noqa: E402
from ml.brain.ct.infer import _load_model  # noqa: E402
from ml.brain.ct.model_ct import is_ct_middle_slice_resnet18, is_sequence_ct_model, patient_logits_from_model  # noqa: E402


def _default_target_layer(model: nn.Module) -> str:
    if is_sequence_ct_model(model):
        return "backbone.features.denseblock4"
    if is_ct_middle_slice_resnet18(model):
        return "layer4"
    return "features.denseblock4"


def _raw_slice_for_overlay(npz_path: Path, t: int) -> torch.Tensor:
    """Single slice (1,3,H,W) float in ~[0,1] for CAM overlay (denormalized display)."""
    data = np.load(npz_path)
    arr = data["arr"].astype(np.float32)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    sl = torch.from_numpy(arr[t].copy()).float().unsqueeze(0)
    if sl.shape[1] == 1:
        sl = sl.repeat(1, 3, 1, 1)
    return sl.clamp(0.0, 1.0)


def _load_volume_npz(npz_path: Path, device: torch.device) -> torch.Tensor:
    data = np.load(npz_path)
    arr = data["arr"].astype(np.float32)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    x = torch.from_numpy(arr).float().to(device)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def _thickness_from_npz(npz_path: Path, device: torch.device) -> torch.Tensor:
    data = np.load(npz_path)
    if "slice_thickness_mm" in data.files:
        tmm = float(np.asarray(data["slice_thickness_mm"]).reshape(-1)[0])
    else:
        tmm = 0.0
    tn = min(max(tmm, 0.0) / 10.0, 3.0)
    return torch.tensor([[tn]], dtype=torch.float32, device=device)


def compute_slice_abnormal_scores(
    model: nn.Module,
    x: torch.Tensor,
    thickness: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, List[float]]:
    """Per-slice P(abnormal) before study pooling (sequence), per-slice (legacy), or per-slice ResNet."""
    model.eval()
    with torch.no_grad():
        if is_sequence_ct_model(model):
            b, k, c, h, w = x.shape
            flat = x.reshape(b * k, c, h, w)
            feat = model.backbone(flat)
            feat = feat.reshape(b, k, model.feat_dim)
            out, _ = model.rnn(feat)
            out = model.rnn_dropout(out)
            slice_logits = model.slice_head(out)
            p_ab = torch.softmax(slice_logits, dim=-1)[:, :, 1].squeeze(0)
        elif is_ct_middle_slice_resnet18(model):
            b, k, c, h, w = x.shape
            probs = []
            for i in range(k):
                sl = x[:, i]
                logits = model(sl)
                probs.append(torch.softmax(logits, dim=1)[0, 1])
            p_ab = torch.stack(probs, dim=0)
        else:
            b, k, c, h, w = x.shape
            flat = x.reshape(b * k, c, h, w)
            logits = model(flat).view(b, k, -1)
            p_ab = torch.softmax(logits, dim=-1)[:, :, 1].squeeze(0)
    scores = [float(p_ab[i].item()) for i in range(p_ab.shape[0])]
    return p_ab, scores


def _study_probability(
    model: nn.Module,
    x: torch.Tensor,
    agg: str,
    thickness: Optional[torch.Tensor],
) -> float:
    model.eval()
    with torch.no_grad():
        logits = patient_logits_from_model(model, x, agg=agg, thickness=thickness)
        p = torch.softmax(logits, dim=1)[0, 1].item()
    return float(p)


def _cam_to_overlay_base64(
    heatmap: torch.Tensor,
    x_display: torch.Tensor,
) -> str:
    """heatmap (H,W) in [0,1]; x_display (1,3,H,W) roughly [0,1] for visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib required for heatmap PNG encoding")

    hmap = heatmap.detach().cpu().numpy()
    img = x_display.detach().cpu().clamp(0, 1).squeeze(0).numpy()
    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(gray, cmap="gray")
    ax.imshow(hmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _slice_subvolume(x: torch.Tensor, t: int) -> torch.Tensor:
    """x: (1,k,3,H,W) -> (1,1,3,H,W)."""
    return x[:, t : t + 1].contiguous()


def _gradcam_plus_plus(
    model: nn.Module,
    layer: nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None
    bwd_handle = None

    def fwd_hook(_m, _inp, out):
        nonlocal activations, bwd_handle
        activations = out

        def bwd_hook(grad: torch.Tensor):
            nonlocal gradients
            gradients = grad

        bwd_handle = out.register_hook(bwd_hook)

    h = layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            logits[0, target_class].backward()
        if activations is None or gradients is None:
            return torch.zeros(1, device=x.device)
        a = activations.detach()
        g = gradients.detach()
        eps = 1e-8
        alpha_num = g.pow(2)
        alpha_denom = 2.0 * g.pow(2) + (a * g.pow(3)).sum(dim=(2, 3), keepdim=True) + eps
        alpha = alpha_num / alpha_denom
        relu_g = F.relu(g)
        weights = (alpha * relu_g).sum(dim=(2, 3), keepdim=True)
        cam = (weights * a).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
    finally:
        h.remove()
        if bwd_handle is not None:
            bwd_handle.remove()


def _layer_cam(
    model: nn.Module,
    layer: nn.Module,
    x: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    activations: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None
    bwd_handle = None

    def fwd_hook(_m, _inp, out):
        nonlocal activations, bwd_handle
        activations = out

        def bwd_hook(grad: torch.Tensor):
            nonlocal gradients
            gradients = grad

        bwd_handle = out.register_hook(bwd_hook)

    h = layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            logits[0, target_class].backward()
        if activations is None or gradients is None:
            return torch.zeros(1, device=x.device)
        cam = (F.relu(gradients * activations)).sum(dim=1).squeeze(0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
    finally:
        h.remove()
        if bwd_handle is not None:
            bwd_handle.remove()


def _score_cam_lite(
    model: nn.Module,
    layer: nn.Module,
    x: torch.Tensor,
    target_class: int,
    max_channels: int = 48,
) -> torch.Tensor:
    """Lightweight channel-ablation style saliency (ScoreCAM-like; many forward passes)."""
    device = x.device
    acts_buf: List[torch.Tensor] = []

    def cap(_m, _inp, out):
        acts_buf.append(out.detach())

    h = layer.register_forward_hook(cap)
    model.eval()
    with torch.no_grad():
        base_logits = model(x)
    h.remove()
    if not acts_buf:
        return torch.zeros(1, device=device)
    A = acts_buf[0]
    _, c, h0, w0 = A.shape
    base_score = base_logits[0, target_class].item()
    idx = torch.linspace(0, c - 1, steps=min(c, max_channels)).long().to(device)
    acc = torch.zeros(h0, w0, device=device)
    for i in idx:
        mask = F.relu(A[0, int(i)])
        if mask.max() <= 0:
            continue
        mask = mask / (mask.max() + 1e-8)
        mask_up = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=x.shape[-2:], mode="bilinear", align_corners=False)
        x_pert = x * (0.3 + 0.7 * mask_up)
        with torch.no_grad():
            s = model(x_pert)[0, target_class].item() - base_score
        acc += s * F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False).squeeze()
    acc = F.relu(acc)
    if acc.max() > 0:
        acc = acc / acc.max()
    return acc


def explain_prediction(
    npz_path: Path | str,
    checkpoint_path: Path | str,
    method: str = "gradcam",
) -> Dict[str, Any]:
    """
    Frontend entry point: GradCAM overlay + slice importance only when method=='gradcam'.
    Other methods should be used from CLI / research code only.
    """
    npz_path = Path(npz_path)
    checkpoint_path = Path(checkpoint_path)
    method_l = (method or "gradcam").lower().replace("+", "plus")
    if method_l in ("gradcam++", "gradcamplusplus"):
        method_l = "gradcam++"

    if method_l != "gradcam":
        raise ValueError("explain_prediction() serves GradCAM to the frontend only; use run_explain_cli offline.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _kind, agg, _k_meta, ckpt_meta = _load_model(checkpoint_path)
    model = model.to(device)
    model.eval()

    x = _load_volume_npz(npz_path, device)
    thick = _thickness_from_npz(npz_path, device) if is_sequence_ct_model(model) and model.thickness_fusion is not None else None
    if thick is None and is_sequence_ct_model(model):
        thick = None

    _, slice_scores = compute_slice_abnormal_scores(model, x, thick)
    k = len(slice_scores)
    top_k = min(3, k)
    top_slices = sorted(range(k), key=lambda i: slice_scores[i], reverse=True)[:top_k]

    target_class = 1
    t_vis = top_slices[0]
    x_sub = _slice_subvolume(x, t_vis)
    layer_name = _default_target_layer(model)
    overlay_raw = _raw_slice_for_overlay(npz_path, t_vis)
    gradcam = GradCAM(model, layer_name)
    heatmap, overlay_np = gradcam(
        x_sub,
        target_class=target_class,
        input_for_overlay=overlay_raw,
    )
    if overlay_np is None:
        heatmap_b64 = _cam_to_overlay_base64(heatmap.cpu(), overlay_raw.cpu().clamp(0, 1))
    else:
        from PIL import Image

        pil = Image.fromarray(overlay_np)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    pred = _study_probability(model, x, agg, thick)
    return {
        "heatmap_base64": heatmap_b64,
        "slice_scores": slice_scores,
        "top_slices": top_slices,
        "prediction": pred,
        "method_used": "gradcam",
    }


def run_explain_offline(
    input_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    method: str,
    layer: Optional[str],
    top_k: int,
    visualize_layers: bool,
) -> None:
    method_norm = (method or "gradcam").lower().replace("+", "plus")
    if method_norm in ("gradcamplusplus",):
        method_norm = "gradcam++"
    if method_norm not in ("gradcam", "gradcam++", "scorecam", "layercam"):
        raise ValueError(f"Unknown method {method!r}; use gradcam, gradcam++, scorecam, layercam")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _kind, agg, _k_meta, _meta = _load_model(checkpoint_path)
    model = model.to(device)

    if input_path.suffix.lower() == ".npz":
        x = _load_volume_npz(input_path, device)
    else:
        raise ValueError("Offline CLI expects .npz volume; use preprocess cache.")

    thick = _thickness_from_npz(input_path, device) if is_sequence_ct_model(model) and model.thickness_fusion is not None else None
    pred = _study_probability(model, x, agg, thick)
    _, slice_scores = compute_slice_abnormal_scores(model, x, thick)
    k = len(slice_scores)
    order = sorted(range(k), key=lambda i: slice_scores[i], reverse=True)[: max(1, min(top_k, k))]

    layer_name = layer or _default_target_layer(model)
    layer_mod = _get_layer(model, layer_name)
    target_class = 1

    method_l = method_norm

    output_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "method": method_l,
        "layer": layer_name,
        "prediction_abnormal_prob": pred,
        "slice_scores": slice_scores,
        "top_slices": order,
    }

    for rank, t in enumerate(order):
        x_sub = _slice_subvolume(x, t)
        if method_l == "gradcam":
            gradcam = GradCAM(model, layer_name)
            heatmap, overlay = gradcam(
                x_sub,
                target_class=target_class,
                input_for_overlay=x[0, t : t + 1].detach().cpu().clamp(0, 1),
            )
            cam2d = heatmap
        elif method_l == "gradcam++":
            cam2d = _gradcam_plus_plus(model, layer_mod, x_sub, target_class)
        elif method_l == "layercam":
            cam2d = _layer_cam(model, layer_mod, x_sub, target_class)
        elif method_l == "scorecam":
            cam2d = _score_cam_lite(model, layer_mod, x_sub, target_class)
        else:
            raise ValueError(f"Unknown method {method}")

        up = F.interpolate(cam2d.unsqueeze(0).unsqueeze(0), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze()
        disp = _raw_slice_for_overlay(input_path, t).cpu().clamp(0, 1)
        png_b64 = _cam_to_overlay_base64(up.cpu(), disp)
        raw = base64.b64decode(png_b64)
        out_png = output_dir / f"slice_{t}_rank{rank}_{method_l}.png"
        out_png.write_bytes(raw)

    if visualize_layers and is_sequence_ct_model(model):
        _save_denseblock_grid(model.backbone, x_sub, output_dir / "denseblock_activations.png", device)

    with open(output_dir / "explain_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4))
        if len(order) == 1:
            axes = [axes]
        for ax, t, rank in zip(axes, order, range(len(order))):
            p = output_dir / f"slice_{t}_rank{rank}_{method_l}.png"
            if p.is_file():
                im = plt.imread(p)
                ax.imshow(im)
            ax.set_title(f"slice {t}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"summary_top{len(order)}.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _save_denseblock_grid(
    backbone: nn.Module,
    x_sub: torch.Tensor,
    out_path: Path,
    device: torch.device,
) -> None:
    """Mean absolute activation map per DenseBlock (paper figure)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    feats: List[torch.Tensor] = []
    seen: set[str] = set()

    def make_hook(name: str):
        def hook(_m, _inp, out):
            if name in seen:
                return
            seen.add(name)
            feats.append(out.detach().abs().mean(dim=1).squeeze(0).cpu())

        return hook

    hooks = []
    fb = getattr(backbone, "features", None)
    if fb is None:
        return
    for attr in ("denseblock1", "denseblock2", "denseblock3", "denseblock4"):
        if hasattr(fb, attr):
            m = getattr(fb, attr)
            hooks.append(m.register_forward_hook(make_hook(attr)))

    backbone.eval()
    with torch.no_grad():
        flat = x_sub.reshape(-1, *x_sub.shape[2:])
        _ = backbone(flat)
    for h in hooks:
        h.remove()

    if not feats:
        return
    nmaps = len(feats)
    fig, axes = plt.subplots(1, nmaps, figsize=(3 * nmaps, 3))
    if nmaps == 1:
        axes = [axes]
    for ax, f in zip(axes, feats):
        g = f.numpy()
        ax.imshow(g, cmap="viridis")
        ax.axis("off")
    fig.suptitle("DenseNet121 backbone — mean |activation| per block (subvolume)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="CT explainability (offline research + PNG export).")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layer", type=str, default=None)
    ap.add_argument(
        "--method",
        type=str,
        default="gradcam",
        help="gradcam | gradcam++ | scorecam | layercam",
    )
    ap.add_argument("--top-k", type=int, default=3, dest="top_k")
    ap.add_argument("--visualize-layers", action="store_true", dest="visualize_layers")
    args = ap.parse_args()
    run_explain_offline(
        args.input,
        args.checkpoint,
        args.output_dir,
        args.method,
        args.layer,
        args.top_k,
        args.visualize_layers,
    )
    print(f"Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
