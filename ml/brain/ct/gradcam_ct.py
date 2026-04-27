"""
CT Grad-CAM: saliency over DenseNet feature maps. This is attention for explanation,
not a tumor mask or automatic segmentation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ml.brain.ct.model_ct import is_sequence_ct_model


def _spatial_hw(
    x: torch.Tensor, input_for_overlay: Optional[torch.Tensor] = None
) -> Tuple[int, int]:
    """(H, W) for heatmap/overlay; prefer overlay tensor when given."""
    ref = input_for_overlay if input_for_overlay is not None else x
    if ref.dim() < 2:
        return 256, 256
    return int(ref.shape[-2]), int(ref.shape[-1])


def _get_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    """Resolve nested layer name (ex. 'features.denseblock4') for Grad-Cam"""
    if "." not in layer_name:
        return getattr(model, layer_name)

    obj = model
    for part in layer_name.split("."):
        obj = getattr(obj, part)
    return obj


def _top5_mean_activation_score(cam_hw: torch.Tensor) -> float:
    """
    After per-slice 0-1 CAM normalization, score slice saliency as the mean of the
    largest 5% of (nonnegative) values — smoother than a single max pixel.
    """
    v = F.relu(cam_hw).detach().float().view(-1)
    if v.numel() == 0:
        return 0.0
    n = v.numel()
    k = max(1, int(0.05 * n + 0.999))
    k = min(k, n)
    topk_vals, _ = torch.topk(v, k)
    return float(topk_vals.mean().item())


def _cam_hw_from_ag(
    a1: torch.Tensor, g1: torch.Tensor, out_h: int, out_w: int
) -> torch.Tensor:
    """Single-slice (1,C,h,w) activations+grads -> (H,W) in [0,1]."""
    weights = g1.mean(dim=(2, 3), keepdim=True)
    cam = (weights * a1).sum(dim=1)
    cam = F.relu(cam).squeeze(0)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    return F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


class GradCAM:
    """
    Grad-CAM on the last DenseNet conv block. For k-slice sequence models, one
    full-volume forward+backward (same as prediction) yields (k, C, h, w) hooks;
    we compute a CAM per slice and display the slice with the strongest
    top-5%-mean saliency on the corresponding raw slice.
    """

    def __init__(self, model: torch.nn.Module, layer_name: str = "features.denseblock4"):
        self.model = model
        self.layer = _get_layer(model, layer_name)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._bwd_handle = None

    def _forward_hook(self, _module: torch.nn.Module, _inp: tuple, out: torch.Tensor):
        self.activations = out

        def _backward_hook(grad: torch.Tensor):
            self.gradients = grad

        self._bwd_handle = out.register_hook(_backward_hook)

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        input_for_overlay: Optional[torch.Tensor] = None,
        thickness: Optional[torch.Tensor] = None,
        x_raw_volume: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, Optional[np.ndarray]],
        Tuple[torch.Tensor, Optional[np.ndarray], Dict[str, Any]],
    ]:
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.layer.register_forward_hook(self._forward_hook)
        k_stack = int(x.shape[1]) if x.dim() == 5 else 1
        k_mid = k_stack // 2
        h, w = _spatial_hw(x, input_for_overlay)
        if x_raw_volume is not None and k_stack > 1:
            h, w = int(x_raw_volume.shape[2]), int(x_raw_volume.shape[3])

        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                if x.dim() == 5 and is_sequence_ct_model(self.model):
                    logits = self.model(x, thickness=thickness)
                elif x.dim() == 5 and k_stack == 1:
                    logits = self.model(x.squeeze(1))
                elif x.dim() == 5:
                    raise ValueError("5D input requires a sequence CT model for Grad-CAM")
                else:
                    logits = self.model(x)
                if target_class is None:
                    target_class = int(logits.argmax(dim=1).item())
                logits[0, target_class].backward()

            if self.activations is None or self.gradients is None:
                return torch.zeros(h, w, device=x.device), None

            a = self.activations.detach()
            g = self.gradients.detach()
            use_multislice = (
                x_raw_volume is not None
                and k_stack > 1
                and a.dim() == 4
                and g.dim() == 4
                and a.shape[0] == k_stack
                and g.shape[0] == k_stack
            )

            if use_multislice:
                heatmaps: List[torch.Tensor] = []
                scores: List[float] = []
                for i in range(k_stack):
                    hi = _cam_hw_from_ag(a[i : i + 1], g[i : i + 1], h, w)
                    heatmaps.append(hi)
                    scores.append(_top5_mean_activation_score(hi))
                best_idx = int(max(range(k_stack), key=lambda j: scores[j]))
                heatmap = heatmaps[best_idx]
                raw_slice = x_raw_volume[best_idx : best_idx + 1]
                overlay = self._overlay(heatmap, raw_slice)
                meta: Dict[str, Any] = {
                    "cam_display_slice_index": best_idx,
                    "cam_center_slice_index": k_mid,
                    "cam_selection_method": "top5_mean_activation",
                    "heatmaps_per_slice": heatmaps,
                    "top5_mean_per_slice": scores,
                }
                return heatmap, overlay, meta

            if a.dim() == 4 and a.shape[0] > 1 and k_stack > 1:
                if a.shape[0] == k_stack and g.shape[0] == k_stack:
                    a = a[k_mid : k_mid + 1]
                    g = g[k_mid : k_mid + 1]
            weights = g.mean(dim=(2, 3), keepdim=True)
            cam = (weights * a).sum(dim=1)
            cam = F.relu(cam).squeeze(0)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / (cam.max() + 1e-8)
            heatmap = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            overlay = None
            if input_for_overlay is not None:
                overlay = self._overlay(heatmap, input_for_overlay)

            return heatmap, overlay

        finally:
            if self._fwd_handle is not None:
                self._fwd_handle.remove()
                self._fwd_handle = None
            if self._bwd_handle is not None:
                self._bwd_handle.remove()
                self._bwd_handle = None

    def _overlay(self, heatmap: torch.Tensor, x: torch.Tensor) -> np.ndarray:
        """
        Heatmap on CT for visualization only; not a pathologic segmentation.
        x: (1, 3, H, W) or (1, 1, 3, H, W).
        """
        heat = heatmap.cpu().numpy()
        t = x
        if t.dim() == 5:
            t = t[:, 0]
        img = t.squeeze(0).cpu().numpy()
        gray = img[0].astype(np.float32)
        if gray.max() > 1.5:
            gray = np.clip(gray / 255.0, 0.0, 1.0)
        else:
            gray = np.clip(gray, 0.0, 1.0)
        p_lo, p_hi = float(np.percentile(gray, 2.0)), float(np.percentile(gray, 98.0))
        if p_hi - p_lo > 1e-6:
            gray = np.clip((gray - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0)
        gray_u8 = (gray * 255.0).astype(np.uint8)

        try:
            import cv2
            heat_uint8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
            heat_colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
            overlay = (0.5 * gray_u8[:, :, np.newaxis] + 0.5 * heat_rgb).astype(np.uint8)
            return overlay
        except ImportError:
            return np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.uint8)
