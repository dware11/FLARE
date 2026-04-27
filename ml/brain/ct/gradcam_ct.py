"""
Demo: Hook model.layer4, compute Grad-CAM heatmap + overlay.
Design: enable_grad required so we can backward on target class for gradients.
"""

from typing import Optional, Tuple
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


class GradCAM:
    """
    Grad-CAM on DenseNet last conv block. For the sequence (k-slice) model, pass the
    full (B, k, C, H, W) volume so the backward matches study-level prediction; use
    center-slice (k//2) activations/gradients for the displayed heatmap, aligned with
    input_for_overlay (same axial index in the preprocessed stack).
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
        # Design: backward hook captures gradients w.r.t. activations for CAM weights.
        def _backward_hook(grad: torch.Tensor):
            self.gradients = grad

        self._bwd_handle = out.register_hook(_backward_hook)

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        input_for_overlay: Optional[torch.Tensor] = None,
        thickness: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Compute Grad-CAM.
        - Sequence model: x is (1, k, C, H, W); pass thickness if the model uses it
          (must match the prediction forward). CAM uses slice index k//2 to match
          input_for_overlay (one axial slice, ~[0,1] raw).
        - Legacy 2D path: x is (1, C, H, W); thickness ignored.
        """
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.layer.register_forward_hook(self._forward_hook)
        h, w = _spatial_hw(x, input_for_overlay)
        k_stack = int(x.shape[1]) if x.dim() == 5 else 1
        k_slice_for_cam = k_stack // 2

        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                if x.dim() == 5 and is_sequence_ct_model(self.model):
                    logits = self.model(x, thickness=thickness)
                elif x.dim() == 5:
                    # Legacy 2D model should not receive 5D here; if it does, fail clearly.
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
            # Backbone processes (B*k) slices: take center slice to match on-screen overlay.
            if a.dim() == 4 and a.shape[0] > 1 and k_stack > 1:
                if a.shape[0] == k_stack and g.shape[0] == k_stack:
                    a = a[k_slice_for_cam : k_slice_for_cam + 1]
                    g = g[k_slice_for_cam : k_slice_for_cam + 1]
            # Compute CAM
            weights = g.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * a).sum(dim=1)  # (1, H, W)
            cam = F.relu(cam).squeeze(0)  # (H, W)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

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
        Blend the heatmap with input for visualization.
        x: (1, 3, H, W) or (1, 1, 3, H, W) (first spatial slice used for 5D).
        Grayscale: robust display (0–1 or 0–255 raw + percentile stretch) so the
        underlay is visible next to the JET colormap.
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
