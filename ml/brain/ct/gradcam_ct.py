"""
Demo: Hook model.layer4, compute Grad-CAM heatmap + overlay.
Design: enable_grad required so we can backward on target class for gradients.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def _spatial_hw(
    x: torch.Tensor, input_for_overlay: Optional[torch.Tensor] = None
) -> Tuple[int, int]:
    """(H, W) for heatmap/overlay; prefer overlay tensor when given."""
    ref = input_for_overlay if input_for_overlay is not None else x
    if ref.dim() < 2:
        return 256, 256
    return int(ref.shape[-2]), int(ref.shape[-1])

def _get_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module: 
    """ Resolve nested layer name (ex. 'features.denseblock4') for Grad-Cam """ 
    if "." not in layer_name: 
        return getattr(model, layer_name) 

    obj = model 
    for part in layer_name.split("."):
        obj =getattr(obj, part) 
    return obj

class GradCAM:
    """
    Minimal Grad-CAM implementation for a 3D CT model.
    Target layer: model.layer4.
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
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Compute Grad-CAM for input x (B=1). Uses predicted class if target_class is None.
        Returns:
            heatmap: tensor [0,1] (H, W) matching input / overlay
            overlay: numpy array (H,W,3) uint8 or None
        """
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.layer.register_forward_hook(self._forward_hook)
        h, w = _spatial_hw(x, input_for_overlay)

        try:
            self.model.zero_grad(set_to_none=True)
            # Note: enable_grad needed so .backward() populates gradients for CAM.
            with torch.enable_grad():
                logits = self.model(x)
                if target_class is None:
                    target_class = int(logits.argmax(dim=1).item())
                logits[0, target_class].backward()

            if self.activations is None or self.gradients is None:
                return torch.zeros(h, w, device=x.device), None

            # Compute CAM
            a = self.activations.detach()           # (1, C, H, W)
            g = self.gradients.detach()             # (1, C, H, W)
            weights = g.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * a).sum(dim=1)          # (1, H, W)
            cam = F.relu(cam).squeeze(0)            # (H, W)
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()

            # Match overlay resolution (e.g. 224 CT vs 256 legacy) to avoid broadcast errors.
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
        Returns: (H, W, 3) uint8 overlay
        """
        heat = heatmap.cpu().numpy()
        t = x
        if t.dim() == 5:
            t = t[:, 0]
        img = t.squeeze(0).cpu().numpy()
        # Design: use first channel (brain window) as grayscale for overlay.
        gray = img[0]
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

        try:
            import cv2
            heat_uint8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
            heat_colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
            overlay = (0.5 * gray[:, :, np.newaxis] + 0.5 * heat_rgb).astype(np.uint8)
            return overlay
        except ImportError:
            # fallback: repeat grayscale across RGB
            return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)