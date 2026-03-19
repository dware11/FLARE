"""
3D Grad-CAM for the DBT mammography classifier.

Target layer: model.enc4 (last encoder block before global avg pool).

Produces:
  - 3D heatmap tensor (D, H, W) in [0,1]
  - 2D overlay (H, W, 3) uint8 on the middle axial slice (brain-window equivalent)

Usage:
    from ml.breast.mammo.gradcam_mammo import GradCAM3D
    cam = GradCAM3D(model)
    heatmap_3d, overlay_2d = cam(x)   # x: (1, 1, 32, 256, 256)
"""
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM3D:
    """Minimal 3D Grad-CAM for DBTClassifier.

    Hooks the last encoder block (enc4) to capture (B, 256, 4, 32, 32) activations.
    Computes weighted combination → 3D saliency volume, then renders middle-slice overlay.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.layer = model.enc4
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._bwd_handle = None

    def _forward_hook(self, _m, _inp, out: torch.Tensor) -> None:
        self.activations = out

        def _bwd(grad: torch.Tensor) -> None:
            self.gradients = grad

        self._bwd_handle = out.register_hook(_bwd)

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """Compute 3D Grad-CAM.

        Args:
            x: (1, 1, D, H, W) input tensor
            target_class: 0 or 1; uses predicted class if None

        Returns:
            heatmap: (D, H, W) tensor in [0,1]
            overlay: (H, W, 3) uint8 on middle axial slice, or None
        """
        self.activations = self.gradients = None
        self._fwd_handle = self.layer.register_forward_hook(self._forward_hook)

        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = self.model(x)
                if target_class is None:
                    target_class = int(logits.argmax(dim=1).item())
                logits[0, target_class].backward()

            if self.activations is None or self.gradients is None:
                D, H, W = x.shape[2], x.shape[3], x.shape[4]
                return torch.zeros(D, H, W, device=x.device), None

            a = self.activations.detach()            # (1, C, d, h, w)
            g = self.gradients.detach()
            weights = g.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
            cam = (weights * a).sum(dim=1)           # (1, d, h, w)
            cam = F.relu(cam).squeeze(0)             # (d, h, w)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            D, H, W = x.shape[2], x.shape[3], x.shape[4]
            heatmap = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(D, H, W),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            overlay = self._overlay(heatmap, x)
            return heatmap, overlay

        finally:
            if self._fwd_handle:
                self._fwd_handle.remove()
                self._fwd_handle = None
            if self._bwd_handle:
                self._bwd_handle.remove()
                self._bwd_handle = None

    @staticmethod
    def _overlay(heatmap: torch.Tensor, x: torch.Tensor) -> Optional[np.ndarray]:
        """Overlay Grad-CAM on the middle axial slice.

        Args:
            heatmap: (D, H, W) float tensor in [0,1]
            x: (1, 1, D, H, W)
        Returns:
            (H, W, 3) uint8 RGB overlay
        """
        mid = heatmap.shape[0] // 2
        heat_slice = heatmap[mid].cpu().numpy()         # (H, W)
        img_slice = x[0, 0, mid].cpu().numpy()          # (H, W)
        gray = (np.clip(img_slice, 0, 1) * 255).astype(np.uint8)

        try:
            import cv2
            heat_uint8 = (np.clip(heat_slice, 0, 1) * 255).astype(np.uint8)
            heat_colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
            gray_rgb = np.stack([gray, gray, gray], axis=-1)
            return (0.5 * gray_rgb + 0.5 * heat_rgb).astype(np.uint8)
        except ImportError:
            return np.stack([gray, gray, gray], axis=-1)
