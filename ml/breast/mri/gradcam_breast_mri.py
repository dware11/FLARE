"""
3D Grad-CAM for the breast MRI (DCE-MRI) classifier.

Target layer: model.enc4 (last encoder block).
Produces:
  - 3D heatmap (D, H, W) in [0,1]
  - 2D overlay (H, W, 3) uint8 on the middle axial slice of the first DCE channel
"""
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM3D:
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

            a = self.activations.detach()
            g = self.gradients.detach()
            weights = g.mean(dim=(2, 3, 4), keepdim=True)
            cam = F.relu((weights * a).sum(dim=1)).squeeze(0)
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

            return heatmap, self._overlay(heatmap, x)

        finally:
            if self._fwd_handle:
                self._fwd_handle.remove()
                self._fwd_handle = None
            if self._bwd_handle:
                self._bwd_handle.remove()
                self._bwd_handle = None

    @staticmethod
    def _overlay(heatmap: torch.Tensor, x: torch.Tensor) -> Optional[np.ndarray]:
        mid = heatmap.shape[0] // 2
        heat_slice = heatmap[mid].cpu().numpy()
        img_slice = x[0, 0, mid].cpu().numpy()   # first DCE channel (pre-contrast)
        gray = (np.clip(img_slice, 0, 1) * 255).astype(np.uint8)
        try:
            import cv2
            heat_uint8 = (np.clip(heat_slice, 0, 1) * 255).astype(np.uint8)
            heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            return (0.5 * np.stack([gray]*3, axis=-1) + 0.5 * heat_rgb).astype(np.uint8)
        except ImportError:
            return np.stack([gray, gray, gray], axis=-1)
