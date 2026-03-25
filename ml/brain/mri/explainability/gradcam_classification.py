from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = None
        self._bwd_handle = None

    def _forward_hook(self, _module, _inp, out):
        self.activations = out

        def _backward_hook(grad):
            self.gradients = grad

        self._bwd_handle = out.register_hook(_backward_hook)

    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)

        try:
            self.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                logits = self.model(x)
                if target_class is None:
                    target_class = int(logits.argmax(dim=1).item())

                score = logits[0, target_class]
                score.backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

            a = self.activations.detach()   # (B, C, H, W)
            g = self.gradients.detach()     # (B, C, H, W)

            weights = g.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
            cam = (weights * a).sum(dim=1)               # (B, H, W)
            cam = F.relu(cam)

            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()

            cam = cam[0]
            return cam, target_class

        finally:
            if self._fwd_handle is not None:
                self._fwd_handle.remove()
                self._fwd_handle = None
            if self._bwd_handle is not None:
                self._bwd_handle.remove()
                self._bwd_handle = None


def make_overlay(image_2d: np.ndarray, heatmap_2d: np.ndarray) -> np.ndarray:
    image_2d = np.clip(image_2d, 0, 1)
    heatmap_2d = np.clip(heatmap_2d, 0, 1)

    gray = (image_2d * 255).astype(np.uint8)

    try:
        import cv2
        heat_uint8 = (heatmap_2d * 255).astype(np.uint8)
        heat_colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * gray[:, :, None] + 0.5 * heat_rgb).astype(np.uint8)
        return overlay
    except ImportError:
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        return gray_rgb
