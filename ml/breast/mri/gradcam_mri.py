"""
Grad-CAM on the last conv block of the breast MRI 3D CNN.

Explainability / saliency visualization only — NOT segmentation or lesion delineation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.breast.mri.config_mri import BREAST_MRI_CAM_DIR, MRI_NUM_CHANNELS
from ml.breast.mri.model_mri import build_mri_model


def grad_cam_3d(
    model: torch.nn.Module, x: torch.Tensor, target_layer: torch.nn.Module
) -> Optional[np.ndarray]:
    acts: torch.Tensor | None = None
    grads: torch.Tensor | None = None

    def fwd_hook(_m, _inp, out):
        nonlocal acts
        acts = out.detach()

    def bwd_hook(_m, _gin, gout):
        nonlocal grads
        grads = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        x_in = x.clone().detach().requires_grad_(True)
        logits = model(x_in)
        logits[0, 0].backward()
    finally:
        h1.remove()
        h2.remove()

    if acts is None or grads is None:
        return None

    w = grads.mean(dim=(2, 3, 4), keepdim=True)
    cam = (w * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam[0, 0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.detach().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=BREAST_MRI_CAM_DIR)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    in_ch = int(ckpt.get("in_channels", MRI_NUM_CHANNELS))
    model = build_mri_model(in_channels=in_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Last spatial conv block before GAP (entire enc4 output)
    target = model.enc4

    data = np.load(args.npz, allow_pickle=True)
    vol = torch.from_numpy(data["image"].astype(np.float32)).unsqueeze(0).to(device)

    cam = grad_cam_3d(model, vol, target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if cam is None:
        print("CAM failed")
        return
    z = cam.shape[0] // 2
    plt.imshow(cam[z], cmap="jet")
    plt.colorbar()
    plt.title("Grad-CAM mid slice (explainability only; not segmentation)")
    out_path = args.out_dir / "cam_mid_z.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
