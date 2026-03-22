import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ml.brain.ct_3d.dataset_ct_3d import CTDataset
from ml.brain.ct_3d.model_ct_3d import build_ct_model

# -------- SETTINGS --------
CHECKPOINT = "/scratch/bckk/flare/ct_brain/outputs/ct_3d_best.pt"
OUT_DIR = Path("/scratch/bckk/flare/ct_brain/outputs/cam_examples_3d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD MODEL --------
model = build_ct_model()
ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

# -------- LOAD DATA --------
ds = CTDataset(split="val")

def simple_gradcam(model, x, target_class):
    x.requires_grad = True
    logits = model(x)
    score = logits[0, target_class]
    score.backward()

    grads = x.grad[0].cpu().numpy()
    data = x[0].detach().cpu().numpy()

    cam = np.mean(grads * data, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam

# -------- RUN --------
for i in range(5):
    x, y, pid = ds[i]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    cam = simple_gradcam(model, x.clone(), pred)

    # take middle slice
    mid = cam.shape[0] // 2
    slice_img = x[0, 0, mid].cpu().numpy()
    heatmap = cam[mid]

    plt.figure()
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")

    out_path = OUT_DIR / f"{pid}_cam.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
