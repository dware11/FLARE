"""Run exam-level inference; write CSV: exam_id, patient_id, label, p_mri."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.breast.mri.config_mri import (
    BREAST_MRI_CAM_DIR,
    BREAST_MRI_MANIFEST_PATH,
    BREAST_MRI_PRED_CSV,
    MRI_NUM_CHANNELS,
    ensure_mri_dirs,
)
from ml.breast.mri.dataset_mri import BreastMRIDataset
from ml.breast.mri.model_mri import build_mri_model


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "unknown"


def _parse_layers(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _reduce_activation(acts: torch.Tensor, reduction: str) -> np.ndarray:
    # Expected shape is (B, C, D, H, W); take first item in batch.
    v = acts[0]
    if reduction == "max":
        v = v.max(dim=0).values
    else:
        v = v.mean(dim=0)
    return v.detach().cpu().numpy().astype(np.float32)


def _to_2d_slice(vol: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mip":
        return vol.max(axis=0)
    return vol[vol.shape[0] // 2]


def _save_activation_png(
    image_2d: np.ndarray, out_path: Path, title: str, cmap: str = "magma"
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = image_2d - image_2d.min()
    mx = float(im.max())
    if mx > 0:
        im = im / mx
    plt.figure(figsize=(6, 6))
    plt.imshow(im, cmap=cmap)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=BREAST_MRI_MANIFEST_PATH)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument(
        "--save-activations",
        action="store_true",
        help="Save intermediate activation visualizations per exam.",
    )
    ap.add_argument(
        "--activation-layers",
        type=str,
        default="enc1,enc2,enc3,enc4",
        help="Comma-separated module names on model (e.g. enc2,enc4).",
    )
    ap.add_argument(
        "--activation-reduction",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Channel reduction used to produce a 3D activation volume.",
    )
    ap.add_argument(
        "--activation-view",
        type=str,
        default="midz",
        choices=["midz", "mip"],
        help="Render mid-z slice or max-intensity-projection over depth.",
    )
    ap.add_argument(
        "--activation-out-dir",
        type=Path,
        default=BREAST_MRI_CAM_DIR / "activations",
    )
    args = ap.parse_args()

    out = args.out_csv or BREAST_MRI_PRED_CSV
    ensure_mri_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_ch = int(ckpt.get("in_channels", MRI_NUM_CHANNELS))
    model = build_mri_model(in_channels=in_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = BreastMRIDataset(args.manifest, split="all", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    layer_names = _parse_layers(args.activation_layers)
    hooks = []
    captured: Dict[str, torch.Tensor] = {}
    if args.save_activations:
        for layer_name in layer_names:
            layer = getattr(model, layer_name, None)
            if layer is None:
                print(f"[warn] skipping unknown layer '{layer_name}'")
                continue

            def _hook(_m, _inp, out, name=layer_name):
                captured[name] = out.detach().cpu()

            hooks.append(layer.register_forward_hook(_hook))

    rows: list[dict] = []
    try:
        with torch.no_grad():
            for x, y, eid, pid, _ in loader:
                x = x.to(device)
                logits = model(x)
                p = torch.sigmoid(logits).view(-1).cpu().numpy()
                for i in range(x.size(0)):
                    exam_id = str(eid[i])
                    patient_id = str(pid[i])
                    rows.append(
                        {
                            "exam_id": exam_id,
                            "patient_id": patient_id,
                            "label": float(y[i].item()),
                            "p_mri": float(p[i]),
                        }
                    )

                    if not args.save_activations:
                        continue

                    exam_safe = _safe_name(exam_id)
                    patient_safe = _safe_name(patient_id)
                    for layer_name, batch_acts in captured.items():
                        if batch_acts.ndim != 5 or i >= batch_acts.size(0):
                            continue
                        acts = batch_acts[i : i + 1]
                        vol = _reduce_activation(acts, args.activation_reduction)
                        img = _to_2d_slice(vol, args.activation_view)
                        out_path = (
                            args.activation_out_dir
                            / layer_name
                            / f"{patient_safe}_{exam_safe}_{layer_name}_{args.activation_view}.png"
                        )
                        _save_activation_png(
                            img,
                            out_path,
                            title=(
                                f"{layer_name} | {args.activation_reduction} | "
                                f"{args.activation_view}"
                            ),
                        )
    finally:
        for h in hooks:
            h.remove()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exam_id", "patient_id", "label", "p_mri"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    if args.save_activations:
        print(f"Saved activation maps to {args.activation_out_dir}")


if __name__ == "__main__":
    main()
