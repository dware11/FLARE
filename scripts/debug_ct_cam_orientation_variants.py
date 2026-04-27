"""
CT Grad-CAM orientation diagnostic (no training, no API changes).

Saves the Grad-CAM overlay (and optional reference CT PNG) in several geometric
variants so you can tell whether a "wrong" heatmap region is a display/orientation
mismatch (e.g. MRI saved with np.rot90 vs DICOM-as-is CT) rather than a model bug.

Default output (override with --out-dir):
  /scratch/bckk/flare/projects/FLARE/artifacts/ct_orientation_debug/

Example (P0001 CT Grad-CAM, default out-dir):

  python scripts/debug_ct_cam_orientation_variants.py \\
    --cam backend/static/cam/P0001_ct.png \\
    --out-dir /scratch/bckk/flare/projects/FLARE/artifacts/ct_orientation_debug/

  # optional reference slice for the contact sheet:
  python scripts/debug_ct_cam_orientation_variants.py \\
    --cam backend/static/cam/P0001_ct.png \\
    --ref backend/static/uploads/P0001_ct.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

DEFAULT_OUT = Path("/scratch/bckk/flare/projects/FLARE/artifacts/ct_orientation_debug")


def _load_rgba(path: Path) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install Pillow: pip install pillow") from e
    im = Image.open(path).convert("RGBA")
    return np.array(im, dtype=np.uint8)


def _save_rgba(path: Path, arr: np.ndarray) -> None:
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install Pillow: pip install pillow") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGBA").save(str(path))


def _variant_ops() -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    def rot90_cw(x: np.ndarray) -> np.ndarray:
        return np.rot90(x, k=-1, axes=(0, 1))

    def rot90_ccw(x: np.ndarray) -> np.ndarray:
        return np.rot90(x, k=1, axes=(0, 1))

    def rot180(x: np.ndarray) -> np.ndarray:
        return np.rot90(x, k=2, axes=(0, 1))

    def flip_lr(x: np.ndarray) -> np.ndarray:
        return np.fliplr(x)

    def flip_ud(x: np.ndarray) -> np.ndarray:
        return np.flipud(x)

    def transpose(x: np.ndarray) -> np.ndarray:
        return np.transpose(x, (1, 0, 2))

    def transpose_flip_lr(x: np.ndarray) -> np.ndarray:
        t = np.transpose(x, (1, 0, 2))
        return np.fliplr(t)

    return [
        ("original", lambda x: x.copy()),
        ("rot90_cw", rot90_cw),
        ("rot90_ccw", rot90_ccw),
        ("rot180", rot180),
        ("flip_lr", flip_lr),
        ("flip_ud", flip_ud),
        ("transpose", transpose),
        ("transpose_flip_lr", transpose_flip_lr),
    ]


def _build_panels(
    cam: np.ndarray, ref: Optional[np.ndarray]
) -> List[Tuple[str, np.ndarray]]:
    """
    If ref is given, first panel is reference (original orientation only);
    then all CAM orientation variants.
    """
    out: List[Tuple[str, np.ndarray]] = []
    if ref is not None:
        if ref.shape[:2] != cam.shape[:2]:
            from PIL import Image  # type: ignore

            h, w = cam.shape[0], cam.shape[1]
            try:
                resample = Image.Resampling.BILINEAR  # Pillow >= 9.1
            except AttributeError:
                resample = Image.BILINEAR
            ref = np.array(
                Image.fromarray(ref).resize((w, h), resample=resample)
            )
        out.append(("reference_ct", ref))
    for name, fn in _variant_ops():
        out.append((f"cam_{name}" if ref is not None else name, fn(cam)))
    return out


def _write_contact_sheet(
    panels: List[Tuple[str, np.ndarray]], out_path: Path, title: str
) -> None:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install matplotlib for contact sheet: pip install matplotlib") from e

    n = len(panels)
    ncols = 3 if n > 4 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    fig.suptitle(title, fontsize=12, y=0.99)
    ax_flat = np.atleast_1d(axes).ravel()
    for i, (label, arr) in enumerate(panels):
        ax = ax_flat[i]
        ax.imshow(arr)
        ax.axis("off")
        ax.set_title(label, fontsize=8, color="#222", pad=2)
    for j in range(n, len(ax_flat)):
        ax_flat[j].axis("off")
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cam",
        type=Path,
        required=True,
        help="Path to CT Grad-CAM overlay PNG (e.g. backend/static/cam/P0010_ct.png).",
    )
    p.add_argument(
        "--ref",
        type=Path,
        default=None,
        help="Optional reference CT slice PNG (same FOV helps comparison).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="File name prefix; default is stem of --cam.",
    )
    args = p.parse_args()

    cam_path = args.cam
    if not cam_path.is_file():
        print(f"ERROR: --cam not found: {cam_path}", file=sys.stderr)
        return 1

    ref: Optional[np.ndarray] = None
    if args.ref is not None:
        if not args.ref.is_file():
            print(f"ERROR: --ref not found: {args.ref}", file=sys.stderr)
            return 1
        ref = _load_rgba(args.ref)

    cam = _load_rgba(cam_path)
    prefix = args.prefix or cam_path.stem
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual PNGs: eight CAM orientation variants, plus optional reference copy.
    for name, fn in _variant_ops():
        arr = fn(cam)
        out_file = out_dir / f"{prefix}_{name}.png"
        _save_rgba(out_file, arr)
        print(f"Wrote {out_file}")

    if ref is not None:
        ref_out = out_dir / f"{prefix}_reference_ct.png"
        _save_rgba(ref_out, ref)
        print(f"Wrote {ref_out}")

    panels = _build_panels(cam, ref)
    sheet = out_dir / f"{prefix}_contact_sheet.png"
    _write_contact_sheet(
        panels, sheet, title=f"CT Grad-CAM orientation: {prefix}"
    )
    print(f"Wrote {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
