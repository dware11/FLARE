"""
RSNA CT preprocessing entry point (wraps scripts/preprocess_ct.py).

Run from repo root, e.g.:
  PYTHONUNBUFFERED=1 python scripts/preprocess_rsna.py --k-slices 21 \\
    --rsna-manifest /scratch/bckk/flare/ct_brain/meta/ct_rsna_manifest.json
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_RSNA_CLI_MARKER = 'choices=("cq500", "rsna")'


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    target = root / "scripts" / "preprocess_ct.py"
    if not target.is_file():
        print(f"preprocess_ct.py not found: {target}", file=sys.stderr)
        sys.exit(1)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Cannot read {target}: {e}", file=sys.stderr)
        sys.exit(1)
    if _RSNA_CLI_MARKER not in text:
        print(
            "This FLARE checkout is missing RSNA support in scripts/preprocess_ct.py.\n"
            "Update the repo on this machine, then retry, e.g.:\n"
            "  cd <path-to-FLARE> && git pull origin main",
            file=sys.stderr,
        )
        sys.exit(2)

    script_name = str(target)
    sys.argv = [script_name, "--dataset-type", "rsna", *sys.argv[1:]]
    runpy.run_path(script_name, run_name="__main__")


if __name__ == "__main__":
    main()
