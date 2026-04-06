#!/usr/bin/env python3
"""
Offline / research CLI for CT explainability.

Backend (GradCAM only): ``from ml.brain.ct.ct_explainability import explain_prediction``
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.brain.ct.ct_explainability import explain_prediction, main  # noqa: E402

__all__ = ["explain_prediction", "main"]

if __name__ == "__main__":
    main()
