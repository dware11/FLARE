"""Shared abnormal-class probability threshold: CLI > env > default."""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional


def resolve_abnormal_threshold(
    cli: Optional[float] = None,
    default: float = 0.5,
    ckpt: Optional[Mapping[str, Any]] = None,
) -> float:
    """
    Operating threshold on P(abnormal) for binary decisions.
    Priority: explicit CLI value > ABNORMAL_THRESHOLD env > checkpoint eval_threshold > default.
    """
    if cli is not None:
        return float(cli)
    raw = os.environ.get("ABNORMAL_THRESHOLD")
    if raw is not None and str(raw).strip() != "":
        try:
            return float(str(raw).strip())
        except ValueError:
            # Debug: bad env should not silently fall through without a trace.
            print(
                f"[threshold] Ignoring invalid ABNORMAL_THRESHOLD={raw!r}; using default={default}",
                flush=True,
            )
    if ckpt is not None:
        et = ckpt.get("eval_threshold")
        if et is not None:
            try:
                return float(et)
            except (TypeError, ValueError):
                print(
                    f"[threshold] Ignoring invalid checkpoint eval_threshold={et!r}; using default={default}",
                    flush=True,
                )
    return float(default)
