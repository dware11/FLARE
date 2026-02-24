import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / ".cursor" / "debug.log"

_enabled: bool | None = None


def enable(value: bool = True) -> None:
    global _enabled
    _enabled = value


def is_enabled() -> bool:
    global _enabled
    if _enabled is not None:
        return _enabled
    _enabled = os.environ.get("FLARE_DEBUG", "").lower() in ("1", "true", "yes")
    return _enabled


def dbg(message: str, data: dict | None = None, location: str = "") -> None:
    if not is_enabled():
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"message": message, "data": data or {}, "location": location}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass