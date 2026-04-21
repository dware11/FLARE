import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4

from flask import g, has_request_context, make_response, request


AUDIT_LOG_DIR = "/scratch/bckk/flare/projects/FLARE/logs/audit/"
AUDIT_LOG_FILE = os.path.join(AUDIT_LOG_DIR, "audit.log")


# Returns the current UTC timestamp in ISO format for audit records.
def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Builds and configures the FLARE audit logger once.
def _build_audit_logger() -> logging.Logger:
    logger = logging.getLogger("flare.audit")
    if logger.handlers:
        return logger

    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # TimedRotatingFileHandler rotates logs daily at midnight and keeps 30 days of history.
    handler = TimedRotatingFileHandler(
        AUDIT_LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
        utc=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


audit_logger = _build_audit_logger()


# Writes one compact JSON audit event with request metadata and optional extra fields.
def _log_event(event_type, extra=None):
    payload = {
        "timestamp": _utc_iso(),
        "event": event_type,
        "request_id": getattr(g, "request_id", None) if has_request_context() else None,
        "ip": request.remote_addr if has_request_context() else None,
        "method": request.method if has_request_context() else None,
        "path": request.path if has_request_context() else None,
        "user_agent": request.headers.get("User-Agent") if has_request_context() else None,
    }
    if extra:
        payload.update(extra)
    audit_logger.info(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))


# Wraps a Flask route to log request, response timing, and route exceptions.
def audit_request(event_type):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            g.request_id = uuid4().hex[:8]
            start = time.perf_counter()
            _log_event(f"{event_type}.request")
            try:
                response = func(*args, **kwargs)
                finalized = make_response(response)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                _log_event(
                    f"{event_type}.response",
                    {"status_code": finalized.status_code, "elapsed_ms": elapsed_ms},
                )
                return response
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                _log_event(
                    f"{event_type}.error",
                    {"error": str(exc), "elapsed_ms": elapsed_ms},
                )
                raise

        return wrapper

    return decorator


# Logs model prediction metadata without storing patient identifiers or filenames.
def log_prediction(modality, result, file_info=None):
    prediction_label = (
        result.get("prediction")
        or result.get("pred_label")
        or result.get("label")
        or result.get("result_class")
    )
    extra = {
        "modality": modality,
        "prediction": prediction_label,
        "confidence": result.get("confidence"),
    }
    if file_info:
        if "file_size_bytes" in file_info:
            extra["file_size_bytes"] = file_info.get("file_size_bytes")
        if "file_ext" in file_info:
            extra["file_ext"] = file_info.get("file_ext")
    _log_event("prediction", extra)
