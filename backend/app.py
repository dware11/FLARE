"""
FLARE MOCK BACKEND (EDITABLE)

- REST surface for the senior design demo + integration with the shared UI repo.
- Geotracker: in-memory review queue; we replace this when we need real persistence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from flask import Flask, jsonify, request, g
from flask_cors import CORS
import json
import random
import os
import uuid as _uuid
from pathlib import Path
from predict_mri import predict_mri
from predict_ct import predict_ct
# audit_log.py tracks every API call and prediction for compliance
from audit_log import audit_request, log_prediction

# ============================================================================
# App + CORS
# ============================================================================

app = Flask(__name__)

# SECURITY: Rate limiting — prevents API abuse on AI inference endpoints (HIPAA consideration)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://",
)


@app.before_request
def _preload_models_once():
    if getattr(_preload_models_once, "_done", False):
        return
    _preload_models_once._done = True

    # CT model
    try:
        from ml.brain.ct.infer import _load_model

        # NOTE: switched to multimodal_finetune_ct after fine-tune on tumor/normal dataset
        # Test AUC: 0.9975, Sensitivity: 99.4%, Specificity: 99.1%
        # Demo patient baselines differ from old checkpoint — re-verify before demo.
        # OLD default: /scratch/bckk/flare/ct_brain/outputs/rsna_windowed_exp2/best.pt
        CT_CKPT = os.environ.get(
            "FLARE_CT_CHECKPOINT",
            "/scratch/bckk/flare/ct_brain/outputs/multimodal_finetune_ct/best_ct_finetune.pt",
        )
        ckpt_path = Path(CT_CKPT)
        if ckpt_path.exists():
            ct_model = _load_model(ckpt_path)
            app.logger.info("CT model preloaded")
            try:
                import torch
                device = next(ct_model.parameters()).device
                ct_model(torch.zeros(1, 1, 64, 64, 64, device=device))
                app.logger.info("CT model warmup complete")
            except Exception as wex:
                app.logger.warning("CT warmup skipped: %s", wex)
        else:
            app.logger.warning("CT checkpoint not found at %s", CT_CKPT)
    except Exception as ex:
        app.logger.warning("CT preload failed: %s", ex)

    # MRI classification model (UPDATED: 5-class)
    try:
        from predict_mri import _load_cls_model

        cls_model = _load_cls_model()
        app.logger.info("MRI classification model preloaded (5-class: glioma, meningioma, pituitary, no_tumor, normal)")
        try:
            import torch
            device = next(cls_model.parameters()).device
            cls_model(torch.zeros(1, 3, 224, 224, device=device))
            app.logger.info("MRI classification warmup complete")
        except Exception as wex:
            app.logger.warning("MRI classification warmup skipped: %s", wex)
    except Exception as ex:
        app.logger.warning("MRI classification preload failed: %s", ex)

    # MRI segmentation model
    try:
        from predict_mri import _load_seg_model

        seg_model = _load_seg_model()
        app.logger.info("MRI segmentation model preloaded")
        try:
            import torch
            device = next(seg_model.parameters()).device
            seg_model(torch.zeros(1, 3, 256, 256, device=device))
            app.logger.info("MRI segmentation warmup complete")
        except Exception as wex:
            app.logger.warning("MRI segmentation warmup skipped: %s", wex)
    except Exception as ex:
        app.logger.warning("MRI segmentation preload failed: %s", ex)

    # BraTS 3D U-Net model
    try:
        from predict_mri import _load_brats_model

        brats_model = _load_brats_model()
        app.logger.info("BraTS model preloaded")
        try:
            import torch
            device = next(brats_model.parameters()).device
            brats_model(torch.zeros(1, 4, 128, 128, 128, device=device))
            app.logger.info("BraTS model warmup complete")
        except Exception as wex:
            app.logger.warning("BraTS warmup skipped: %s", wex)
    except Exception as ex:
        app.logger.warning("BraTS preload failed: %s", ex)


_DEV_ORIGINS = [
    "https://flare-woad.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

CORS(
    app,
    origins=_DEV_ORIGINS,
    supports_credentials=True,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "ngrok-skip-browser-warning",
    ],
)


@app.after_request
def add_security_headers(response):
    # No-store: browser must not cache scan results (HIPAA requirement)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    # Prevents MIME-type sniffing attacks
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Prevents clickjacking — page can't be loaded in an iframe
    response.headers["X-Frame-Options"] = "DENY"
    # Forces HTTPS for 1 year
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # Attach request ID to response so frontend can reference it
    response.headers["X-Request-ID"] = getattr(g, "request_id", "unknown")
    return response


UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MRI_MODALITIES = {"brain_mri", "brain_brats"}

# ============================================================================
# Static hospital registry (identity + map coordinates)
# ============================================================================

HOSPITAL_REGISTRY: dict[str, dict] = {
    "H001": {
        "id": "H001",
        "name": "Houston Methodist Hospital",
        "latitude": 29.7095,
        "longitude": -95.3986,
    },
    "H002": {
        "id": "H002",
        "name": "Memorial Hermann - Texas Medical Center",
        "latitude": 29.7080,
        "longitude": -95.4022,
    },
    "H003": {
        "id": "H003",
        "name": "Baylor St. Luke's Medical Center",
        "latitude": 29.7105,
        "longitude": -95.3965,
    },
    "H004": {
        "id": "H004",
        "name": "Ben Taub Hospital",
        "latitude": 29.7040,
        "longitude": -95.4013,
    },
    "H005": {
        "id": "H005",
        "name": "Texas Children's Hospital",
        "latitude": 29.7060,
        "longitude": -95.4013,
    },
}

# Legacy mock dashboard numbers (not geotracker approval counts)
HOSPITAL_STATIC_DEMO_METRICS: dict[str, dict] = {
    "H001": {"total_cases": 1200, "high_risk_cases": 240, "high_risk_percentage": 20.0},
    "H002": {"total_cases": 950, "high_risk_cases": 190, "high_risk_percentage": 20.0},
    "H003": {"total_cases": 780, "high_risk_cases": 156, "high_risk_percentage": 20.0},
    "H004": {"total_cases": 620, "high_risk_cases": 155, "high_risk_percentage": 25.0},
    "H005": {"total_cases": 450, "high_risk_cases": 68, "high_risk_percentage": 15.1},
}


# ============================================================================
# Review queue + geotracker (in-memory)
# ============================================================================
# TODO: replace with SQLite/Postgres or JSON file if you need persistence.

_review_cases: dict[str, dict] = {}
_ehr_records: dict[str, dict] = {}
# I back GET/POST /cases here in-memory until we persist somewhere real.
_flare_saved_cases: list[dict] = []
# Same idea for /api/cases — we have to keep this aligned with what the frontend ships.
_api_cases_store: list[dict] = []


def _seed_demo_data() -> None:
    # Tuples: patient_id, hospital, modality, confidence (0–1), AI result, cancer type (brain)
    # AI: 7 Malignant, 4 Benign, 4 Normal (labels per row). Modalities: mix brain_ct, brain_mri, brain_fusion.
    # Hospitals: 3 cases each across H001–H005.
    # Four cases are seeded pending (demoCase ids in _PENDING_DEMO_SEED_CASE_IDS) for EHR review demo.
    demo_cases = [
        ("DEMO_001", "H001", "brain_ct", 0.8900, "Malignant", "Glioma"),
        ("DEMO_002", "H002", "brain_mri", 0.9200, "Malignant", "Meningioma"),
        ("DEMO_003", "H003", "brain_fusion", 0.7800, "Malignant", "Glioma"),
        ("DEMO_004", "H004", "brain_ct", 0.9600, "Malignant", "Glioma"),
        ("DEMO_005", "H005", "brain_mri", 0.8400, "Malignant", "Meningioma"),
        ("DEMO_006", "H001", "brain_fusion", 0.9100, "Malignant", "Pituitary"),
        ("DEMO_007", "H002", "brain_ct", 0.8500, "Malignant", "Meningioma"),
        ("DEMO_008", "H003", "brain_mri", 0.7200, "Benign", "Meningioma"),
        ("DEMO_009", "H004", "brain_fusion", 0.6100, "Benign", "Pituitary"),
        ("DEMO_010", "H005", "brain_ct", 0.7400, "Benign", "Pituitary"),
        ("DEMO_011", "H001", "brain_mri", 0.6900, "Benign", "Meningioma"),
        ("DEMO_012", "H002", "brain_fusion", 0.9000, "Normal", "Normal"),
        ("DEMO_013", "H003", "brain_ct", 0.8800, "Normal", "Normal"),
        ("DEMO_014", "H004", "brain_mri", 0.9500, "Normal", "Normal"),
        ("DEMO_015", "H005", "brain_fusion", 0.9300, "Normal", "Normal"),
    ]

    _PENDING_DEMO_SEED_CASE_IDS = frozenset(
        {"demo_case_003", "demo_case_007", "demo_case_010", "demo_case_013"}
    )

    hospital_names = {
        "H001": "Houston Methodist Hospital",
        "H002": "Memorial Hermann - Texas Medical Center",
        "H003": "Baylor St. Luke's Medical Center",
        "H004": "Ben Taub Hospital",
        "H005": "Texas Children's Hospital",
    }

    base_time = datetime.now(timezone.utc) - timedelta(hours=48)

    for i, (pid, hid, modality, conf, result_class, cancer_type) in enumerate(demo_cases):
        case_id = f"demo_case_{i + 1:03d}"
        created = (base_time + timedelta(hours=i * 3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        is_abn = result_class in ("Malignant", "Benign")
        pred = "Abnormal" if is_abn else "Normal"
        is_pending = case_id in _PENDING_DEMO_SEED_CASE_IDS

        if is_pending:
            _review_cases[case_id] = {
                "caseId": case_id,
                "patient_id": pid,
                "hospitalId": hid,
                "hospitalName": hospital_names[hid],
                "modality": modality,
                "is_abnormal": is_abn,
                "prediction": pred,
                "confidence": conf,
                "review_status": "pending",
                "createdAt": created,
                "approvedAt": None,
                "rejectedAt": None,
                "reviewerId": None,
                "reject_reason": None,
                "result_class": result_class,
            }
            _ehr_records[case_id] = {
                "caseId": case_id,
                "patient_id": pid,
                "firstName": "Demo",
                "lastName": f"Patient {i + 1}",
                "dob": "1975-06-15",
                "hospitalId": hid,
                "hospitalName": hospital_names[hid],
                "modality": modality,
                "prediction": pred,
                "confidence": conf,
                "is_abnormal": is_abn,
                "result_class": result_class,
                "cancer_type": cancer_type,
                "review_status": "pending",
                "createdAt": created,
                "approvedAt": None,
                "rejectedAt": None,
                "reviewerId": None,
                "reject_reason": None,
                "signature": None,
            }
        else:
            _review_cases[case_id] = {
                "caseId": case_id,
                "patient_id": pid,
                "hospitalId": hid,
                "hospitalName": hospital_names[hid],
                "modality": modality,
                "is_abnormal": is_abn,
                "prediction": pred,
                "confidence": conf,
                "review_status": "approved",
                "createdAt": created,
                "approvedAt": created,
                "rejectedAt": None,
                "reviewerId": "demo_reviewer",
                "reject_reason": None,
                "result_class": result_class,
            }

            _ehr_records[case_id] = {
                "caseId": case_id,
                "patient_id": pid,
                "firstName": "Demo",
                "lastName": f"Patient {i + 1}",
                "dob": "1975-06-15",
                "hospitalId": hid,
                "hospitalName": hospital_names[hid],
                "modality": modality,
                "prediction": pred,
                "confidence": conf,
                "is_abnormal": is_abn,
                "result_class": result_class,
                "cancer_type": cancer_type,
                "review_status": "approved",
                "createdAt": created,
                "approvedAt": created,
                "rejectedAt": None,
                "reviewerId": "demo_reviewer",
                "reject_reason": None,
                "signature": "Demo Reviewer MD",
            }


def _reset_demo_data() -> None:
    _review_cases.clear()
    _ehr_records.clear()
    _flare_saved_cases.clear()
    _api_cases_store.clear()
    _seed_demo_data()


_seed_demo_data()


SUPPORTED_MODALITIES = ["brain_ct", "brain_mri", "brain_brats"]

def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# SECURITY: File upload validation — blocks malicious or oversized uploads (HIPAA consideration)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB (DICOM zips can be large)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".npz", ".nii", ".gz", ".dcm", ".zip"}


def _validate_upload(file) -> tuple[bool, str]:
    if file is None:
        return False, "No file provided"
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_UPLOAD_BYTES:
        return False, f"File too large (max 200MB, got {size // 1024 // 1024}MB)"
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if filename.endswith(".nii.gz"):
        ext = ".gz"
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type not allowed: {ext}"
    return True, "ok" 


def hospital_exists(hospital_id: str | None) -> bool:
    if hospital_id is None:
        return False
    return hospital_id in HOSPITAL_REGISTRY


def is_abnormal_pred_label(pred_label: str) -> bool:
    """Mock model uses abnormal/normal; extend for real tumor class names later."""
    label = (pred_label or "").strip().lower()
    return label in ("abnormal", "tumor", "malignant")


def mock_severity(confidence: float) -> str:
    if confidence >= 0.92:
        return "high"
    if confidence >= 0.88:
        return "medium"
    return "low"


def run_mock_model(patient_id: str, modality: str) -> dict:
    """Odd last digit => abnormal (same idea as original /api/run-model)."""
    last_char = patient_id[-1] if patient_id else ""
    if last_char.isdigit() and int(last_char) % 2 == 1:
        pred_label = "abnormal"
    else:
        pred_label = "normal"

    confidence = round(random.uniform(0.85, 0.95), 4)

    if pred_label == "abnormal":
        probabilities = {"normal": round(1.0 - confidence, 4), "abnormal": confidence}
    else:
        probabilities = {"normal": confidence, "abnormal": round(1.0 - confidence, 4)}

    return {
        "patient_id": patient_id,
        "modality": modality,
        "pred_label": pred_label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def _absolute_url_for_path(path: str | None) -> str | None:
    """Full URL for static assets — we need this when Vite and Flask run on different ports."""
    if not path:
        return None
    p = path.strip()
    if p.startswith("http://") or p.startswith("https://"):
        return p
    base = request.host_url.rstrip("/")
    if not p.startswith("/"):
        p = "/" + p
    return base + p


def _run_mri_full_pipeline(
    *,
    file_storage,
    patient_id: str,
    modality: str,
    hospital_id: str,
    first_name: str | None,
    last_name: str | None,
    dob: str | None,
) -> tuple[dict, int]:
    """
    Shared brain MRI/BraTS path for /api/mri/predict and POST /predict (one pipeline, two entrypoints).
    Returns (response_dict, http_status).
    UPDATED: Handles 5-class model (glioma, meningioma, pituitary, no_tumor, normal)
    """
    ext = Path(file_storage.filename or "").suffix.lower()
    fname_lower = (file_storage.filename or "").lower()
    # Path.suffix is ".gz" for "x.nii.gz", not ".nii.gz" — use endswith for NIfTI
    is_nifti = fname_lower.endswith(".nii.gz") or fname_lower.endswith(".nii")

    if is_nifti and modality == "brain_brats":
        return {"error": "BraTS modality requires NPZ, not NIfTI"}, 400

    if not is_nifti and ext not in {".jpg", ".jpeg", ".png", ".npz"}:
        return {"error": "Invalid file type — use JPG, PNG, NPZ, or NIfTI (.nii / .nii.gz)"}, 400

    # Keep .nii vs .nii.gz on disk (do not rename uncompressed volumes to .gz)
    if is_nifti:
        nifti_ext = ".nii.gz" if fname_lower.endswith(".nii.gz") else ".nii"
        filename = f"{patient_id}_{_uuid.uuid4().hex[:8]}{nifti_ext}"
    else:
        filename = f"{patient_id}_{_uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(file_path)

    result = predict_mri(file_path, modality=modality, patient_id=patient_id)
    if "error" in result:
        return result, 500

    abnormal = result.get("result_class") in ("Malignant", "Benign")
    case_id = None

    if abnormal:
        case_id = str(uuid4())
        site = HOSPITAL_REGISTRY[hospital_id]
        conf = float(result["confidence"])
        _review_cases[case_id] = {
            "caseId": case_id,
            "hospitalId": hospital_id,
            "hospitalName": site["name"],
            "patient_id": patient_id,
            "modality": modality,
            "pred_label": result["pred_label"],
            "result_class": result.get("result_class"),
            "confidence": conf,
            "probabilities": result.get("probabilities"),
            "is_abnormal": True,
            "severity": mock_severity(conf),
            "review_status": "pending",
            "createdAt": _utc_iso(),
            "approvedAt": None,
            "rejectedAt": None,
            "reviewerId": None,
            "reject_reason": None,
            "signature": None,
            "segmentation": result.get("segmentation"),
            "input_image_url": result.get("input_image_url"),
            "gradcam_url": result.get("gradcam_url"),
            "gradcam_ready": result.get("gradcam_ready", False),
        }
        _ehr_records[case_id] = {
            **_review_cases[case_id],
            "firstName": first_name,
            "lastName": last_name,
            "dob": dob,
            "signature": None,
        }

    out = {
        **result,
        "hospitalId": hospital_id,
        "caseId": case_id,
        "review_required": abnormal,
    }
    return out, 200


def _mri_to_flare_predict_response(cancer_type: str, body: dict) -> dict:
    """Map MRI JSON into the predict response shape we already committed to on the client — I'm adapting here so we don't break the demo."""
    rc = body.get("result_class") or "Normal"
    if rc not in ("Normal", "Benign", "Malignant"):
        rc = (
            "Malignant"
            if body.get("pred_label", "").lower() in ("abnormal", "tumor", "malignant")
            else "Normal"
        )
    seg = body.get("segmentation") or {}
    loc = seg.get("overlay_url") or seg.get("mask_url") or body.get("gradcam_url")
    out = {
        "cancer_type": cancer_type,
        "prediction": rc,
        "confidence": float(body.get("confidence", 0)),
        "localization_url": _absolute_url_for_path(loc),
        "gradcam_ready": bool(body.get("gradcam_ready", False)),
    }
    if body.get("probabilities") is not None:
        out["probabilities"] = body.get("probabilities")
    return out


# ============================================================================
# Routes: health + legacy run-model + hospitals
# ============================================================================


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "version": "1.0", "mode": "MOCK"}), 200


@app.route("/api/run-model", methods=["POST"])
def run_model():
    """Mock inference only — no hospitalId, no review queue (legacy clients)."""
    data = request.get_json(silent=True) or {}
    patient_id = data.get("patient_id")
    modality = data.get("modality")

    if not patient_id:
        return (
            jsonify({"error": "Missing required field: patient_id", "code": "MISSING_PATIENT_ID"}),
            400,
        )
    if modality not in SUPPORTED_MODALITIES:
        return (
            jsonify(
                {
                    "error": "Invalid modality",
                    "code": "INVALID_MODALITY",
                    "supported_modalities": SUPPORTED_MODALITIES,
                }
            ),
            400,
        )

    payload = run_mock_model(patient_id, modality)
    payload["_mode"] = "MOCK"
    return jsonify(payload), 200


@app.route("/api/hospitals", methods=["GET"])
def get_hospitals():
    """Same JSON array shape as before: coords + static demo caseload fields."""
    rows: list[dict] = []
    for hid, site in HOSPITAL_REGISTRY.items():
        rows.append({**site, **HOSPITAL_STATIC_DEMO_METRICS[hid]})
    return jsonify(rows), 200


# ============================================================================
# Predict with geotracker + reviews + summary
# ============================================================================


@app.route("/api/predict", methods=["POST"])
@limiter.limit("20 per minute")
def api_predict():
    """
    Body: patient_id, modality, hospitalId (camelCase).
    Abnormal predictions create a pending review case.
    """
    data = request.get_json(silent=True) or {}
    patient_id = data.get("patient_id")
    modality = data.get("modality")
    hospital_id = data.get("hospitalId")
    first_name = data.get("first_name") 
    last_name = data.get("last_name") 
    dob = data.get("dob") 

    if not patient_id:
        return jsonify({"error": "Missing patient_id", "code": "MISSING_PATIENT_ID"}), 400
    if modality not in SUPPORTED_MODALITIES:
        return (
            jsonify(
                {
                    "error": "Invalid modality",
                    "code": "INVALID_MODALITY",
                    "supported_modalities": SUPPORTED_MODALITIES,
                }
            ),
            400,
        )
    if not hospital_id:
        return jsonify({"error": "Missing hospitalId", "code": "MISSING_HOSPITAL_ID"}), 400
    if not hospital_exists(hospital_id):
        return jsonify({"error": "Unknown hospitalId", "code": "INVALID_HOSPITAL_ID"}), 400

    result = run_mock_model(patient_id, modality)
    abnormal = is_abnormal_pred_label(result["pred_label"])

    case_id = None
    review_required = False

    if abnormal:
        case_id = str(uuid4())
        review_required = True
        site = HOSPITAL_REGISTRY[hospital_id]
        conf = float(result["confidence"])
        _review_cases[case_id] = {
            "caseId": case_id,
            "hospitalId": hospital_id,
            "hospitalName": site["name"],
            "patient_id": patient_id,
            "modality": modality,
            "pred_label": result["pred_label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "is_abnormal": True,
            "severity": mock_severity(conf),
            "review_status": "pending",
            "createdAt": _utc_iso(),
            "approvedAt": None,
            "rejectedAt": None,
            "reviewerId": None,
            "reject_reason": None,
            "signature": None,
        }
        _ehr_records[case_id] = {
            "caseId": case_id, 
            "patient_id": patient_id, 
            "firstName": first_name, 
            "lastName": last_name, 
            "dob": dob, 
            "hospitalId": hospital_id, 
            "hospitalName": site["name"], 
            "modality": modality, 
            "prediction": result["pred_label"], 
            "confidence": result["confidence"], 
            "review_status": "pending", 
            "createdAt": _review_cases[case_id]["createdAt"], 
            "approvedAt": None, 
            "rejectedAt": None, 
            "reviewerId": None, 
            "reject_reason": None, 
            "signature": None, 
        }

    out = {
        **result,
        "hospitalId": hospital_id,
        "caseId": case_id,
        "review_required": review_required,
        "_mode": "MOCK",
    }
    return jsonify(out), 200

@app.route("/api/mri/predict", methods=["POST"])
@audit_request("mri_predict")
@limiter.limit("20 per minute")
def api_mri_predict():
    """
    Brain MRI / BraTS prediction (real pipeline).
    UPDATED: 5-class model support
    Multipart/form-data:
        file        → JPG/PNG for brain_mri | NPZ for brain_brats
        patient_id  → string
        modality    → "brain_mri" or "brain_brats"
        hospitalId  → string
        first_name, last_name, dob → optional
    """
    patient_id  = request.form.get("patient_id")
    modality    = request.form.get("modality", "brain_mri")
    hospital_id = request.form.get("hospitalId")
    first_name  = request.form.get("first_name")
    last_name   = request.form.get("last_name")
    dob         = request.form.get("dob")

    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400
    if modality not in MRI_MODALITIES:
        return (
            jsonify(
                {
                    "error": f"Invalid modality for MRI endpoint: {modality}",
                    "valid": list(MRI_MODALITIES),
                }
            ),
            400,
        )

    if not hospital_id or not hospital_exists(hospital_id):
        return jsonify({"error": "Missing or invalid hospitalId"}), 400

    # BraTS patient folder: multipart t1n, t1c, t2w, t2f (one NIfTI each). Fusion → NPZ / 3D not implemented yet.
    t1n_f = request.files.get("t1n")
    t1c_f = request.files.get("t1c")
    t2w_f = request.files.get("t2w")
    t2f_f = request.files.get("t2f")
    quartet = [t1n_f, t1c_f, t2w_f, t2f_f]
    if any(f is not None and getattr(f, "filename", None) for f in quartet):
        if modality != "brain_brats":
            return (
                jsonify({"error": "t1n/t1c/t2w/t2f multipart requires modality=brain_brats"}),
                400,
            )
        if not all(f is not None and getattr(f, "filename", None) for f in quartet):
            missing = [
                n
                for n, f in zip(("t1n", "t1c", "t2w", "t2f"), quartet)
                if f is None or not getattr(f, "filename", None)
            ]
            return (
                jsonify(
                    {
                        "error": "BraTS folder: all four fields required: t1n, t1c, t2w, t2f.",
                        "missing": missing,
                    }
                ),
                400,
            )
        return (
            jsonify(
                {
                    "error": (
                        "BraTS multi-sequence NIfTI folder ingest is not implemented yet. "
                        "Use a single .npz (4×128³) with modality brain_brats, or single-file BRISC (brain_mri)."
                    ),
                    "hint": "Planned: stack t1n/t1c/t2w/t2f on server into NPZ or 3D pipeline input.",
                }
            ),
            501,
        )

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    upload = request.files["file"]
    ok, msg = _validate_upload(upload)
    if not ok:
        return jsonify({"error": msg, "code": "INVALID_FILE"}), 400

    mri_file_ext = Path(upload.filename or "").suffix.lower() or None
    upload.seek(0, os.SEEK_END)
    mri_file_size = upload.tell()
    upload.seek(0)

    out, status = _run_mri_full_pipeline(
        file_storage=upload,
        patient_id=patient_id,
        modality=modality,
        hospital_id=hospital_id,
        first_name=first_name,
        last_name=last_name,
        dob=dob,
    )
    if status < 400:
        log_prediction(
            "mri",
            out,
            file_info={"file_size_bytes": mri_file_size, "file_ext": mri_file_ext},
        )
    return jsonify(out), status


def _ct_cam_static_dir() -> str:
    """Grad-CAM PNGs under Flask static for /static/cam/*.png."""
    d = os.path.join(os.path.dirname(__file__), "static", "cam")
    os.makedirs(d, exist_ok=True)
    return d


@app.route("/api/ct/predict", methods=["POST"])
@audit_request("ct_predict")
@limiter.limit("20 per minute")
def api_ct_predict():
    """CT prediction from uploaded NPZ, JPEG/PNG, or zipped DICOM study."""
    patient_id = request.form.get("patient_id")
    hospital_id = request.form.get("hospitalId") or "H001"
    first_name = request.form.get("first_name")
    last_name = request.form.get("last_name")
    dob = request.form.get("dob")

    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400
    if not hospital_exists(hospital_id):
        return jsonify({"error": "Invalid hospitalId"}), 400

    ct_file = request.files.get("file")
    if not ct_file:
        return jsonify({"error": "No file uploaded"}), 400

    ok, msg = _validate_upload(ct_file)
    if not ok:
        return jsonify({"error": msg, "code": "INVALID_FILE"}), 400

    try:
        saved_path = _save_upload(ct_file, patient_id, prefix="ct")
        ct_file_ext = Path(ct_file.filename or "").suffix.lower() or None
        ct_file_size = os.path.getsize(saved_path)
        cam_dir = _ct_cam_static_dir()
        try:
            ct_result = predict_ct(
                saved_path,
                patient_id=patient_id,
                allow_image=True,
                checkpoint=None,
                threshold=0.488,
                cam_dir=cam_dir,
            )
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        if not ct_result:
            return jsonify({"error": "CT inference failed — check uploaded file"}), 500
        log_prediction(
            "ct",
            ct_result,
            file_info={"file_size_bytes": ct_file_size, "file_ext": ct_file_ext},
        )

        is_abnormal = ct_result["label"] == "abnormal"
        site = HOSPITAL_REGISTRY[hospital_id]
        case_id = None
        review_required = False

        if is_abnormal:
            case_id = str(uuid4())
            review_required = True
            conf = float(ct_result["confidence"])
            created_at = _utc_iso()
            _review_cases[case_id] = {
                "caseId": case_id,
                "hospitalId": hospital_id,
                "hospitalName": site["name"],
                "patient_id": patient_id,
                "modality": "brain_ct",
                "pred_label": "Abnormal",
                "result_class": "Abnormal",
                "confidence": conf,
                "probabilities": {
                    "normal": float(ct_result["p_normal"]),
                    "abnormal": float(ct_result["p_abnormal"]),
                },
                "is_abnormal": True,
                "severity": mock_severity(conf),
                "review_status": "pending",
                "createdAt": created_at,
                "approvedAt": None,
                "rejectedAt": None,
                "reviewerId": None,
                "reject_reason": None,
                "signature": None,
                "gradcam_url": None,
                "input_image_url": None,
                "segmentation": None,
            }
            _ehr_records[case_id] = {
                **_review_cases[case_id],
                "firstName": first_name,
                "lastName": last_name,
                "dob": dob,
            }

        ct_path = ct_result.get("inference_path") or saved_path
        stem = Path(ct_path).stem
        input_format = ct_result.get("input_format")
        if input_format not in {"dicom_zip", "npz", "image"}:
            input_format = "npz"
        cam_disk = os.path.join(cam_dir, f"{stem}.png")
        cam_url = (
            _absolute_url_for_path(f"/static/cam/{stem}.png")
            if ct_result.get("cam_path") and os.path.isfile(cam_disk)
            else None
        )

        return (
            jsonify(
                {
                    "patient_id": patient_id,
                    "modality": "brain_ct",
                    "pred_label": ct_result["label"],
                    "result_class": "Abnormal" if is_abnormal else "Normal",
                    "confidence": ct_result["confidence"],
                    "p_normal": ct_result["p_normal"],
                    "p_abnormal": ct_result["p_abnormal"],
                    "cam_url": cam_url,
                    "hospitalId": hospital_id,
                    "hospitalName": site["name"],
                    "caseId": case_id,
                    "review_required": review_required,
                    "input_format": input_format,
                }
            ),
            200,
        )
    except Exception as ex:
        return jsonify({"error": f"CT inference failed: {ex}"}), 500


@app.route("/predict", methods=["POST"])
def legacy_flare_predict():
    """
    Integration: multipart file + ?cancer_type=brain|breast to match the cancer demo's existing fetch.
    Brain uses our real MRI stack; I default hospitalId to H001 when the request omits it.
    Breast stays mock until we actually ship a breast model here.
    """
    cancer_type = (request.args.get("cancer_type") or "brain").lower()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    ok, msg = _validate_upload(file)
    if not ok:
        return jsonify({"error": msg, "code": "INVALID_FILE"}), 400

    patient_id = request.form.get("patient_id") or f"flare_{uuid4().hex[:10]}"
    hospital_id = request.form.get("hospitalId") or "H001"
    fn = request.form.get("first_name")
    ln = request.form.get("last_name")
    dob = request.form.get("dob")

    if cancer_type == "brain":
        modality = request.form.get("modality") or "brain_mri"
        if modality not in MRI_MODALITIES:
            modality = "brain_mri"
        body, status = _run_mri_full_pipeline(
            file_storage=file,
            patient_id=patient_id,
            modality=modality,
            hospital_id=hospital_id,
            first_name=fn,
            last_name=ln,
            dob=dob,
        )
        if status >= 400:
            return jsonify(body), status
        return jsonify(_mri_to_flare_predict_response("brain", body)), 200

    if cancer_type == "breast":
        m = run_mock_model(patient_id, "brain_ct")
        pred = "Malignant" if m["pred_label"] == "abnormal" else "Normal"
        return (
            jsonify(
                {
                    "cancer_type": "breast",
                    "prediction": pred,
                    "confidence": m["confidence"],
                    "localization_url": None,
                }
            ),
            200,
        )

    return jsonify({"error": "Unsupported cancer_type", "supported": ["brain", "breast"]}), 400


@app.route("/cases", methods=["GET", "POST"])
def legacy_flare_cases():
    """Backs save/list case calls from the demo — in-memory until we decide on storage."""
    if request.method == "GET":
        return jsonify(_flare_saved_cases), 200
    data = request.get_json(silent=True) or {}
    _flare_saved_cases.append(data)
    return jsonify({"ok": True}), 200


@app.route("/api/cases", methods=["GET", "POST"])
def api_cases():
    """
    POST: multipart (payload + files) for the create-case flow we're wiring to this backend.
    GET: JSON array for the EHR-style case list. POST is still mock inference — I still have to upgrade that path when we're ready.
    """
    if request.method == "GET":
        return jsonify(_api_cases_store), 200

    if request.content_type and "multipart/form-data" in request.content_type:
        raw = request.form.get("payload")
        try:
            p = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid payload JSON"}), 400
        patient = p.get("patient") or {}
        patient_id = patient.get("medicalId") or patient.get("medical_id")
        if not patient_id:
            return jsonify({"error": "Missing patient.medicalId"}), 400
        ct = (p.get("cancerType") or p.get("cancer_type") or "brain").lower()
        modality = "brain_mri" if ct == "brain" else "brain_ct"
        r = run_mock_model(patient_id, modality)
        classification = "Malignant" if r["pred_label"] == "abnormal" else "Normal"
        case = {
            "caseId": str(uuid4()),
            "classification": classification,
            "confidence": r["confidence"],
            "localizationImageUrl": None,
            "createdAt": _utc_iso(),
        }
        _api_cases_store.append(case)
        return jsonify(case), 200

    data = request.get_json(silent=True) or {}
    patient_id = data.get("patient_id") or (data.get("patient") or {}).get("medicalId")
    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400
    modality = data.get("modality") or "brain_mri"
    if modality not in SUPPORTED_MODALITIES:
        modality = "brain_mri"
    r = run_mock_model(patient_id, modality)
    classification = "Malignant" if r["pred_label"] == "abnormal" else "Normal"
    case = {
        "caseId": str(uuid4()),
        "classification": classification,
        "confidence": r["confidence"],
        "localizationImageUrl": None,
        "createdAt": _utc_iso(),
    }
    _api_cases_store.append(case)
    return jsonify(case), 200


@app.route("/api/patients", methods=["GET"])
def api_patients_stub():
    """Stub so the CT patient list call stops dying on a missing route; I'll return real IDs once CT is integrated."""
    return jsonify({"patients": []}), 200


@app.route("/api/outbreaks", methods=["GET"])
def api_outbreaks_stub():
    """Stub for now — we should either map geotracker summary into this shape or define a real outbreaks source."""
    return jsonify([]), 200


@app.route("/api/reviews/pending", methods=["GET"])
def reviews_pending():
    pending = [
        c
        for c in _review_cases.values()
        if c["review_status"] == "pending" and c.get("is_abnormal")
    ]
    pending.sort(key=lambda c: c["createdAt"], reverse=True)
    return jsonify({"cases": pending}), 200


@app.route("/api/reviews/<case_id>/approve", methods=["POST"])
@audit_request("review_approve")
def reviews_approve(case_id: str):
    """Idempotent: approving again does not double-count."""
    case = _review_cases.get(case_id)
    if not case:
        return jsonify({"error": "Unknown caseId", "code": "NOT_FOUND"}), 404

    body = request.get_json(silent=True) or {}

    if case["review_status"] == "approved":
        return jsonify({"ok": True, "caseId": case_id, "status": "approved", "note": "already approved"}), 200
    if case["review_status"] == "rejected":
        return jsonify({"error": "Case already rejected", "code": "INVALID_STATE"}), 409
    if case["review_status"] != "pending":
        return jsonify({"error": "Invalid review state", "code": "INVALID_STATE"}), 400

    reviewer_name = (body.get("reviewerName") or body.get("reviewer_name") or "").strip()
    signature_text = (body.get("signature") or body.get("digitalSignature") or "").strip()
    if not reviewer_name or not signature_text:
        return (
            jsonify(
                {
                    "error": "Reviewer name and digital signature are required",
                    "code": "MISSING_SIGNATURE",
                }
            ),
            400,
        )

    reviewer_id = reviewer_name
    case["review_status"] = "approved"
    case["approvedAt"] = _utc_iso()
    case["reviewerId"] = reviewer_id
    case["reviewerName"] = reviewer_name
    case["signature"] = signature_text

    ehr = _ehr_records.get(case_id) 
    if ehr: 
        ehr["review_status"] = "approved"
        ehr["approvedAt"] = case["approvedAt"]
        ehr["rejectedAt"] = None 
        ehr["reviewerId"] = reviewer_id
        ehr["reject_reason"] = None 
        ehr["signature"] = signature_text
    return jsonify({"ok": True, "caseId": case_id, "status": "approved"}), 200


@app.route("/api/reviews/<case_id>/reject", methods=["POST"])
@audit_request("review_reject")
def reviews_reject(case_id: str):
    case = _review_cases.get(case_id)
    if not case:
        return jsonify({"error": "Unknown caseId", "code": "NOT_FOUND"}), 404

    body = request.get_json(silent=True) or {}
    reviewer_id = body.get("reviewerId", "anonymous-reviewer")
    reason = body.get("reason")
    signature = body.get("signature")

    if case["review_status"] == "rejected":
        return jsonify({"ok": True, "caseId": case_id, "status": "rejected", "note": "already rejected"}), 200
    if case["review_status"] == "approved":
        return jsonify({"error": "Case already approved", "code": "INVALID_STATE"}), 409
    if case["review_status"] != "pending":
        return jsonify({"error": "Invalid review state", "code": "INVALID_STATE"}), 400

    case["review_status"] = "rejected"
    case["rejectedAt"] = _utc_iso()
    case["reviewerId"] = reviewer_id
    case["reject_reason"] = reason
    if signature is not None:
        case["signature"] = signature

    ehr = _ehr_records.get(case_id) 
    if ehr: 
        ehr["review_status"] = "rejected"
        ehr["rejectedAt"] = case["rejectedAt"]
        ehr["approvedAt"] = None 
        ehr["reviewerId"] = reviewer_id
        ehr["reject_reason"] = reason 
        if signature is not None:
            ehr["signature"] = signature
    return jsonify({"ok": True, "caseId": case_id, "status": "rejected"}), 200


@app.route("/api/admin/reset", methods=["POST"])
def admin_reset_demo_data():
    # DEMO ONLY — remove before production deployment
    _reset_demo_data()
    return (
        jsonify(
            {
                "status": "reset",
                "message": "Demo data reset to 15 seeded cases",
                "total_cases": 15,
            }
        ),
        200,
    )


@app.route("/api/ehr", methods=["GET"]) 
def ehr_list(): 
    rows = list(_ehr_records.values()) 
    rows.sort(key=lambda r: r.get("createdAt") or "", reverse=True) 
    return jsonify({"records": rows}), 200

@app.route("/api/ehr/<case_id>", methods=["GET"]) 
def ehr_one(case_id: str): 
    rec = _ehr_records.get(case_id) 
    if not rec: 
        return jsonify({"error": "Unknown caseId", "code": "NOT_FOUND"}), 404
    return jsonify(rec), 200


def _count_hospital(hospital_id: str) -> tuple[int, int]:
    """(approved_abnormal, pending_abnormal) for one hospital."""
    approved_n = 0
    pending_n = 0
    for c in _review_cases.values():
        if c["hospitalId"] != hospital_id or not c.get("is_abnormal"):
            continue
        if c["review_status"] == "approved":
            approved_n += 1
        elif c["review_status"] == "pending":
            pending_n += 1
    return approved_n, pending_n


@app.route("/api/geotracker/hospital/<hospital_id>/cases", methods=["GET"])
def geotracker_hospital_cases(hospital_id: str):
    if not hospital_exists(hospital_id):
        return jsonify({"error": "Unknown hospitalId", "code": "INVALID_HOSPITAL_ID"}), 404

    cases = [
        r
        for r in _ehr_records.values()
        if r.get("hospitalId") == hospital_id
        and r.get("review_status") == "approved"
        and (
            r.get("is_abnormal") is True
            or str(r.get("prediction", "")).lower() in ("abnormal", "tumor", "malignant", "benign")
            or str(r.get("result_class", "")).lower() in ("malignant", "benign")
        )
    ]

    cases.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

    return jsonify({"cases": cases[:5]}), 200


@app.route("/api/outbreak/status", methods=["GET"])
def outbreak_status():
    total_approved = sum(
        1
        for c in _ehr_records.values()
        if c.get("is_abnormal") and c.get("review_status") == "approved"
    )

    hospital_counts: dict[str, int] = {}
    for hid in HOSPITAL_REGISTRY:
        count = sum(
            1
            for c in _ehr_records.values()
            if c.get("hospitalId") == hid
            and c.get("is_abnormal")
            and c.get("review_status") == "approved"
        )
        hospital_counts[hid] = count

    population = 2_300_000
    expected_rate = 0.005
    expected_cases = population * expected_rate

    outbreak_level = "Normal"
    outbreak_color = "#94a3b8"
    triggered = False

    if total_approved >= 10:
        outbreak_level = "Critical"
        outbreak_color = "#e11d48"
        triggered = True
    elif total_approved >= 5:
        outbreak_level = "Elevated"
        outbreak_color = "#f97316"
        triggered = True
    elif total_approved >= 3:
        outbreak_level = "Moderate"
        outbreak_color = "#eab308"
        triggered = False

    hotspot_hospitals = [hid for hid, count in hospital_counts.items() if count >= 3]

    return (
        jsonify(
            {
                "outbreak_level": outbreak_level,
                "outbreak_color": outbreak_color,
                "triggered": triggered,
                "total_approved_abnormal": total_approved,
                "expected_baseline": expected_cases,
                "percent_of_baseline": round((total_approved / expected_cases) * 100, 4)
                if expected_cases
                else 0.0,
                "hospital_counts": hospital_counts,
                "hotspot_hospitals": hotspot_hospitals,
                "population": population,
                "threshold_elevated": 5,
                "threshold_critical": 10,
            }
        ),
        200,
    )


@app.route("/api/geotracker/summary", methods=["GET"])
def geotracker_summary():
    hospitals_out: list[dict] = []
    total_pending = 0
    total_approved = 0

    for hid, site in HOSPITAL_REGISTRY.items():
        approved_n, pending_n = _count_hospital(hid)
        total_pending += pending_n
        total_approved += approved_n

        severity_color = "#94a3b8"
        if approved_n >= 5:
            severity_color = "#e11d48"
        elif approved_n >= 2:
            severity_color = "#f97316"
        elif pending_n >= 1:
            severity_color = "#eab308"

        trend = "flat"
        if approved_n > pending_n:
            trend = "up"
        elif pending_n > approved_n and pending_n > 0:
            trend = "attention"

        hospitals_out.append(
            {
                "id": hid,
                "hospitalId": hid,
                "name": site["name"],
                "latitude": site["latitude"],
                "longitude": site["longitude"],
                "total_cases": approved_n + pending_n,
                "pending": pending_n,
                "approved": approved_n,
                "approved_abnormal": approved_n,
                "approvedAbnormalCount": approved_n,
                "pendingCount": pending_n,
                "severityColor": severity_color,
                "trend": trend,
            }
        )

    return (
        jsonify(
            {
                "hospitals": hospitals_out,
                "totals": {
                    "pending": total_pending,
                    "approved": total_approved,
                    "approved_abnormal": total_approved,
                    "approvedAbnormal": total_approved,
                },
            }
        ),
        200,
    )


def _save_upload(file_storage, patient_id: str, prefix: str = "file") -> str:
    import os
    from werkzeug.utils import secure_filename
    upload_dir = os.path.join(os.path.dirname(__file__), "static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file_storage.filename or "")[-1] or ".bin"
    fname = f"{patient_id}_{prefix}{ext}"
    fpath = os.path.join(upload_dir, secure_filename(fname))
    file_storage.seek(0)
    file_storage.save(fpath)
    return fpath


@app.route("/api/fusion/predict", methods=["POST"])
@audit_request("fusion_predict")
@limiter.limit("20 per minute")
def api_fusion_predict():
    """Late fusion: CT + MRI weighted average (CT=0.4, MRI=0.6)."""
    patient_id  = request.form.get("patient_id")
    hospital_id = request.form.get("hospitalId") or "H001"
    first_name  = request.form.get("first_name")
    last_name   = request.form.get("last_name")
    dob         = request.form.get("dob")

    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400
    if not hospital_exists(hospital_id):
        return jsonify({"error": "Invalid hospitalId"}), 400

    ct_file  = request.files.get("ct_file")
    mri_file = request.files.get("mri_file")

    if not ct_file and not mri_file:
        return jsonify({"error": "At least one of ct_file or mri_file required"}), 400

    for f in [ct_file, mri_file]:
        if f is not None:
            ok, msg = _validate_upload(f)
            if not ok:
                return jsonify({"error": msg, "code": "INVALID_FILE"}), 400

    ct_prob    = None
    ct_result  = None
    mri_prob   = None
    mri_result = None

    if ct_file is not None:
        try:
            ct_saved = _save_upload(ct_file, patient_id, prefix="ct")
            cam_dir_fusion = _ct_cam_static_dir()
            ct_result = predict_ct(
                ct_saved,
                patient_id=patient_id,
                allow_image=True,
                checkpoint=None,
                threshold=0.488,
                cam_dir=cam_dir_fusion,
            )
            if ct_result:
                ct_prob = float(ct_result["p_abnormal"])
            else:
                app.logger.warning("Fusion: CT inference returned empty result")
        except ValueError as ve:
            msg = str(ve)
            if "CT inference requires" in msg:
                msg = "Fusion CT file must be .npz volume, .zip DICOM study, or .jpg/.jpeg/.png image"
            return jsonify({"error": msg}), 400
        except Exception as ex:
            return jsonify({"error": f"CT inference failed: {ex}"}), 500

    if mri_file is not None:
        try:
            mri_path = _save_upload(mri_file, patient_id, prefix="mri")
            from predict_mri import predict_mri
            mri_result = predict_mri(mri_path, "brain_mri", patient_id)
            if mri_result and "error" not in mri_result:
                raw_conf = mri_result.get("confidence")
                if raw_conf is not None and isinstance(raw_conf, (int, float)):
                    mri_conf = float(raw_conf)
                    mri_is_abnormal = mri_result.get("result_class", "").lower() in (
                        "malignant", "abnormal", "tumor"
                    )
                    mri_prob = mri_conf if mri_is_abnormal else (1.0 - mri_conf)
                else:
                    app.logger.warning(
                        "Fusion: MRI returned no valid confidence — falling back to CT-only. "
                        "mri_result=%s", mri_result
                    )
            else:
                app.logger.warning(
                    "Fusion: MRI inference returned error or empty — falling back to CT-only. "
                    "mri_result=%s", mri_result
                )
        except Exception as ex:
            return jsonify({"error": f"MRI inference failed: {ex}"}), 500

    CT_WEIGHT  = 0.4
    MRI_WEIGHT = 0.6

    if ct_prob is not None and mri_prob is not None:
        fusion_score = (ct_prob * CT_WEIGHT) + (mri_prob * MRI_WEIGHT)
        fusion_mode  = "ct_mri"
    elif ct_prob is not None:
        fusion_score = ct_prob
        fusion_mode  = "ct_only"
    elif mri_prob is not None:
        fusion_score = mri_prob
        fusion_mode  = "mri_only"
    else:
        return jsonify({"error": "Both models returned no result"}), 500

    FUSION_THRESHOLD = 0.5
    is_abnormal  = fusion_score >= FUSION_THRESHOLD
    pred_label   = "Abnormal" if is_abnormal else "Normal"
    result_class = "Malignant" if is_abnormal else "Benign"
    log_prediction(
        "fusion",
        {"prediction": pred_label, "confidence": float(fusion_score)},
    )
    case_id = None
    review_required = False

    ct_cam_url = None
    mri_input_url = None
    mri_overlay_url = None
    if ct_result:
        stem = ct_result.get("cam_path") or ""
        if stem:
            cam_disk = os.path.join(_ct_cam_static_dir(), os.path.basename(stem))
            if os.path.isfile(cam_disk):
                ct_cam_url = _absolute_url_for_path(f"/static/cam/{os.path.basename(stem)}")
    if mri_result and isinstance(mri_result, dict):
        mri_input_url = mri_result.get("input_image_url")
        seg = mri_result.get("segmentation")
        if isinstance(seg, dict):
            mri_overlay_url = seg.get("overlay_url")

    if is_abnormal:
        case_id = str(uuid4())
        review_required = True
        site = HOSPITAL_REGISTRY[hospital_id]
        created_at = _utc_iso()
        _review_cases[case_id] = {
            "caseId": case_id,
            "hospitalId": hospital_id,
            "hospitalName": site["name"],
            "patient_id": patient_id,
            "modality": "brain_fusion",
            "pred_label": pred_label,
            "result_class": "Abnormal",
            "confidence": float(fusion_score),
            "probabilities": {
                "normal": round(1.0 - float(fusion_score), 4),
                "abnormal": round(float(fusion_score), 4),
            },
            "is_abnormal": True,
            "severity": mock_severity(float(fusion_score)),
            "review_status": "pending",
            "createdAt": created_at,
            "approvedAt": None,
            "rejectedAt": None,
            "reviewerId": None,
            "reject_reason": None,
            "signature": None,
            "gradcam_url": ct_cam_url,
            "input_image_url": mri_input_url,
            "segmentation": {"overlay_url": mri_overlay_url} if mri_overlay_url else None,
        }
        _ehr_records[case_id] = {
            **_review_cases[case_id],
            "firstName": first_name,
            "lastName": last_name,
            "dob": dob,
        }

    return jsonify({
        "patient_id":       patient_id,
        "fusion_mode":      fusion_mode,
        "fusion_score":     round(fusion_score, 4),
        "pred_label":       pred_label,
        "result_class":     result_class,
        "confidence":       round(fusion_score, 4),
        "is_abnormal":      is_abnormal,
        "ct_prob":          round(ct_prob, 4) if ct_prob is not None else None,
        "mri_prob":         round(mri_prob, 4) if mri_prob is not None else None,
        "ct_weight":        CT_WEIGHT,
        "mri_weight":       MRI_WEIGHT,
        "threshold":        FUSION_THRESHOLD,
        "ct_cam_url":       ct_cam_url,
        "mri_input_url":    mri_input_url,
        "mri_overlay_url":  mri_overlay_url,
        "caseId":           case_id,
        "review_required":  review_required,
        "ct_details":       ct_result,
        "mri_details":      mri_result,
    }), 200


# SECURITY: HTTP security headers — prevents caching of PHI on client devices (HIPAA consideration)
@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
    return response


def _save_upload(file_storage, patient_id: str, prefix: str = "file") -> str:
    import os
    from werkzeug.utils import secure_filename
    upload_dir = os.path.join(os.path.dirname(__file__), "static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file_storage.filename or "")[-1] or ".bin"
    fname = f"{patient_id}_{prefix}{ext}"
    fpath = os.path.join(upload_dir, secure_filename(fname))
    file_storage.seek(0)
    file_storage.save(fpath)
    return fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLARE Flask backend")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=5000, help="Bind port")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)