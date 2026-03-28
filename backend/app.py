"""
FLARE MOCK BACKEND (EDITABLE)

- Local mock API for frontend + senior design demo.
- Geotracker: in-memory review queue; only approved abnormal cases affect counts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import os
import uuid as _uuid
from pathlib import Path
from predict_mri import predict_mri

# ============================================================================
# App + CORS
# ============================================================================

app = Flask(__name__)

CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"]}},
)

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
        "latitude": 29.7079,
        "longitude": -95.3984,
    },
    "H002": {
        "id": "H002",
        "name": "Memorial Hermann - Texas Medical Center",
        "latitude": 29.7041,
        "longitude": -95.3995,
    },
    "H003": {
        "id": "H003",
        "name": "Baylor St. Luke's Medical Center",
        "latitude": 29.7047,
        "longitude": -95.3981,
    },
    "H004": {
        "id": "H004",
        "name": "Ben Taub Hospital",
        "latitude": 29.7136,
        "longitude": -95.3955,
    },
    "H005": {
        "id": "H005",
        "name": "Texas Children's Hospital",
        "latitude": 29.7073,
        "longitude": -95.4010,
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

SUPPORTED_MODALITIES = ["brain_ct", "brain_mri", "brain_brats"]

def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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

def api_mri_predict():
    """
    Brain MRI / BraTS prediction (real pipeline).
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
        return jsonify({"error": f"Invalid modality for MRI endpoint: {modality}",
                        "valid": list(MRI_MODALITIES)}), 400
    if not hospital_id or not hospital_exists(hospital_id):
        return jsonify({"error": "Missing or invalid hospitalId"}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    ext  = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in {".jpg", ".jpeg", ".png", ".npz"}:
        return jsonify({"error": "Invalid file type — use JPG, PNG, or NPZ"}), 400

    # Save uploaded file
    filename  = f"{patient_id}_{_uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Run real MRI pipeline
    result = predict_mri(file_path, modality=modality, patient_id=patient_id)
    if "error" in result:
        return jsonify(result), 500

    # MRI: Malignant or Benign = abnormal, Normal = not abnormal
    abnormal = result.get("result_class") in ("Malignant", "Benign")
    case_id  = None

    if abnormal:
        case_id = str(uuid4())
        site    = HOSPITAL_REGISTRY[hospital_id]
        conf    = float(result["confidence"])

        _review_cases[case_id] = {
            "caseId":          case_id,
            "hospitalId":      hospital_id,
            "hospitalName":    site["name"],
            "patient_id":      patient_id,
            "modality":        modality,
            "pred_label":      result["pred_label"],
            "result_class":    result.get("result_class"),
            "confidence":      conf,
            "probabilities":   result.get("probabilities"),
            "is_abnormal":     True,
            "severity":        mock_severity(conf),
            "review_status":   "pending",
            "createdAt":       _utc_iso(),
            "approvedAt":      None,
            "rejectedAt":      None,
            "reviewerId":      None,
            "reject_reason":   None,
            "segmentation":    result.get("segmentation"),
            "input_image_url": result.get("input_image_url"),
            "gradcam_url":     result.get("gradcam_url"),
            "gradcam_ready":   result.get("gradcam_ready", False),
        }
        _ehr_records[case_id] = {
            **_review_cases[case_id],
            "firstName": first_name,
            "lastName":  last_name,
            "dob":       dob,
            "signature": None,
        }

    return jsonify({
        **result,
        "hospitalId":      hospital_id,
        "caseId":          case_id,
        "review_required": abnormal,
    }), 200


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
def reviews_approve(case_id: str):
    """Idempotent: approving again does not double-count."""
    case = _review_cases.get(case_id)
    if not case:
        return jsonify({"error": "Unknown caseId", "code": "NOT_FOUND"}), 404

    body = request.get_json(silent=True) or {}
    reviewer_id = body.get("reviewerId", "anonymous-reviewer")

    if case["review_status"] == "approved":
        return jsonify({"ok": True, "caseId": case_id, "status": "approved", "note": "already approved"}), 200
    if case["review_status"] == "rejected":
        return jsonify({"error": "Case already rejected", "code": "INVALID_STATE"}), 409
    if case["review_status"] != "pending":
        return jsonify({"error": "Invalid review state", "code": "INVALID_STATE"}), 400

    case["review_status"] = "approved"
    case["approvedAt"] = _utc_iso()
    case["reviewerId"] = reviewer_id

    ehr = _ehr_records.get(case_id) 
    if ehr: 
        ehr["review_status"] = "approved"
        ehr["approvedAt"] = case["approvedAt"]
        ehr["rejectedAt"] = None 
        ehr["reviewerId"] = reviewer_id
        ehr["reject_reason"] = None 
        sig = body.get("signature")
        if sig is not None: 
            ehr["signature"] = sig 
    return jsonify({"ok": True, "caseId": case_id, "status": "approved"}), 200


@app.route("/api/reviews/<case_id>/reject", methods=["POST"])
def reviews_reject(case_id: str):
    case = _review_cases.get(case_id)
    if not case:
        return jsonify({"error": "Unknown caseId", "code": "NOT_FOUND"}), 404

    body = request.get_json(silent=True) or {}
    reviewer_id = body.get("reviewerId", "anonymous-reviewer")
    reason = body.get("reason")

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

    ehr = _ehr_records.get(case_id) 
    if ehr: 
        ehr["review_status"] = "rejected"
        ehr["rejectedAt"] = case["rejectedAt"]
        ehr["approvedAt"] = None 
        ehr["reviewerId"] = reviewer_id
        ehr["reject_reason"] = reason 
        sig = body.get("signature")
        if sig is not None:
            ehr["signature"] = sig
    return jsonify({"ok": True, "caseId": case_id, "status": "rejected"}), 200

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
                "hospitalId": hid,
                "name": site["name"],
                "latitude": site["latitude"],
                "longitude": site["longitude"],
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
                "totals": {"pending": total_pending, "approvedAbnormal": total_approved},
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
