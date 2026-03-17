"""
FLARE MOCK BACKEND (EDITABLE)

- This file is a simple, local-only mock API.
- Safe to edit and extend for frontend development.
- No real models, databases, or auth.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import random

app = Flask(__name__)

# CORS: explicitly allow local frontend dev origins
CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}},
)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "version": "1.0",
            "mode": "MOCK",
        }
    ), 200


@app.route("/api/run-model", methods=["POST"])
def run_model():
    """Mock model inference endpoint - safe to tweak."""
    data = request.get_json(silent=True) or {}

    patient_id = data.get("patient_id")
    modality = data.get("modality")

    # Validation
    if not patient_id:
        return (
            jsonify(
                {
                    "error": "Missing required field: patient_id",
                    "code": "MISSING_PATIENT_ID",
                }
            ),
            400,
        )

    supported_modalities = ["brain_ct", "brain_mri", "breast_mammo", "breast_mri"]
    if modality not in supported_modalities:
        return (
            jsonify(
                {
                    "error": "Invalid modality",
                    "code": "INVALID_MODALITY",
                    "supported_modalities": supported_modalities,
                }
            ),
            400,
        )

    # Decide label based on last character of patient_id
    last_char = patient_id[-1]
    if last_char.isdigit() and int(last_char) % 2 == 1:
        pred_label = "abnormal"
    else:
        pred_label = "normal"

    # Confidence between 0.85 and 0.95
    confidence = round(random.uniform(0.85, 0.95), 4)

    # Simple probability distribution consistent with pred_label
    if pred_label == "abnormal":
        probabilities = {
            "normal": round(1.0 - confidence, 4),
            "abnormal": confidence,
        }
    else:
        probabilities = {
            "normal": confidence,
            "abnormal": round(1.0 - confidence, 4),
        }

    response = {
        "patient_id": patient_id,
        "modality": modality,
        "pred_label": pred_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "_mode": "MOCK",
    }
    return jsonify(response), 200


@app.route("/api/hospitals", methods=["GET"])
def get_hospitals():
    """Return mock geo + case data for Houston-area hospitals (editable)."""
    hospitals = [
        {
            "id": "H001",
            "name": "Houston Methodist Hospital",
            "latitude": 29.7079,
            "longitude": -95.3984,
            "total_cases": 1200,
            "high_risk_cases": 240,
            "high_risk_percentage": 20.0,
        },
        {
            "id": "H002",
            "name": "Memorial Hermann - Texas Medical Center",
            "latitude": 29.7041,
            "longitude": -95.3995,
            "total_cases": 950,
            "high_risk_cases": 190,
            "high_risk_percentage": 20.0,
        },
        {
            "id": "H003",
            "name": "Baylor St. Luke's Medical Center",
            "latitude": 29.7047,
            "longitude": -95.3981,
            "total_cases": 780,
            "high_risk_cases": 156,
            "high_risk_percentage": 20.0,
        },
        {
            "id": "H004",
            "name": "Ben Taub Hospital",
            "latitude": 29.7136,
            "longitude": -95.3955,
            "total_cases": 620,
            "high_risk_cases": 155,
            "high_risk_percentage": 25.0,
        },
        {
            "id": "H005",
            "name": "Texas Children's Hospital",
            "latitude": 29.7073,
            "longitude": -95.4010,
            "total_cases": 450,
            "high_risk_cases": 68,
            "high_risk_percentage": 15.1,
        },
    ]
    return jsonify(hospitals), 200


if __name__ == "__main__":
    # Dev-only server; keep simple and explicit for frontend work.
    app.run(host="127.0.0.1", port=5000, debug=True)