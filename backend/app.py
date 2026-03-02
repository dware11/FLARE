""" 
Flask backend for Monday demo (CT-only) 

Endpoints: 
GET /api/patients 
POST /api/predict
GET /api/cam/<patient_id>.png 
""" 

import sys 
from pathlib import Path 

ROOT = Path(__file__).resolve().parents[1] # FLARE/ 

sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify, send_from_directory 
from flask_cors import CORS

app = Flask(__name__) 
CORS(app, resources={r"/api/*": {"origins": "*"}})
 
# Grad-Cam PNGs will be saved here by run_ct_for_patient() 
CAM_DIR = ROOT / "backend" / "cam" 
CAM_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/api/patients", methods=["GET"]) 
def api_patients(): 
    """
    Hard-coded patient IDs for demo.
    """
    patients = [
        "CQ500-CT-0",
        "CQ500-CT-1",
        "CQ500-CT-2",
        "Demo-Patient-A",
        "Demo-Patient-B",
    ]
    return jsonify({"patients": patients}) 

@app.route("/api/predict", methods=["POST"]) 
def api_predict(): 
    """
    Mock prediction endpoint for demo.
    """
    data = request.get_json(silent=True) or {} 
    patient_id = (data.get("patient_id") or "").strip() or "UNKNOWN"

    p_abnormal = 0.82
    label = "abnormal"

    ct_result = {
        "p_abnormal": p_abnormal,
        "label": label,
        "model_version": "ct_mock_v1",
        "cam_path": None,
    }

    fused_payload = {
        "modality": "CT",
        "fused_score": round(p_abnormal, 4),
        "flagged_for_review": p_abnormal > 0.8,
        "details": {"note": "Mock fused result for demo only"},
    }

    return jsonify({
        "patient_id": patient_id, 
        "ct": ct_result, 
        "fused": fused_payload,
    })

@app.route("/api/cam/<patient_id>.png", methods=["GET"])
def api_cam(patient_id): 
    """\
    Serves backend/cam/<patient_id>.png
    Frontend: http://localhost:5000/api/cam/<id>.png
    """
    return send_from_directory(str(CAM_DIR), f"{patient_id}.png") 

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=5000, debug=True) 
    