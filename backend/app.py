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

from src.config import META 
from ml.brain.ct.infer import run_ct_for_patient, _load_manifest
from ml.brain.output.outputs import ModalityResult, fuse_results

app = Flask(__name__) 
CORS(app, resources={r"/api/*": {"origins": "*"}})
 
# Grad-Cam PNGs will be saved here by run_ct_for_patient() 
CAM_DIR = ROOT / "backend" / "cam" 
CAM_DIR.mkdir(parents=True, exist_ok=True)
 
# If frontend is from a different port uncomment this 
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        # Flask will build an empty response automatically
        return "", 204


@app.after_request
def add_cors(r): 
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization" 
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r 

@app.route("/api/patients", methods=["GET"]) 
def api_patients(): 
    """
    Returns: { "patients": ["CQ600-CT-0", ...]}
    Reads META/ct_processed_manifest.json via _load_manifest() 
    """

    manifest_path = META / "ct_processed_manifest.json" 
    try: 
        entries = _load_manifest(manifest_path)
        patients = [e.get("patient_id") for e in entries if e.get("patient_id")]
    except Exception: 
        patients = [] 
    return jsonify({"patients": patients}) 

@app.route("/api/predict", methods=["POST"]) 
def api_predict(): 
    """

    """
    data = request.get_json(silent=True) or {} 
    patient_id = (data.get("patient_id") or "").strip()
    if not patient_id: 
        return jsonify({"error": "missing patient_id"}), 400
    
    result = run_ct_for_patient(patient_id, cam_dir=CAM_DIR)
    if result is None:
        return jsonify({"error": "Patient not found or inference failed"}), 404

    # prediction MUST be float probability (p_abnormal), NOT class index 
    ct_result = ModalityResult( 
        modality="CT", 
        prediction=result["p_abnormal"], 
        label=result["label"], 
        model_version="ct_v1", 
        explainablity_path=result.get("cam_path"),
    )

    # MRI intentionally None for demo; fusion is designed to accept mri=None safely 
    fuse_out = fuse_results(ct=ct_result, mri=None, w_ct=1.0, w_mri=0.0) 

    fused_payload = dict(fuse_out) 
    fused_payload["fused_score"] = round(ct_result.prediction, 4) 
    fused_payload["flagged_for_review"] = False 
    
    return jsonify({
        "patient_id": patient_id, 
        "ct": result, 
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
    