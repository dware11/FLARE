import io
import sys
import unittest
from pathlib import Path
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Test-only stub so app import does not require local MRI checkpoint files.
stub_predict_mri = types.ModuleType("predict_mri")
stub_predict_mri.predict_mri = lambda *args, **kwargs: {
    "pred_label": "normal",
    "result_class": "Normal",
    "confidence": 0.5,
}
stub_predict_mri._load_cls_model = lambda: None
stub_predict_mri._load_seg_model = lambda: None
stub_predict_mri._load_brats_model = lambda: None
sys.modules.setdefault("predict_mri", stub_predict_mri)

stub_flask_limiter = types.ModuleType("flask_limiter")


class _Limiter:
    def __init__(self, *args, **kwargs):
        pass

    def limit(self, *args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator


stub_flask_limiter.Limiter = _Limiter
sys.modules.setdefault("flask_limiter", stub_flask_limiter)

stub_flask_limiter_util = types.ModuleType("flask_limiter.util")
stub_flask_limiter_util.get_remote_address = lambda: "127.0.0.1"
sys.modules.setdefault("flask_limiter.util", stub_flask_limiter_util)

from app import app


class ModelEndpointContractsTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("app._run_mri_full_pipeline")
    def test_api_mri_predict_success_contract(self, mock_run_pipeline):
        mock_run_pipeline.return_value = (
            {
                "patient_id": "MRI_001",
                "pred_label": "abnormal",
                "result_class": "Malignant",
                "confidence": 0.91,
                "probabilities": {"normal": 0.09, "abnormal": 0.91},
                "hospitalId": "H001",
                "caseId": "case_1",
                "review_required": True,
            },
            200,
        )

        resp = self.client.post(
            "/api/mri/predict",
            data={
                "patient_id": "MRI_001",
                "modality": "brain_mri",
                "hospitalId": "H001",
                "file": (io.BytesIO(b"mock"), "scan.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIsInstance(body, dict)
        for key in (
            "patient_id",
            "pred_label",
            "result_class",
            "confidence",
            "probabilities",
            "hospitalId",
            "caseId",
            "review_required",
        ):
            self.assertIn(key, body)

    @patch("app.predict_ct")
    def test_api_ct_predict_success_contract(self, mock_predict_ct):
        mock_predict_ct.return_value = {
            "label": "abnormal",
            "confidence": 0.82,
            "p_normal": 0.18,
            "p_abnormal": 0.82,
            "cam_path": "backend/static/cam/P001_ct.png",
            "input_format": "image",
            "inference_path": "backend/static/uploads/P001_ct.png",
        }

        resp = self.client.post(
            "/api/ct/predict",
            data={
                "patient_id": "P001",
                "hospitalId": "H001",
                "file": (io.BytesIO(b"mock"), "ct.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIsInstance(body, dict)
        for key in (
            "patient_id",
            "modality",
            "pred_label",
            "result_class",
            "confidence",
            "p_normal",
            "p_abnormal",
            "cam_url",
            "hospitalId",
            "hospitalName",
            "caseId",
            "review_required",
            "input_format",
        ):
            self.assertIn(key, body)

    @patch("app.predict_mri")
    @patch("app.predict_ct")
    def test_api_fusion_predict_success_contract(self, mock_predict_ct, mock_predict_mri):
        mock_predict_ct.return_value = {
            "label": "abnormal",
            "confidence": 0.85,
            "p_normal": 0.15,
            "p_abnormal": 0.85,
            "cam_path": "backend/static/cam/P001_ct.png",
            "input_format": "image",
            "inference_path": "backend/static/uploads/P001_ct.png",
        }
        mock_predict_mri.return_value = {
            "pred_label": "meningioma",
            "result_class": "Benign",
            "confidence": 0.90,
            "probabilities": [
                {"label": "meningioma", "value": 0.6},
                {"label": "glioma", "value": 0.2},
                {"label": "pituitary", "value": 0.1},
                {"label": "no_tumor", "value": 0.05},
                {"label": "normal", "value": 0.05},
            ],
            "input_image_url": "/static/uploads/mri.png",
            "segmentation": {"overlay_url": "/static/masks/overlay.png"},
        }

        resp = self.client.post(
            "/api/fusion/predict",
            data={
                "patient_id": "P001",
                "hospitalId": "H001",
                "ct_file": (io.BytesIO(b"ct"), "ct.png"),
                "mri_file": (io.BytesIO(b"mri"), "mri.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIsInstance(body, dict)
        for key in (
            "patient_id",
            "fusion_mode",
            "fusion_score",
            "pred_label",
            "result_class",
            "confidence",
            "is_abnormal",
            "ct_prob",
            "mri_prob",
            "ct_weight",
            "mri_weight",
            "threshold",
            "ct_cam_url",
            "mri_input_url",
            "mri_overlay_url",
            "caseId",
            "review_required",
            "ct_details",
            "mri_details",
        ):
            self.assertIn(key, body)

    @patch("app._run_mri_full_pipeline")
    def test_legacy_predict_brain_success_contract(self, mock_run_pipeline):
        mock_run_pipeline.return_value = (
            {
                "pred_label": "abnormal",
                "result_class": "Malignant",
                "confidence": 0.91,
                "probabilities": {"normal": 0.09, "abnormal": 0.91},
                "segmentation": {"overlay_url": "/static/masks/overlay.png"},
                "gradcam_ready": True,
            },
            200,
        )

        resp = self.client.post(
            "/predict?cancer_type=brain",
            data={
                "patient_id": "P001",
                "hospitalId": "H001",
                "modality": "brain_mri",
                "file": (io.BytesIO(b"mock"), "brain.png"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIsInstance(body, dict)
        for key in ("cancer_type", "prediction", "confidence", "localization_url", "gradcam_ready"):
            self.assertIn(key, body)


if __name__ == "__main__":
    unittest.main()
