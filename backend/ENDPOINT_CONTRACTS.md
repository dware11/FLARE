# Backend Model Endpoint Contracts (Frozen Baseline)

This document captures the current request/response behavior before CT wrapper refactor.
It is intended as a demo-stability baseline.

## `POST /api/mri/predict`

- **Content type:** `multipart/form-data`
- **Fields:**
  - `file` (required)
  - `patient_id` (required)
  - `modality` (optional, default `brain_mri`; valid: `brain_mri`, `brain_brats`)
  - `hospitalId` (required)
  - `first_name`, `last_name`, `dob` (optional)
- **Special BraTS folder fields:** `t1n`, `t1c`, `t2w`, `t2f` (multipart quartet path currently returns 501)
- **Success status:** `200`
- **Success JSON:** MRI pipeline payload + `hospitalId`, `caseId`, `review_required`
- **Error status/JSON examples:**
  - `400 {"error":"Missing patient_id"}`
  - `400 {"error":"Invalid modality for MRI endpoint: ...","valid":[...]}`
  - `400 {"error":"Missing or invalid hospitalId"}`
  - `400 {"error":"No file uploaded"}`
  - `400 {"error":"...","code":"INVALID_FILE"}`
  - `400 {"error":"t1n/t1c/t2w/t2f multipart requires modality=brain_brats"}`
  - `400 {"error":"BraTS folder: all four fields required: ...","missing":[...]}`
  - `501 {"error":"BraTS multi-sequence NIfTI folder ingest is not implemented yet. ...","hint":"..."}`

## `POST /api/ct/predict`

- **Content type:** `multipart/form-data`
- **Fields:**
  - `file` (required)
  - `patient_id` (required)
  - `hospitalId` (optional, default `H001`)
  - `first_name`, `last_name`, `dob` (optional)
- **Accepted file types:** `.npz`, `.zip` (DICOM study), `.jpg`, `.jpeg`, `.png`
- **Success status:** `200`
- **Success JSON keys:**
  - `patient_id`, `modality`, `pred_label`, `result_class`, `confidence`
  - `p_normal`, `p_abnormal`, `cam_url`
  - `hospitalId`, `hospitalName`, `caseId`, `review_required`
  - `input_format` (`dicom_zip` | `npz` | `image`)
- **Error status/JSON examples:**
  - `400 {"error":"Missing patient_id"}`
  - `400 {"error":"Invalid hospitalId"}`
  - `400 {"error":"No file uploaded"}`
  - `400 {"error":"...","code":"INVALID_FILE"}`
  - `400 {"error":"Fusion CT file must be .npz volume, .zip DICOM study, or .jpg/.jpeg/.png image"}`
  - `400 {"error":"Uploaded file is not a valid ZIP archive"}`
  - `400 {"error":"ZIP contains no DICOM files (.dcm/.dicom). ..."}`
  - `500 {"error":"CT inference failed — check uploaded file"}`
  - `500 {"error":"CT inference failed: ..."}`

## `POST /api/fusion/predict`

- **Content type:** `multipart/form-data`
- **Fields:**
  - `patient_id` (required)
  - `hospitalId` (optional, default `H001`)
  - `ct_file` (optional)
  - `mri_file` (optional)
  - `first_name`, `last_name`, `dob` (optional)
- **Constraint:** At least one of `ct_file` or `mri_file` is required.
- **Success status:** `200`
- **Success JSON keys:**
  - `patient_id`, `fusion_mode`, `fusion_score`, `pred_label`, `result_class`
  - `confidence`, `is_abnormal`
  - `ct_prob`, `mri_prob`
  - `ct_weight`, `mri_weight`, `threshold`
  - `ct_cam_url`, `mri_input_url`, `mri_overlay_url`
  - `caseId`, `review_required`
  - `ct_details`, `mri_details`
- **Error status/JSON examples:**
  - `400 {"error":"Missing patient_id"}`
  - `400 {"error":"Invalid hospitalId"}`
  - `400 {"error":"At least one of ct_file or mri_file required"}`
  - `400 {"error":"...","code":"INVALID_FILE"}`
  - `400 {"error":"CT inference requires a .npz volume, .zip DICOM study, or .jpg/.jpeg/.png image"}`
  - `500 {"error":"CT inference failed: ..."}`
  - `500 {"error":"MRI inference failed: ..."}`
  - `500 {"error":"Both models returned no result"}`

## `POST /predict`

- **Content type:** `multipart/form-data`
- **Query string:** `cancer_type=brain|breast` (default `brain`)
- **Fields:**
  - `file` (required)
  - `patient_id` (optional; generated if missing)
  - `hospitalId` (optional; default `H001`)
  - `modality` (brain path only; defaults/falls back to `brain_mri`)
  - `first_name`, `last_name`, `dob` (optional)
- **Success status:** `200`
- **Success JSON (brain):**
  - `cancer_type`, `prediction`, `confidence`, `localization_url`, `gradcam_ready`
  - optional `probabilities`
- **Success JSON (breast):**
  - `cancer_type`, `prediction`, `confidence`, `localization_url`
- **Error status/JSON examples:**
  - `400 {"error":"No file uploaded"}`
  - `400 {"error":"...","code":"INVALID_FILE"}`
  - `400 {"error":"Unsupported cancer_type","supported":["brain","breast"]}`

