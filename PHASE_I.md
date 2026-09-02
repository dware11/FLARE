# FLARE — Phase I Implementation Baseline

**Fusion-based Learning for Automated Radiology and Epidemiology (FLARE)**  
**Senior Design — Phase I**

This document defines the implementation baseline that the FLARE team can confidently demonstrate, discuss, and defend at the end of Phase I. It separates implemented functionality from planned research extensions so that project presentations, partner discussions, and technical interviews accurately reflect the current system.

> **Research / clinical-use notice:** FLARE is a research prototype for clinical decision support. It is not a certified medical device, and AI outputs require human review.

## 1. Phase I Objective

Phase I establishes an end-to-end brain-imaging decision-support prototype that can:

1. accept brain CT and MRI inputs,
2. preprocess the inputs for their modality-specific models,
3. run classification inference,
4. generate visual explainability/localization outputs,
5. combine CT and MRI abnormality probabilities through late decision-level fusion,
6. return structured results through a Flask REST API,
7. display those results in a React/TypeScript clinician-facing interface,
8. place cases into a human review workflow, and
9. use reviewed abnormal-case data for prototype epidemiological visualization.

Breast imaging remains a future extension and should not be presented as a completed Phase I model pipeline.

## 2. Implemented Model Stack

| Task | Phase I implementation | Primary output |
| --- | --- | --- |
| Brain CT screening | DenseNet121 feature extractor + bidirectional GRU sequence model | Binary normal/abnormal probability |
| Brain MRI classification | EfficientNetB0 | 5-class probabilities: Glioma, Meningioma, Pituitary, No Tumor, Normal |
| Brain MRI 2D segmentation | MiT-B3 / SegFormer-style U-Net pipeline | Tumor mask, area, tissue percentage |
| Brain MRI 3D segmentation | 3D U-Net with InstanceNorm3d | Whole Tumor, Tumor Core, Enhancing Tumor volumes |
| Explainability | Grad-CAM | Attention heatmaps / overlays |
| Multi-modal integration | Decision-level late fusion | Unified abnormality score and review flag |

## 3. Late-Fusion Logic

The current fusion engine operates on **probabilities**, not raw images or intermediate feature tensors.

For MRI, the abnormality probability is calculated as:

```text
P_MRI_abnormal = P(glioma) + P(meningioma) + P(pituitary)
```

The normal/no-tumor classes are not counted as abnormal.

For a paired CT/MRI case:

```text
fusion_score = 0.4 × P_CT_abnormal + 0.6 × P_MRI_abnormal
```

Current threshold:

```text
fusion_score >= 0.50  -> abnormal
fusion_score <  0.50  -> normal
```

This is **late fusion / decision-level fusion** because the independently generated model predictions are combined after inference. FLARE is not currently performing feature-level fusion where hidden representations from CT and MRI networks are concatenated and jointly learned.

## 4. End-to-End Request Flow

```text
Clinician
   |
   v
React + TypeScript UI
   |
   | HTTP request using Fetch API
   | multipart/form-data for uploaded scans
   v
Flask REST API
   |
   +--> validate patient metadata, hospital and file format
   |
   +--> route to CT predictor, MRI predictor, or fusion orchestrator
   |
   +--> modality-specific preprocessing
   |
   +--> PyTorch model inference
   |
   +--> Grad-CAM / segmentation generation where applicable
   |
   +--> probability normalization and late fusion if paired modalities
   |
   +--> case/review record + audit information
   v
JSON response
   |
   v
React UI renders classification, confidence, localization and review state
```

## 5. API Integration Contract

The web application and inference layer communicate through REST-style HTTP endpoints.

### MRI

`POST /api/mri/predict`

- request encoding: `multipart/form-data`
- primary fields: `file`, `patient_id`, `hospitalId`, `modality`
- response: JSON containing MRI inference output plus case/review metadata

### CT

`POST /api/ct/predict`

- request encoding: `multipart/form-data`
- supported upload paths include NPZ, zipped DICOM study, JPEG and PNG
- response: JSON containing prediction, probabilities, confidence, Grad-CAM URL and case metadata

### CT + MRI fusion

`POST /api/fusion/predict`

- request encoding: `multipart/form-data`
- inputs: `ct_file`, `mri_file`, patient metadata
- output: JSON containing per-modality probabilities, weights, fused score, final label, confidence and visualization URLs

### Why `multipart/form-data`?

Medical scan uploads are binary files. `multipart/form-data` allows the browser to send files and text metadata in the same HTTP request without attempting to embed the binary scan directly inside JSON.

### Why JSON responses?

JSON provides a language-independent structured response that TypeScript can parse into UI state while the Python backend can generate it naturally.

## 6. Authentication and Application Security Boundary

The Phase I UI uses Auth0's React integration for authentication. Conceptually:

1. the browser redirects/authenticates through the identity provider,
2. Auth0 manages the authenticated application session/token lifecycle,
3. protected application pages can determine whether the user is authenticated,
4. API authorization can use bearer tokens/JWT validation as the backend security boundary is hardened.

The preliminary report describes OAuth 2.0 and JWT as the intended security architecture. The team should distinguish between **identity/login integration present in the UI** and any production-grade backend authorization controls that still require deployment validation.

## 7. Frontend Integration

The active UI is a Vite-built React + TypeScript application. It uses:

- React for component rendering and application state,
- TypeScript for compile-time type checking,
- React Router for client-side routes,
- Material UI for interface components,
- ECharts for data visualization,
- Leaflet / React-Leaflet for geographic visualization,
- Fetch API for backend communication,
- Auth0 React SDK for authentication.

The frontend should be described as the **presentation/orchestration client**, not as the machine-learning runtime. Model inference occurs behind the Flask API.

## 8. Backend Integration

The backend is the integration layer between the UI and machine-learning code. Its responsibilities include:

- defining HTTP endpoints,
- validating requests and uploads,
- calling modality-specific inference functions,
- translating Python/model outputs into API-friendly JSON,
- exposing generated visualization assets,
- executing fusion logic,
- attaching hospital/case/review metadata,
- supporting audit/review workflow behavior.

This separation is important: the Flask application does not replace the models. It **orchestrates** them.

## 9. Explainability

Grad-CAM is used to visualize areas that influenced a convolutional model's classification decision.

High-level process:

1. run the image through the network,
2. select the class score of interest,
3. calculate gradients of that score with respect to a convolutional feature map,
4. globally average the gradients to obtain channel importance weights,
5. combine the weighted feature maps,
6. apply ReLU so positive evidence is emphasized,
7. resize the heatmap to image size and overlay it on the scan.

Grad-CAM is an **attribution visualization**, not a guaranteed tumor segmentation. FLARE therefore keeps segmentation outputs and Grad-CAM explanations conceptually separate.

## 10. Segmentation vs. Classification

**Classification asks:** "What class does this scan belong to?"

**Segmentation asks:** "Which pixels/voxels belong to the suspected tumor region?"

These jobs require different output structures:

- classifier -> vector of class probabilities,
- 2D segmentation model -> pixel mask,
- 3D segmentation model -> voxel mask / tumor sub-region volumes.

This distinction matters in partner conversations because a high classification accuracy does not automatically imply precise tumor boundaries.

## 11. Phase I Validation Snapshot

The repository currently documents the following research results:

- CT in-domain screening: 87.00% accuracy, 86.20% sensitivity, 87.80% specificity, 0.9459 ROC-AUC.
- CT cross-domain CQ500 test: 71.62% accuracy, 0.7359 ROC-AUC.
- MRI 5-class classification: 97.80% accuracy, 0.978 weighted F1, 0.9992 macro ROC-AUC.
- 2D MRI segmentation: 0.877 mean Dice, 0.790 mean IoU.
- 3D MRI segmentation: WT 0.913 Dice, TC 0.840 Dice, ET 0.797 Dice.
- Early paired-fusion proof of concept: 20 tumor-positive patients / 79 paired slices.

These metrics should always be presented with their dataset and validation limitations. In particular, the MRI Normal class currently has an acquisition-protocol mismatch between normal control and tumor cohorts, and the paired fusion cohort does not yet contain the healthy controls needed to establish full specificity/false-positive performance.

## 12. Known Phase I Limitations

The following should be stated openly rather than hidden:

- breast imaging is not yet a completed implementation,
- paired fusion validation is small and tumor-positive only,
- MRI sequence/domain harmonization is still required,
- patient-stratified validation must be strengthened to reduce leakage risk,
- multi-sequence BraTS folder upload is not fully implemented through the current HTTP endpoint,
- production HIPAA compliance cannot be claimed solely from architecture choices; it requires validated deployment, operational, legal and governance controls,
- clinical use would require substantially more validation and regulatory work.

## 13. Phase II Research Direction

Phase II should focus on moving from a functioning research prototype toward a more defensible experimental system:

1. acquire / curate paired healthy CT + MRI controls,
2. enforce patient-level train/validation/test splitting,
3. harmonize MRI acquisition protocols,
4. benchmark cross-institution generalization,
5. compare late fusion against stronger fusion strategies,
6. calibrate model probabilities and decision thresholds,
7. strengthen backend JWT authorization and role enforcement,
8. formalize model cards, data cards and experiment tracking,
9. quantify uncertainty / disagreement cases,
10. evaluate epidemiological aggregation only on clinician-reviewed outputs.

## 14. One-Minute Technical Explanation

> FLARE Phase I is an end-to-end multimodal brain-imaging research prototype. CT and MRI are processed by separate specialized deep-learning pipelines because the modalities have different image characteristics and diagnostic roles. The CT side uses a DenseNet121 plus bidirectional GRU architecture to screen a study for abnormality, while MRI uses EfficientNetB0 for five-class tumor classification and separate segmentation pipelines for localization and volumetrics. Each model returns probabilities independently. If both modalities are available, the backend performs late decision-level fusion using a 40-percent CT and 60-percent MRI weighted abnormality score. A Flask REST API orchestrates inference and returns JSON to a React/TypeScript UI, while Grad-CAM and segmentation provide visual evidence for clinician review. The important Phase II work is improving domain harmonization, patient-level validation, paired healthy controls, security hardening and external generalization.

## 15. What Every Team Member Should Be Able to Explain

Before a partner meeting, every presenter should be able to answer:

- Why are CT and MRI processed by different models?
- What is the difference between classification, segmentation and Grad-CAM?
- Why is the fusion called late/decision-level fusion?
- Where do the 0.4/0.6 weights enter the system?
- What does `multipart/form-data` do?
- What does the Flask API do that React does not?
- What does React do that Flask does not?
- What is JSON and why is it used between frontend/backend?
- What is an HTTP POST request?
- What is an API endpoint?
- What is OAuth/Auth0 conceptually?
- What is a JWT / bearer token conceptually?
- What is Docker buying the project?
- What is Grad-CAM actually showing?
- What does Dice score measure?
- What is domain shift?
- What is data leakage and why does patient-level splitting matter?
- What evidence supports each performance number?
- What are the current limitations of the fusion experiment?
- What is implemented now versus proposed for Phase II?

---

**Status:** Phase I implementation baseline  
**Scope:** Brain CT/MRI prototype, explainability, late fusion, clinician review and epidemiological proof-of-concept  
**Next:** Phase II validation, harmonization, generalization and security hardening
