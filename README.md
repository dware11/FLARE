# FLARE: Fusion-based Learning for Automated Radiology and Epidemiology

**A Multi-Modal Deep Learning System for Brain CT & MRI Tumor Screening, Grad-CAM Interpretability, and Epidemiological Surveillance**

> **Clinical Notice & Ongoing Research:**  
> FLARE is an evolving research prototype designed strictly for clinical decision support and is not a certified medical device. All model inferences, segmentation boundaries, and Grad-CAM heatmaps are exploratory aids intended to assist licensed clinicians and require mandatory human-in-the-loop review. 
> 
> This project remains under active development. Our research team is continually refining the underlying architectures, expanding paired multi-modal datasets, and enhancing domain harmonization—specifically addressing protocol variances in healthy control cohorts (e.g., T1 non-contrast vs. contrast-enhanced sequences) and evaluating patient-stratified generalizations[cite: 1].
>
> 🌐 **Interactive Web Interface:** [https://flare-woad.vercel.app/home](https://flare-woad.vercel.app/home)
---

## Overview

FLARE provides a unified multi-modal AI framework dedicated to **brain medical imaging**. The platform processes head CT and brain MRI scans independently, combines predictions via late fusion when dual modalities exist, visualizes model attention using Grad-CAM, and logs clinician-approved abnormal cases to an Electronic Health Record (EHR) review queue and regional outbreak surveillance map.

*(Note: Breast imaging modules are planned future extensions and are currently marked as Coming Soon in the platform).*

| Modality / Task | Model Architecture | Primary Dataset(s) | Output / Clinical Target |
| :--- | :--- | :--- | :--- |
| **Brain CT Screening** | DenseNet121 + BiGRU | RSNA ICH & CQ500 | Patient-level Normal vs. Abnormal flagging |
| **Brain MRI Classification** | EfficientNetB0 | BRISC 2025 + IXI | 5-Class: Glioma, Meningioma, Pituitary, No Tumor, Normal |
| **Brain MRI 2D Segmentation** | MiT-B3 UNet (SegFormer) | BRISC 2025 | Boundary mask, area ($mm^2$), and tissue percentage |
| **Brain MRI 3D Volumetrics** | 3D U-Net (InstanceNorm3d) | BraTS 2023 | Sub-regions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET) |
| **Multi-Modal Late Fusion** | Weighted Average ($0.4 \cdot P_{\text{CT}} + 0.6 \cdot P_{\text{MRI}}$) | JUH Paired CT/MRI Cohort | Unified abnormality decision & divergence review flagging |
| **Explainability (XAI)** | Grad-CAM | DenseNet121 & EfficientNetB0 | Feature attribution heatmaps and visual overlay |

---

## System Architecture

```text
       +-------------------------------+       +-------------------------------+
       |     Head CT Study (DICOM/NIfTI)|       |     Brain MRI Study (NPZ/JPG) |
       +---------------+---------------+       +---------------+---------------+
                       |                                       |
       +---------------v---------------+       +---------------v---------------+
       |  3-Channel HU Windowing       |       |  Preprocessing & Scaling      |
       | (Brain, Subdural, Bone)       |       | (Z-Score / Normalization)     |
       +---------------+---------------+       +---------------+---------------+
                       |                                       |
       +---------------v---------------+       +---------------v---------------+
       |     DenseNet121 + BiGRU       |       |        EfficientNetB0         |
       |       (P_CT_abnormal)         |       |      (5-Class Probabilities)  |
       +---------------+---------------+       +---------------+---------------+
                       |                                       |
                       |   +-----------------------------------+
                       |   | (P_MRI_abnormal = P(Glioma) + P(Meningioma) + P(Pituitary))
                       v   v
            +-----------------------------------+
            |        Late Fusion Engine         |
            | Score = 0.4(P_CT) + 0.6(P_MRI)    |
            +-----------------+-----------------+
                              |
              +---------------+---------------+
              |                               |
     [If Tumor Detected]             [All Decisions]
              |                               |
+-------------v-------------+   +-------------v-------------+
| Segmentation Pipelines    |   | Grad-CAM Visual Heatmaps  |
| • MiT-B3 UNet (2D Area)   |   | • Spatial attention maps  |
| • 3D U-Net (BraTS Volumes)|   | • Radiologist transparency|
+-------------+-------------+   +-------------+-------------+
              |                               |
              +---------------+---------------+
                              |
                +-------------v-------------+
                |  Flask REST API (Delta HPC)|
                +-------------+-------------+
                              |
                +-------------v-------------+
                |  React + TypeScript UI    |
                |  • EHR Clinician Review   |
                |  • Houston Outbreak Map   |
                +---------------------------+
```
## Validated Results & Benchmark Summary

**Brain CT Abnormality Screening (DenseNet121 + BiGRU)**
* **In-Domain (RSNA ICH):** **87.00%** accuracy, **86.20%** sensitivity, **87.80%** specificity, and **0.9459** test ROC-AUC (0.9595 validation AUC)[cite: 1].
* **Screening Threshold (0.488):** Tuned specifically to favor sensitivity and prevent missed abnormalities in clinical triage.
* **Cross-Domain (CQ500):** **71.62%** accuracy and **0.7359** ROC-AUC, reflecting realistic distribution shift across scanner hardware.

**Brain MRI 5-Class Tumor Classification (EfficientNetB0)**
* **Overall Test Performance:** **97.80%** accuracy, **0.978** weighted F1-score, and **0.9992** macro ROC-AUC across 1,000 held-out samples[cite: 1].
* **Per-Class AUC (One-vs-Rest):**
  * Normal: **1.0000** *(see sequence protocol note below)*
  * Pituitary: **0.9999**
  * No Tumor: **0.9998**
  * Glioma: **0.9985**
  * Meningioma: **0.9980**

> **Sequence Protocol & Domain Shift Note:**  
> The near-perfect metrics for the Normal class reflect an underlying acquisition protocol disparity: healthy control scans from IXI are T1 non-contrast sequences, whereas BRISC tumor cases are T1 contrast-enhanced (T1c) scans[cite: 1]. This introduces contrast-related distribution bias that inflates boundary separation[cite: 1]. We are actively addressing this domain shift by harmonizing sequence protocols and re-evaluating on contrast-matched control cohorts to ensure true anatomical discrimination[cite: 1].

**Tumor Segmentation & Volumetric Quantification**
* **2D Boundary Masking (MiT-B3 UNet on BRISC):** **0.877** Mean Dice and **0.790** Mean IoU.
* **3D Volumetric Sub-Regions (3D U-Net on BraTS 2023):**
  * Whole Tumor (WT): **0.913** Dice
  * Tumor Core (TC): **0.840** Dice
  * Enhancing Tumor (ET): **0.797** Dice

**Multi-Modal Late Fusion (JUH Cohort — Proof of Concept)**

| Pipeline | Model Architecture | Abnormal Slice Detection Rate | Dataset Scope |
| :--- | :--- | :--- | :--- |
| **CT Alone** | DenseNet121 + BiGRU | **73.4%** | 20 patients / 79 paired slices |
| **MRI Alone** | EfficientNetB0 | **98.7%** | 20 patients / 79 paired slices |
| **Fused Pipeline** | $0.4 \cdot P_{\text{CT}} + 0.6 \cdot P_{\text{MRI}}$ | **100.0%** | 20 patients / 79 paired slices |

> **Research Note on Fusion:**  
> The late fusion pipeline is an early-stage proof of concept. Because the JUH validation cohort contains only 20 tumor-positive cases, these figures measure **abnormality detection sensitivity only**. Evaluating full clinical specificity and false-positive rates requires acquiring paired healthy control datasets, which remains an active area of our research.

## Active Research & Ongoing Improvements

We are actively refining FLARE's modeling pipelines and dataset curation to improve clinical robustness:

* **MRI Domain Harmonization:** Addressing the sequence mismatch between IXI normal controls (T1 non-contrast) and BRISC tumor cases (T1 contrast-enhanced) through sequence normalization and acquisition of matched T1c healthy cohorts.
* **External Multi-Center MRI Benchmarking:** Evaluating cross-dataset generalization beyond single-dataset splits to assess real-world scanner variability.
* **Patient-Stratified Validation:** Transitioning from slice-level splits to fully patient-stratified partitions to prevent intra-patient data leakage.
* **Paired Fusion Control Expansion:** Seeking paired healthy CT and MRI studies to measure clinical specificity and false-positive rates alongside current tumor detection sensitivity.
* **3D Sequence Modeling for MRI:** Adapting volumetric sequence architectures (similar to the CT DenseNet + BiGRU pipeline) to process contiguous MRI slice stacks.

---

## Directory Structure

```text
FLARE/
├── artifacts/                     # Demo patient scans and manifests for web evaluation
│   └── brain_mri/demo_patients/
├── backend/                       # Flask API server & inference orchestrator
│   ├── app.py                     # Primary REST endpoints (/api/run-model, /api/health)
│   ├── predict_mri.py             # MRI inference glue and output formatting
│   └── run_flask.sbatch           # HPC Slurm deployment script
├── docs/                          # Architecture diagrams and setup guides
├── ml/
│   └── brain/                     # Core brain imaging pipelines
│       ├── ct/                    # DenseNet121 + BiGRU CT abnormality pipeline
│       ├── ct_3d/                 # 3D CT classification and volumetric preprocessing
│       └── mri/                   # 5-class EfficientNetB0, MiT-B3, BraTS 3D U-Net, Grad-CAM
├── results/                       # Confusion matrices, ROC curves, and metric logs
├── scripts/                       # HPC batch jobs, threshold sweeps, and data transforms
│   └── slurm/                     # SLURM execution scripts for NCSA Delta HPC
└── ui/                            # React 18 + TypeScript frontend application
    ├── src/
    │   ├── api/                   # Typed API clients (flareAPI.ts)
    │   ├── pages/                 # Cancer Detection, EHR Database, Outbreak Tracker
    │   └── auth/                  # Auth0 JWT authentication configuration
    └── vite.config.ts

```
## FLARE Platform — User Guide

This guide provides a comprehensive operational overview of how to navigate the FLARE web application for AI-assisted neuroimaging inference, physician-in-the-loop review, electronic health record logging, and regional epidemiological monitoring.

### 1. Authentication & Clinical Notices

* **Secure Authentication:** Access the platform login screen and sign in via Auth0 integration using authorized institutional credentials (e.g., hospital network credentials or Google authentication).
* **Privacy Policy Acknowledgement:** Using the platform requires acknowledging the privacy policy and HIPAA-aligned data protections displayed at the bottom of the interface[cite: 1].
* **Clinical Decision Support Notice:** FLARE explicitly presents a clinical disclaimer stating that all AI inferences, segmentation boundaries, and Grad-CAM overlays serve exclusively as clinical decision-support aids and require mandatory validation by a qualified medical professional.

---

### 2. AI Cancer Detection & Image Analysis

* **Clinician Verification Modal:** Before initiating an analysis run, an AI Assistance Notice modal appears[cite: 1]. You must confirm your role as a qualified clinician by selecting **Continue**[cite: 1].
* **Patient Metadata Entry:**
  * Select the medical facility from the hospital drop-down menu (e.g., *Baylor St. Luke's Medical Center*, *Houston Methodist Hospital*, *Memorial Hermann*)[cite: 1].
  * Select the target cancer type (**Brain** is fully operational; **Breast** imaging is marked as *Coming Soon*).
  * Enter the patient's first name, last name, unique Medical ID, and Date of Birth[cite: 1].
  * *Note:* Scans for patients under 18 years old are restricted due to specialized pediatric consent requirements[cite: 1].
* **Modality Configuration & Scan Upload:**
  * Choose the target pipeline: **Brain MRI**, **Brain CT**, or **CT + MRI Fusion**[cite: 1].
  * For isolated single modalities, choose between uploading a **Single Scan** or an entire **Patient Folder**[cite: 1].
  * Supported input formats include `.npz`, `.nii` (NIfTI), `.jpg`, `.png`, and `.dcm` (DICOM)[cite: 1].
  * For multi-modal fusion, upload the paired scans into their respective CT and MRI input containers[cite: 1].
* **Executing Inference:** Click **Run Scan**[cite: 1]. Initial model warmup passes require several seconds, while subsequent runs execute in real time[cite: 1].
* **Reviewing AI Visualizations & Metrics:**
  * Inspect the multi-panel visual suite containing the raw input scan, the **Grad-CAM attention heatmap**, and the **MiT-B3 / 3D U-Net segmentation overlay**.
  * Review quantitative metric cards displaying the overall classification label (e.g., *Glioma*, *Meningioma*, *Pituitary*, *Abnormal*, *Normal*), model confidence percentages, fusion weighted scores, and calculated tumor physical area ($mm^2$ or $mm^3$).

---

### 3. Electronic Health Record (EHR) Database

* **Centralized Registry:** Navigate to the **EHR Database** tab to inspect all patient scans, model predictions, and clinician audit trails across the hospital network.
* **Search & Multi-Field Filtering:** Search and filter records dynamically by patient name, Medical ID, cancer classification type, scanning modality, or AI diagnostic output.
* **Review Status Flags:** Cases awaiting clinical review are highlighted with dynamic pending visual badges[cite: 1].
* **Physician Review & Audit Workflow:**
  * Click on any patient record to open the detailed diagnostic sidebar[cite: 1].
  * Select **View Scan** to inspect the original neuroimaging file, or **View Localization** to examine the corresponding Grad-CAM or segmentation overlay.
  * Clinicians can formally validate the AI output by clicking **Approve** or **Reject**[cite: 1].
  * Authenticate the audit decision by typing clinician initials/signature (e.g., *Dr. Williams*) and selecting **Confirm Approval** to clear the pending flag and commit the record[cite: 1].
* **Exporting Clinical Reports:**
  * Select any fully reviewed patient case. *(Unreviewed or pending records cannot be exported)*[cite: 1].
  * Click **Export Selected Report** to generate a clinical dossier compiling patient demographics, scan metadata, AI classification scores, visual overlays, and the auditing physician's digital signature.

---

### 4. Outbreak Analytics

* **Longitudinal Trend Surveillance:** Open the **Outbreak Analytics** module to evaluate monthly epidemiological diagnostic trends across reporting healthcare networks.
* **Regional & Facility Filtering:** Filter analytical trend curves to view the macro Houston regional trajectory or zoom into facility-specific case lines (e.g., *Texas Children's Hospital*, *Memorial Hermann*)[cite: 1].
* **Statistical Deviation Tracking:** Track temporal fluctuations in abnormal case volumes to determine whether regional tumor detections are expanding or contracting relative to historical baselines[cite: 1].
* **Institutional Alert Statuses:** Monitor automated alert banners (such as *Critical Status* flags triggered when positive scan volumes surge significantly above expected baseline detection rates)[cite: 1].

---

### 5. Geo Tracker

* **GIS Map Interface:** Navigate to the **Geo Tracker** page to inspect an interactive map displaying geographic distribution across registered medical centers.
* **Color-Coded Severity Triage:** Institutional map markers are categorized by case volume thresholds[cite: 1]:
  * 🟢 **Green:** Normal baseline threshold (low volume of abnormal cases)[cite: 1].
  * 🟠 **Orange:** Moderate status threshold[cite: 1].
  * 🔴 **Red:** Critical status threshold (elevated abnormal volume significantly exceeding standard hospital baselines)[cite: 1].
* **Interactive Facility Summary Cards:** Click on any geographic pin to display a summary card containing the hospital name, confirmed abnormal rate, and current queue of pending scans awaiting review[cite: 1].


## Tech Stack & Architecture

| Layer | Technology | Direct Source Reference | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | React.js (v18) & TypeScript (v5.0) | Report Table 2.5[cite: 1]; Presentation Slide 5[cite: 2] | Type-safe clinical interface built with Vite |
| **UI Components** | Material UI (MUI v5) | Report Table 2.5[cite: 1]; Presentation Slide 5[cite: 2] | Component-based clinical user interface |
| **Data Visualization** | Apache ECharts (v6) | Report Table 2.5[cite: 1]; Presentation Slide 5[cite: 2] | Outbreak analytics charts and regional trend visualizations |
| **Authentication** | Auth0 & JWT Tokens | Report Table 2.5, Sec. 2.8[cite: 1]; Presentation Slide 17[cite: 2] | Secure user login and role-based session control |
| **Backend Framework** | Python Flask | Report Table 2.4[cite: 1]; Presentation Slide 5[cite: 2] | REST API routing and inference orchestration |
| **API Security** | Flask-Limiter & Security Headers | Report Table 2.4, Sec. 2.8[cite: 1]; Presentation Slide 17[cite: 2] | Rate limiting (20 req/min inference, 60 req/min global) and HIPAA headers (`no-store`, `DENY`, `nosniff`) |
| **ML & AI Frameworks** | PyTorch, OpenCV, NumPy, Scikit-learn | Report Fig. 2.2, Table 2.6[cite: 1]; Presentation Slide 5[cite: 2] | Core model architectures, tensor processing, and evaluation metrics |
| **HPC Compute** | NCSA Delta HPC (NVIDIA A40 GPUs) | Report Table 2.6, Sec. 2.10.3[cite: 1]; Presentation Slide 5[cite: 2] | High-performance model training and GPU-accelerated inference |
| **Job Scheduling** | SLURM | Report Table 2.6, Sec. 2.10.3[cite: 1]; Presentation Slide 5[cite: 2] | Batch script and backend lifecycle management |
| **Frontend Hosting** | Vercel | Report Table 2.6[cite: 1]; Presentation Slide 5[cite: 2] | Cloud deployment of the React web client |

---

### Core ML Pipelines (Report & Presentation Alignment)

* **Brain CT Abnormality Screening:** DenseNet121 feature extractor with a Bidirectional Gated Recurrent Unit (BiGRU) and Top-K mean pooling ($K=3$) trained on 3-channel Hounsfield Unit (HU) windowed slices (Brain, Subdural, Bone).
* **Brain MRI 5-Class Classification:** Transfer-learned EfficientNetB0 fine-tuned on multi-class brain tumor categories (Glioma, Meningioma, Pituitary, No Tumor, Normal).
* **2D MRI Tumor Boundary Delineation:** MiT-B3 UNet (SegFormer family) combining a Vision Transformer encoder with a UNet decoder to output pixel masks and physical tumor area ($mm^2$).
* **3D Volumetric Glioma Segmentation:** 3D U-Net configured with `InstanceNorm3d` across multi-sequence BraTS inputs (T1, T1ce, T2, FLAIR) to quantify Necrotic Core (NCR), Peritumoral Edema (ED), and Enhancing Tumor (ET) in $mm^3$.
* **Explainability (XAI):** Grad-CAM spatial heatmaps generated from the final convolutional blocks (DenseNet121 `denseblock` and EfficientNetB0 final feature extraction layer).
* **Multi-Modal Late Fusion:** Weighted ensemble ($0.4 \cdot P_{\text{CT}} + 0.6 \cdot P_{\text{MRI}}$) merging CT and MRI probabilities into a unified abnormality score ($T \ge 0.5$) with automatic flagging for divergent cases.
