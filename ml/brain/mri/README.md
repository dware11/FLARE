# FLARE MRI: Brain Tumor Classification & Segmentation

## Overview

The FLARE MRI module performs **morphological imaging analysis** on multi-sequence brain MRI scans. It combines three core capabilities:

1. **5-Class Tumor Classification** — EfficientNetB0 identifies glioma, meningioma, pituitary tumors, non-tumor lesions, and normal brain
2. **2D Tumor Segmentation** — MiT-B3 UNet delineates tumor boundaries on axial slices
3. **3D Volumetric Segmentation** — 3D U-Net segments glioma sub-regions (whole tumor, tumor core, enhancing tumor) for surgical planning

Grad-CAM visual explanations accompany every prediction, enabling clinicians to validate model decisions.

---

## Results Summary

### Classification (5-Class EfficientNetB0)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **97.8%** |
| **Weighted F1 Score** | 0.978 |
| **Macro AUC** | 0.9992 |
| **Test Set Size** | 1,000 slices |

**Per-Class Performance (One-vs-Rest ROC-AUC):**

| Class | F1 Score | Accuracy | AUC |
|-------|----------|----------|-----|
| Glioma | 0.9657 | 96.6% | 0.9993 |
| Meningioma | 0.9646 | 96.5% | 0.9976 |
| Pituitary | 0.9967 | 99.7% | 1.0000 |
| No Tumor | 0.9894 | 98.9% | 1.0000 |
| Normal (IXI) | — | — | 1.0000 |
| **Macro Average** | **0.978** | **97.8%** | **0.9992** |

**Dataset:** BRISC 2025 (glioma, meningioma, pituitary, no tumor) + IXI (normal healthy scans)

---

### 2D Segmentation (MiT-B3 UNet on BRISC)

| Metric | Value |
|--------|-------|
| **Dice Score** | **0.877** |
| **IoU (Jaccard)** | 0.790 |
| **Test Set** | ~600 slices (15% of 4,000) |

The MiT-B3 (SegFormer) encoder captures multi-scale context; the UNet decoder restores spatial resolution via skip connections, producing precise tumor boundary delineation.

---

### 3D Volumetric Segmentation (3D U-Net on BraTS 2023)

| Sub-Region | Description | Dice Score |
|------------|-------------|-----------|
| **Whole Tumor (WT)** | All labeled regions + edema | 0.913 |
| **Tumor Core (TC)** | Necrotic core + enhancing tumor | 0.840 |
| **Enhancing Tumor (ET)** | Actively growing boundary | 0.797 |
| **Average** | Across 3 sub-regions | **0.877** |

- **Test Set:** 202 held-out BraTS 2023 cases
- **Architecture:** 3D U-Net with InstanceNorm3d
- **Volumes Reported:** Necrotic core, edema, and enhancing tumor in both voxels and mm³ (1mm isotropic)

---

## Architecture & Training

### Classification Pipeline

**Model:** EfficientNetB0 (ImageNet pretrained)

**Training Scheme:** Two-phase transfer learning
- **Phase 1 (Epochs 1–5):** Freeze backbone, train classification head only
- **Phase 2 (Epochs 6–50):** Unfreeze all layers, fine-tune at 1×10⁻⁵ LR (10× reduced)

**Loss:** Cross-entropy  
**Optimizer:** AdamW with weight decay 1×10⁻⁴  
**LR Schedule:** Cosine annealing, min 1×10⁻⁶  
**Augmentation:** Random flips, 15° rotations, subtle shifts  
**Input:** 224×224 grayscale, normalized with ImageNet statistics  
**Early Stopping:** 10 epochs patience on validation weighted F1

**Note on Preprocessing:** Images are scaled to [0,255] then normalized with ImageNet statistics — a preprocessing quirk consistent between training and inference that does not affect metric validity but should be documented in clinical deployment.

### 2D Segmentation Pipeline

**Model:** MiT-B3 UNet (SegFormer encoder, ImageNet pretrained)

**Loss:** Combined Dice + BCE with pos_weight=10 (class imbalance)  
**Optimizer:** AdamW, LR 1×10⁻⁴, weight decay 1×10⁻⁴  
**Threshold:** 0.7 (optimized via grid search; training evaluated at 0.5)  
**Input:** 512×512 grayscale, normalized to [0,1]  
**Design:** Prioritizes sensitivity over specificity (clinically safer: slight over-prediction preferred to missing tumor pixels)

**Previous Baseline:** Attention U-Net (Oktay et al., 2018) achieving Dice 0.8771 is documented in `train_segmentation.py` for comparison.

### 3D Volumetric Segmentation Pipeline

**Model:** 3D U-Net with InstanceNorm3d

**Loss:** Dice + cross-entropy  
**Optimizer:** AdamW with weight decay  
**Normalization:** Z-score across 4 modalities (T1, T1ce, T2, FLAIR)  
**Preprocessing:** Tumor-aware crop/pad to 128³, handles over-sized tumors via lossless crop or ROI+zoom  
**Train/Val/Test Split:** 70% / 15% / 15%  
**Dataset:** BraTS 2023 GLI Challenge + BraTS-Africa cohort

---

### 3D Volumetric Segmentation Pipeline

**Model:** 3D U-Net with InstanceNorm3d

**Loss:** Dice + cross-entropy  
**Optimizer:** AdamW with weight decay  
**Normalization:** Z-score across 4 modalities (T1, T1ce, T2, FLAIR)  
**Preprocessing:** Tumor-aware crop/pad to 128³, handles over-sized tumors via lossless crop or ROI+zoom  
**Train/Val/Test Split:** 70% / 15% / 15%  
**Dataset:** BraTS 2023 GLI Challenge + BraTS-Africa cohort

---

## Directory Structure

```
## Directory Structure

## Directory Structure

ml/brain/mri/
├── README.md
├── __init__.py
├── requirements.txt
├── checkpoints/
│   └── training_curves.png
├── classification/
│   ├── brisc_summary.json                 # Dataset statistics
│   ├── explore_dataset.ipynb              # BRISC + IXI class distribution analysis
│   ├── job_mri_classification.slurm       # SLURM training job
│   ├── train_classification.py            # 5-class EfficientNetB0 training
│   └── figures/
│       ├── class_distribution.png
│       ├── intensity_distributions.png
│       ├── sample_images_per_class.png
│       ├── segmentation_masks.png
│       ├── size_distribution.png
│       └── view_distribution.png
├── explainability/
│   └── gradcam_classification.py          # Grad-CAM engine (EfficientNetB0)
├── preprocessing/
│   ├── brisc/
│   │   └── preprocess_brisc.py            # BRISC → grayscale/512×512/Otsu mask/NPZ
│   └── brats/
│       ├── config.py                      # Dataset paths & configuration
│       ├── create_splits.py               # 70/15/15 train/val/test
│       ├── preprocess_all.py              # Z-score norm, tumor-aware crop
│       ├── tumor_aware_crop.py            # Adaptive padding for large tumors
│       └── splits/
│           ├── train_cases.txt
│           ├── val_cases.txt
│           └── test_cases.txt
├── processed/                             # Sample BraTS test NPZ files (not full dataset)
│   ├── test_labels.npz
│   └── test/
│       ├── BraTS-GLI-00012-000.npz
│       ├── BraTS-GLI-00018-000.npz
│       ├── BraTS-GLI-00024-001.npz
│       ├── BraTS-GLI-00028-000.npz
│       ├── BraTS-GLI-00032-001.npz
│       ├── BraTS-GLI-00053-000.npz
│       ├── BraTS-GLI-00058-000.npz
│       ├── BraTS-GLI-00059-000.npz
│       ├── BraTS-GLI-00061-000.npz
│       ├── BraTS-GLI-00094-000.npz
│       ├── BraTS-GLI-00099-001.npz
│       └── BraTS-GLI-00101-000.npz
├── segmentation_brisc/
│   ├── ensemble_segmentation.py           # Soft-vote ensemble
│   ├── job_ensemble.slurm
│   ├── job_mri_segmentation.slurm
│   ├── job_swin_segmentation.slurm
│   ├── train_segmentation.py              # Attention U-Net baseline
│   ├── train_swin_unet.py                 # MiT-B3 UNet (deployed model)
│   ├── ensemble/                          # Ensemble results (best, not deployed)
│   │   ├── ensemble_predictions.png
│   │   ├── ensemble_results.json
│   │   └── model_comparison.png
│   ├── swin_unet/                         # MiT-B3 UNet results (deployed)
│   │   ├── prediction_samples.png
│   │   ├── results.json
│   │   └── training_curves.png
│   └── training/                          # Attention U-Net results
│       ├── prediction_samples.png
│       ├── results.json
│       └── training_curves.png
└── segmentation_brats/
    ├── data_understanding.ipynb
    ├── full_pipeline_demo.ipynb
    ├── test_evaluation.py                 # Volumetric Dice/IoU evaluation
    ├── train_local.py                     # 3D U-Net training
    ├── volume_quantification_slides.py    # Per-region volume calculations
    ├── presentation_figures/
    │   ├── clinical_report.png
    │   ├── manual_vs_ai.png
    │   ├── manual_vs_Model.png
    │   ├── volume_composition.png
    │   └── volume_quantification.png
    └── test_results/
        ├── test_results.png
        └── test_visualization.png
```

---

## Grad-CAM Explainability

**Implementation:** Integrated with EfficientNetB0 classification layer

- Computes class-specific activation gradients at final convolutional layer (`model.features[-1]`)
- Generates spatial heatmaps (per-pixel contribution to prediction)
- Output: PNG heatmap + blended overlay (yellow-to-red colormap per radiology convention)
- Timing: <2 seconds per scan

**Validation on Real Data:**
- True positives (tumor cases): Heatmaps highlight known tumor regions
- True negatives (normal cases): Distributed attention consistent with normal anatomy
- **Status:** Integrated and tested; ready for clinical evaluation

---

## Limitations & Known Issues

### Classification

- **IXI Protocol Mismatch:** IXI dataset uses T1-weighted non-contrast scans; BRISC uses T1-contrast-enhanced (T1ce). Both labeled as "T1" in training, creating subtle domain gap. Normal class is retained because it represents the critical "healthy" baseline.

- **Patient-Level Leakage:** Train/val/test splits are performed at slice level (not patient level) using manifest row indexing. Multiple slices from the same patient can appear across splits. This should be addressed in future work via patient-aware stratification.

### Segmentation

- **Threshold Mismatch:** Training and evaluation compute Dice/IoU at threshold=0.5; deployed API uses threshold=0.7. This was optimized for demonstration purposes (favors precision). Deployed-threshold metrics should be re-reported before publication.

- **Ensemble Not Deployed:** The best-performing ensemble (Dice 0.8914, soft-vote of Attention U-Net + MiT-B3) is implemented but not wired into the API. Only single MiT-B3 model is deployed.

### 3D Volumetric

- **Pre-Identified Glioma Cases:** BraTS inputs are pre-labeled as glioma; 3D U-Net performs only sub-region (WT/TC/ET) segmentation, not tumor detection. API response does not classify overall malignancy; it assumes glioma presence and segments sub-regions.

### Explainability

- **2D Classification Only:** Grad-CAM is implemented for 2D slice-level classification. 3D Grad-CAM for volumetric segmentation is planned as future work.

---

## How to Run

### Prerequisites
```bash
# Create environment
conda create -n flare_mri python=3.10
conda activate flare_mri
pip install -r requirements.txt

# Set paths (optional; code has fallbacks)
export FLARE_PROJECT_ROOT=/path/to/FLARE
export FLARE_BRATS_DATA=/path/to/brats/processed
```

### Classification Training
```bash
cd ml/brain/mri/classification
python train_classification.py \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --data_dir /path/to/brisc_processed /path/to/ixi_processed
```

### 2D Segmentation Training (MiT-B3)
```bash
cd ml/brain/mri/segmentation_brisc
python train_mit_b3_unet.py \
  --epochs 60 \
  --batch_size 8 \
  --lr 1e-4 \
  --data_dir /path/to/brisc_processed
```

### 3D Volumetric Segmentation (BraTS)
```bash
cd ml/brain/mri/segmentation_brats
python train_local.py \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4
```

### Inference (via Flask Backend)
The MRI module is invoked by `backend/predict_mri.py` when the Flask API receives a POST request to `/api/mri/predict`.

```python
# Example
from backend.predict_mri import predict_mri
result = predict_mri(mri_volume_path, modality='t1ce')
print(result['pred_label'], result['confidence'])
```

Response includes: `pred_label`, `confidence`, `probabilities`, `segmentation` (mask URLs, area metrics), `volumes` (if glioma detected), `gradcam_url`.

---

## Datasets

| Dataset | Task | Classes | Split | Source |
|---------|------|---------|-------|--------|
| BRISC 2025 | Classification | Glioma, Meningioma, Pituitary, No Tumor | Train/Val/Test | https://brisc-challenge.org/ |
| BRISC 2025 | 2D Segmentation | Binary (tumor/background) | Train/Val/Test | https://brisc-challenge.org/ |
| IXI | Normal class | Healthy brains | Test | https://brain-development.org/ixi-dataset/ |
| BraTS 2023 | 3D Segmentation | 4-class (bg/necrotic/edema/enhancing) | Train/Val/Test | https://www.synapse.org/#!Synapse:syn51156910/wiki/ |
| BraTS-Africa | 3D Segmentation | Sub-Saharan glioma cohort | Train | https://www.synapse.org/#!Synapse:syn51156910/wiki/ |

---

## Key References

- **EfficientNetB0:** Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML. DOI: 10.1109/CVPR.2017.243

- **SegFormer/MiT-B3:** Xie et al. (2021). "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS.

- **Attention U-Net:** Oktay et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas." arXiv:1804.03999.

- **3D U-Net:** Çiçek et al. (2016). "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." MICCAI.

- **BraTS Challenge:** Baid et al. (2021). "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification." arXiv:2107.02314.

- **Grad-CAM:** Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.


---

*Last Updated: August 22, 2026*  
*Metrics derived from Final Report, May 2026*