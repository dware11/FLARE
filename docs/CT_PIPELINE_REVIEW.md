# CT Brain Pipeline — Code Review & Improvement Roadmap

## 1. What the Code Actually Does

### Model (`ml/brain/ct/model_ct.py`)
- **DenseNet121** (upgraded from ResNet18), ImageNet1K_V1 pretrained
- Final classifier replaced with `Linear(1024, 2)` for binary classification
- ~8M parameters; dense feature reuse improves gradient flow vs ResNet18

### Input / Preprocessing (`scripts/preprocess_ct.py`, `ml/brain/ct/ct_transforms.py`)
- DICOM → Hounsfield Units → **3 HU windows per slice** (pseudo-RGB):
  - Brain: center=40, width=80
  - Subdural: center=75, width=200
  - Bone: center=400, width=1000
- k=5 center slices selected per patient
- Each window: min-max normalized [0,1], resized to 256×256
- **ImageNet normalization** applied in `CTDataset.__getitem__` (mean/std per ImageNet stats)
- Cached as NPZ: shape `(5, 3, 256, 256)`

### Training (`ml/brain/ct/train_ct.py`)
- Slices flattened: `(B, k, C, H, W)` → `(B×k, C, H, W)`, each through DenseNet121
- Patient-level logits aggregated by mean (or max) across k slices
- **Differential LR**: backbone at `lr×0.1`, classifier head at `lr`
- CrossEntropyLoss (ignore_index=-1), AdamW, ReduceLROnPlateau
- 10 epochs default, batch size 4

### Dataset (`ml/brain/ct/dataset_ct.py`)
- CQ500 brain CT, binary labels (any abnormality = 1)
- 70/15/15 train/val/test split, seed=42
- Abnormalities: ICH, IPH, IVH, SDH, EDH, SAH, MassEffect, MidlineShift, Fracture

### Grad-CAM (`ml/brain/ct/gradcam_ct.py`)
- Hooks `model.features.denseblock4` (last DenseNet dense block)
- Produces 256×256 heatmap overlaid on brain-window image
- JET colormap, 50% alpha blend

### Fusion (`ml/brain/output/outputs.py`)
- `fuse_results(ct, mri, w_ct=0.5, w_mri=0.5)` — hardcoded equal weights
- Weighted average of prediction scores, threshold at 0.5

---

## 2. Mismatches Between Report Language and Code

| Claim (Likely in Report) | Reality in Code | Severity |
|---|---|---|
| "U-Net architecture" or "segmentation network" | Zero U-Net code in CT pipeline; pure classification DenseNet121 | **Critical** — remove entirely |
| "Localization of pathology via Grad-CAM" | Grad-CAM is saliency visualization, not localization (no bounding box/mask output) | **High** — rephrase |
| "2.5D input" | 3 channels = 3 HU windows of the **same** slice, not adjacent slices; this is multi-window pseudo-RGB | **Medium** — clarify |
| "Validated fusion weights" | Hardcoded 0.5/0.5, never tuned to validation data | **Medium** — overstatement |
| "Segmentation-assisted diagnosis" | No segmentation mask produced at any stage | **Critical** — remove |

---

## 3. Top 5 Improvements (Priority Order)

### 1. ✅ Swap Backbone: ResNet18 → DenseNet121 *(DONE)*
- **Why**: DenseNet121 has stronger precedent for medical imaging (CheXNet), dense feature reuse, comparable parameter count (~8M vs ~11M).
- **Fix applied**: `ml/brain/ct/model_ct.py`
- **Grad-CAM updated**: target layer changed to `features.denseblock4`

### 2. ✅ Fix ImageNet Normalization *(DONE)*
- **Problem**: ResNet18/DenseNet121 backbones expect inputs normalized with ImageNet stats. Raw [0,1] values misalign pretrained feature distributions.
- **Fix applied**: `ml/brain/ct/dataset_ct.py` — `TF.normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` applied per slice after multi-window normalization
- **Expected gain**: +2–5% AUC

### 3. ✅ Differential Learning Rate *(DONE)*
- **Problem**: Same lr=1e-4 applied to pretrained backbone and randomly-initialized head. Head should train faster.
- **Fix applied**: `ml/brain/ct/train_ct.py` — backbone at `lr×0.1`, classifier head at `lr`

### 4. Validation-Tuned Fusion Weights *(PENDING)*
- **Problem**: 0.5/0.5 CT/MRI weights are arbitrary. AUC of CT and MRI models likely differs significantly.
- **Fix**: After training both models, compute per-modality validation AUC:
  ```python
  w_ct  = auc_ct  / (auc_ct + auc_mri)
  w_mri = auc_mri / (auc_ct + auc_mri)
  ```
  Or: grid-search α ∈ [0,1] on validation set, save best α to config.
- **File**: `ml/brain/output/outputs.py` — add `tune_fusion_weights()` utility
- **Risk**: Very low. No architecture changes needed.

### 5. True 2.5D Slice Triplets (A/B Experiment) *(PENDING)*
- **Problem**: Current "2.5D" is 3 windows of the **same** slice → pseudo-RGB. True 2.5D = 3 adjacent slices as channels, giving through-plane context.
- **Fix**: Add preprocessing mode that stacks `(slice_n-1, slice_n, slice_n+1)` as 3 channels using brain-window only. Run as controlled A/B vs current multi-window approach.
- **Risk**: Medium — requires preprocessing change. Keep existing multi-window as baseline.
- **Files**: `scripts/preprocess_ct.py`, `ml/brain/ct/dataset_ct.py`

---

## 4. Safest Next Experiment

Fixes 1–3 are already applied. The next experiment is **zero-risk**:

```bash
# Re-train with all fixes
python ml/brain/ct/train_ct.py --epochs 10 --batch-size 4 --lr 1e-4

# Evaluate on test split
python ml/brain/ct/infer.py --split test --out-csv results/ct_v2_test.csv
```

Compare val AUC from new checkpoint against previous `ct_baseline_best.pt`.

---

## 5. Suggested Report Wording

### Replace "U-Net" language:
> ~~"We employ a U-Net architecture for CT analysis..."~~
>
> **"We employ DenseNet121 pretrained on ImageNet as a binary classification backbone, adapted for brain CT via multi-window HU preprocessing and multi-slice aggregation."**

### Replace "localization" language:
> ~~"Grad-CAM provides lesion localization..."~~
>
> **"Grad-CAM class activation mapping provides visual saliency explanations, highlighting regions most discriminative for the model's classification decision. This does not constitute pixel-level localization or segmentation."**

### Replace "2.5D" (if used ambiguously):
> ~~"We use 2.5D input slices..."~~
>
> **"We represent each CT scan using k=5 center axial slices, each encoded as a 3-channel pseudo-RGB image via brain (W=80), subdural (W=200), and bone (W=1000) Hounsfield Unit windowing. Slice-level predictions are aggregated at patient level via mean pooling."**

### Replace "validated fusion weights":
> ~~"Fusion weights are optimized..."~~
>
> **"Multimodal fusion currently uses equal weighting (w_CT=0.5, w_MRI=0.5) as a baseline. Future work will tune weights proportionally to per-modality validation AUC."**

---

## 6. Critical File Map

| File | Role |
|---|---|
| `ml/brain/ct/model_ct.py` | DenseNet121 classifier — ✅ updated |
| `ml/brain/ct/dataset_ct.py` | Data loading + ImageNet norm — ✅ updated |
| `ml/brain/ct/train_ct.py` | Training loop + differential LR — ✅ updated |
| `ml/brain/ct/gradcam_ct.py` | Grad-CAM, target `features.denseblock4` — ✅ updated |
| `ml/brain/ct/infer.py` | Inference pipeline |
| `ml/brain/output/outputs.py` | Fusion — weights need tuning (item #4) |
| `scripts/preprocess_ct.py` | Preprocessing — 2.5D experiment entry point (item #5) |
