from __future__ import annotations 
from dataclasses import dataclass 
from typing import Optional, Dict, Any

# Creating dataclass so Outputs are Standardized for Models 
@dataclass 
class ModalityResult: 
    modality: str           # ex, "CT" or "MRI"
    prediction: float       # probability score, ex, 0.84
    label: str              # ex, "tumor" or "normal"
    model_version: str      # ex, "ct_v1"
    explainablity_path: Optional[str] = None    # Grad-CAM or segmentation file
    extra: Optional[Dict[str, Any]] = None      # optional (tumor size, notes, etc)

# Function that fuses results (weights predetermined)
def fuse_results (ct=None, mri=None, w_ct=0.5, w_mri=0.5): 
    contributors = {}
    final_prediction = 0.0 
    used_modalities = [] 

    if ct: 
        contributors["CT"] = ct.prediction
        final_prediction += w_ct * ct.prediction 
        used_modalities.append("CT")

    if mri: 
        contributors["MRI"] = mri.prediction 
        final_prediction += w_mri * mri.prediction 
        used_modalities.append("MRI")
    
    final_label = "tumor" if final_prediction >= 0.5 else "normal"

    return{
        "final_predictions": round(final_prediction, 4), 
        "final_label": final_label, 
        "used_modalities": used_modalities, 
        "contributors": contributors, 
        "weights": {"CT": w_ct, "MRI": w_mri},
    }
    