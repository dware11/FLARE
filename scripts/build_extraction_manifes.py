""" Build a manifest of all patients that have been extracted. """

import json 
import sys 
from pathlib import Path 

ROOT = Path(__file__).resolve().parent.parent 
sys.path.insert(0, str(ROOT)) 
from src.config import RAW_CT, META 

DICOM_EXTS = (".dcm", ".dicom")

def main(): 
    META.mkdir(parents=True, exist_ok=True) 
    manifest = [] 
    for pdir in sorted(RAW_CT.iterdir()): 
        if not pdir.is_dir(): 
            continue 
        ct_dir = pdir / "CT_SELECTED" 
        if not ct_dir.is_dir(): 
            continue
        dicoms = [f for f in ct_dir.iterdir() if f.is_file() and f.suffix.lower() in DICOM_EXTS]
        if not dicoms: 
            continue
        manifest.append({
            "patient_id": pdir.name, 
            "path": str(ct_dir), 
            "slice_count": len(dicoms),
        })
    out_path = META / "ct_extraction_manifest.json" 
    with open(out_path, "w", encoding="utf-8") as f: 
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} entries to  {out_path}")

if __name__ == "__main__": 
    main()
    print("DONE.")