import os
import sys
from pathlib import Path
import pandas as pd

print("="*70)
print("PREPROCESSING DIAGNOSTIC CHECK - WITH PATH CONVERSION")
print("="*70)

# Check 1: Python version
print(f"\n1. Python Version: {sys.version.split()[0]}")

# Check 2: Import requirements
print("\n2. Checking imports...")
try:
    import numpy as np
    print("   ✓ numpy")
except:
    print("   ✗ numpy - MISSING")

try:
    import pandas as pd
    print("   ✓ pandas")
except:
    print("   ✗ pandas - MISSING")

try:
    from PIL import Image
    print("   ✓ PIL")
except:
    print("   ✗ PIL - MISSING")

try:
    import cv2
    print("   ✓ cv2 (opencv)")
except:
    print("   ✗ cv2 - MISSING")

# Check 3: BRISC directory
print("\n3. Checking BRISC directory...")
BRISC_DIR = '/scratch/bckk/flare/mri_brain/classification/raw/brisc2025'

if os.path.exists(BRISC_DIR):
    print(f"   ✓ BRISC_DIR exists: {BRISC_DIR}")
else:
    print(f"   ✗ BRISC_DIR NOT FOUND: {BRISC_DIR}")
    sys.exit(1)

# Check 4: Manifest file and path conversion
print("\n4. Checking manifest.csv and path conversion...")
MANIFEST_PATH = os.path.join(BRISC_DIR, 'manifest.csv')

if os.path.exists(MANIFEST_PATH):
    print(f"   ✓ manifest.csv exists")
    
    df = pd.read_csv(MANIFEST_PATH)
    print(f"   ✓ Loaded {len(df)} entries")
    
    # Show raw path
    raw_path = df.iloc[0]['relative_path']
    print(f"\n   Raw path from CSV:")
    print(f"     {repr(raw_path)}")
    
    # Convert path
    converted_path = raw_path.str.replace('\\', '/').str.replace('/', os.sep) if hasattr(raw_path, 'str') else raw_path.replace('\\', os.sep)
    if isinstance(raw_path, str):
        converted_path = raw_path.replace('\\', os.sep).replace('/', os.sep)
    
    print(f"\n   Converted path:")
    print(f"     {repr(converted_path)}")
    
    # Apply conversion to all
    df['relative_path'] = df['relative_path'].str.replace('\\', '/', regex=False).str.replace('/', os.sep, regex=False)
    if 'linked_image' in df.columns:
        df['linked_image'] = df['linked_image'].str.replace('\\', '/', regex=False).str.replace('/', os.sep, regex=False)
    
    df['full_path'] = df['relative_path'].apply(lambda x: os.path.join(BRISC_DIR, x))
    
    print(f"\n   Full path for first entry:")
    print(f"     {df.iloc[0]['full_path']}")
    
else:
    print(f"   ✗ manifest.csv NOT FOUND: {MANIFEST_PATH}")
    sys.exit(1)

# Check 5: Sample image file (with converted path)
print("\n5. Checking sample image files (with converted path)...")
classification_df = df[df['task'] == 'classification'].iloc[0]
sample_img_path = classification_df['full_path']

print(f"   Looking for: {sample_img_path}")

if os.path.exists(sample_img_path):
    print(f"   ✓ Sample classification image EXISTS")
    print(f"     File size: {os.path.getsize(sample_img_path) / 1024:.2f} KB")
    
    try:
        img = Image.open(sample_img_path)
        print(f"     Image mode: {img.mode}")
        print(f"     Image size: {img.size}")
    except Exception as e:
        print(f"     ✗ Error loading image: {e}")
else:
    print(f"   ✗ Sample image NOT FOUND")
    
    # Debug: check what directories exist
    parent_dir = os.path.dirname(sample_img_path)
    print(f"\n   Debug: Checking parent directory: {parent_dir}")
    
    if os.path.exists(parent_dir):
        print(f"     ✓ Parent directory exists")
        files = os.listdir(parent_dir)
        print(f"     Files in parent: {files[:5]}")
    else:
        print(f"     ✗ Parent directory does not exist")
        
        # Check one level up
        parent_parent = os.path.dirname(parent_dir)
        print(f"\n   Checking: {parent_parent}")
        if os.path.exists(parent_parent):
            print(f"     ✓ Exists, contents: {os.listdir(parent_parent)}")

# Check 6: Sample segmentation mask
print("\n6. Checking sample segmentation mask (with converted path)...")
segmentation_df = df[(df['task'] == 'segmentation') & (df['is_mask'] == True)].iloc[0]
sample_mask_path = segmentation_df['full_path']

print(f"   Looking for: {sample_mask_path}")

if os.path.exists(sample_mask_path):
    print(f"   ✓ Sample segmentation mask EXISTS")
    print(f"     File size: {os.path.getsize(sample_mask_path) / 1024:.2f} KB")
    
    try:
        mask = Image.open(sample_mask_path)
        print(f"     Mask mode: {mask.mode}")
        print(f"     Mask size: {mask.size}")
        
        mask_arr = np.array(mask)
        print(f"     Unique values: {sorted(np.unique(mask_arr))[:10]}")
    except Exception as e:
        print(f"     ✗ Error loading mask: {e}")
else:
    print(f"   ✗ Sample mask NOT FOUND")
    
    parent_dir = os.path.dirname(sample_mask_path)
    print(f"\n   Debug: Checking parent directory: {parent_dir}")
    if os.path.exists(parent_dir):
        print(f"     ✓ Parent directory exists")
        files = os.listdir(parent_dir)
        print(f"     Files in parent: {files[:5]}")

# Check 7: Output directory
print("\n7. Checking output directory...")
PROCESSED_DIR = '/scratch/bckk/flare/mri_brain/classification/processed'

if os.path.exists(PROCESSED_DIR):
    print(f"   ✓ Output directory exists: {PROCESSED_DIR}")
else:
    print(f"   ℹ Output directory will be created: {PROCESSED_DIR}")

# Check 8: Disk space
print("\n8. Checking disk space...")
try:
    stat = os.statvfs(BRISC_DIR)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    print(f"   ✓ Free disk space: {free_gb:.2f} GB")
except:
    print(f"   ✗ Could not check disk space")

print("\n" + "="*70)
if os.path.exists(classification_df['full_path']) and os.path.exists(segmentation_df['full_path']):
    print("✓ DIAGNOSTIC COMPLETE - ALL PATHS VERIFIED!")
    print("Ready to run: python 02_preprocess.py")
else:
    print("⚠️  PATH CONVERSION ISSUE - Check debug info above")
print("="*70)