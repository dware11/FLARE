"""
BRISC PREPROCESSING - Individual 2D Slices
Each image is saved as individual NPZ (no patient grouping)
Output: /scratch/bckk/flare/mri_brain/data/brisc_processed/
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging
from tqdm import tqdm

class Config:
    BRISC_DIR = '/scratch/bckk/flare/mri_brain/data/raw/brisc2025'
    MANIFEST_PATH = os.path.join(BRISC_DIR, 'manifest.csv')
    PROCESSED_DIR = '/scratch/bckk/flare/mri_brain/data/brisc_processed'
    
    TARGET_SIZE = (512, 512)
    
    TUMOR_LABEL_MAP = {
        'glioma': 0,
        'meningioma': 1,
        'no_tumor': 2,
        'pituitary': 3
    }

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'brisc_preprocessing.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class BRISCPreprocessor:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.stats = {
            'total_images': 0,
            'npz_saved': 0,
            'failed': 0,
            'errors': []
        }
    
    def load_manifest(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.MANIFEST_PATH)
        df['relative_path'] = df['relative_path'].str.replace('\\', '/', regex=False)
        df['relative_path'] = df['relative_path'].str.replace('/', os.sep, regex=False)
        brisc_dir = self.config.BRISC_DIR
        df['full_path'] = df['relative_path'].apply(lambda x: os.path.join(brisc_dir, x))
        self.logger.info(f"Loaded manifest: {len(df)} entries")
        return df
    
    def load_image(self, filepath: str) -> np.ndarray:
        try:
            pil_img = Image.open(filepath)
            
            if pil_img.mode == 'L':
                arr = np.array(pil_img, dtype=np.uint8)
            elif pil_img.mode == 'RGB':
                arr = np.array(pil_img, dtype=np.uint8)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                if np.array_equal(r, g) and np.array_equal(g, b):
                    arr = r
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                pil_img = pil_img.convert('L')
                arr = np.array(pil_img, dtype=np.uint8)
            
            return arr
        except Exception as e:
            raise ValueError(f"Failed to load {filepath}: {str(e)}")
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (self.config.TARGET_SIZE[1], self.config.TARGET_SIZE[0]),
                            interpolation=cv2.INTER_LINEAR)
        return resized
    
    def process_dataset(self) -> None:
        self.logger.info("="*70)
        self.logger.info("BRISC PREPROCESSING - INDIVIDUAL 2D SLICES")
        self.logger.info("="*70)
        
        df = self.load_manifest()
        
        # Filter: classification task, non-mask images only
        classification_df = df[
            (df['task'] == 'classification') & 
            (df['is_mask'] == False)
        ]
        
        self.logger.info(f"Processing {len(classification_df)} classification images\n")
        
        os.makedirs(self.config.PROCESSED_DIR, exist_ok=True)
        self.stats['total_images'] = len(classification_df)
        
        for idx, (_, row) in enumerate(tqdm(classification_df.iterrows(), total=len(classification_df), desc="Processing images")):
            filepath = row['full_path']
            
            if not os.path.exists(filepath):
                self.logger.warning(f"File not found: {filepath}")
                self.stats['failed'] += 1
                continue
            
            try:
                # Load image
                image = self.load_image(filepath)
                
                # Check for corruption
                if image.max() == 0:
                    self.logger.warning(f"Corrupted (all zeros): {filepath}")
                    self.stats['failed'] += 1
                    continue
                
                # Resize
                image_resized = self.resize_image(image)
                
                # Get metadata
                patient_id = int(row['index'])
                split = row['split']
                tumor_label = row['tumor_label']
                plane_label = row['plane_label']
                label = self.config.TUMOR_LABEL_MAP.get(tumor_label, -1)
                
                # Create filename: classification_[split]_[id].npz
                filename = f"classification_{split}_{patient_id:05d}.npz"
                filepath_npz = os.path.join(self.config.PROCESSED_DIR, filename)
                
                # Save NPZ
                np.savez_compressed(
                    filepath_npz,
                    image=image_resized.astype(np.uint8),
                    label=label,
                    view=plane_label,
                    tumor_label=tumor_label,
                    dataset='BRISC',
                    patient_id=patient_id,
                    split=split
                )
                
                self.stats['npz_saved'] += 1
            
            except Exception as e:
                self.logger.warning(f"Failed to process {filepath}: {str(e)}")
                self.stats['errors'].append((filepath, str(e)))
                self.stats['failed'] += 1
        
        # Log results
        self.logger.info("\n" + "="*70)
        self.logger.info("PREPROCESSING COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Total images: {self.stats['total_images']}")
        self.logger.info(f"NPZ files saved: {self.stats['npz_saved']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Output: {self.config.PROCESSED_DIR}")
        self.logger.info("="*70)
        
        # Verify
        npz_count = len(list(Path(self.config.PROCESSED_DIR).glob("classification_*.npz")))
        self.logger.info(f"\nVerification: {npz_count} NPZ files in output")

if __name__ == "__main__":
    config = Config()
    logger = setup_logging(config.PROCESSED_DIR)
    preprocessor = BRISCPreprocessor(config, logger)
    preprocessor.process_dataset()
