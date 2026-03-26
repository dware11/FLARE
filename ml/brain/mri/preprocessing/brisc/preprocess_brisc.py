import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional, Union


class Config:
    BRISC_DIR = '/scratch/bckk/flare/mri_brain/classification/raw/brisc2025'
    MANIFEST_PATH = os.path.join(BRISC_DIR, 'manifest.csv')
    PROCESSED_DIR = '/scratch/bckk/flare/mri_brain/classification/processed'
    
    TARGET_SIZE = (512, 512)
    MIN_TUMOR_SIZE = (16, 16)
    
    RANDOM_SEED = 42
    
    TUMOR_LABEL_MAP = {
        'glioma': 0,
        'meningioma': 1,
        'no_tumor': 2,
        'pituitary': 3
    }
    
    VIEW_MAP = {
        'axial': 'axial',
        'coronal': 'coronal',
        'sagittal': 'sagittal'
    }


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'preprocessing.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class ImageLoader:
    @staticmethod
    def load_and_convert_to_grayscale(image_path: Union[str, Path]) -> np.ndarray:
        try:
            pil_img = Image.open(image_path)
            
            if pil_img.mode == 'L':
                arr = np.array(pil_img, dtype=np.uint8)
                return arr
            
            elif pil_img.mode == 'RGB':
                arr = np.array(pil_img, dtype=np.uint8)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                
                if np.array_equal(r, g) and np.array_equal(g, b):
                    return r
                else:
                    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                    return gray
            
            elif pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')
                arr = np.array(pil_img, dtype=np.uint8)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                return gray
            
            else:
                pil_img = pil_img.convert('L')
                arr = np.array(pil_img, dtype=np.uint8)
                return arr
        
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
    
    @staticmethod
    def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
        try:
            pil_mask = Image.open(mask_path)
            
            if pil_mask.mode != 'L':
                pil_mask = pil_mask.convert('L')
            
            arr = np.array(pil_mask, dtype=np.uint8)
            return arr
        
        except Exception as e:
            raise ValueError(f"Failed to load mask {mask_path}: {str(e)}")


class ImageResizer:
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        resized = cv2.resize(image, (target_size[1], target_size[0]),
                            interpolation=cv2.INTER_LINEAR)
        return resized
    
    @staticmethod
    def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        resized = cv2.resize(mask.astype(np.float32),
                            (target_size[1], target_size[0]),
                            interpolation=cv2.INTER_NEAREST)
        return resized.astype(np.uint8)


class MaskAnalyzer:
    @staticmethod
    def get_tumor_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        non_zero = np.where(mask > 0)
        
        if len(non_zero[0]) == 0:
            return None
        
        y_min, y_max = non_zero[0].min(), non_zero[0].max()
        x_min, x_max = non_zero[1].min(), non_zero[1].max()
        
        return (x_min, y_min, x_max, y_max)
    
    @staticmethod
    def get_tumor_size_after_resize(bbox: Tuple[int, int, int, int],
                                   original_shape: Tuple[int, int],
                                   target_size: Tuple[int, int]) -> Tuple[int, int]:
        if bbox is None:
            return (0, 0)
        
        x_min, y_min, x_max, y_max = bbox
        orig_h, orig_w = original_shape
        target_h, target_w = target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        width = int((x_max - x_min) * scale_x)
        height = int((y_max - y_min) * scale_y)
        
        return (width, height)
    
    @staticmethod
    def apply_otsu_binarization(mask: np.ndarray) -> np.ndarray:
        if mask.max() == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        
        thresh, binary_mask = cv2.threshold(mask, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        binary_mask = (binary_mask > 0).astype(np.uint8)
        return binary_mask


class DataValidator:
    @staticmethod
    def validate_image_shape(image: np.ndarray) -> bool:
        return len(image.shape) == 2
    
    @staticmethod
    def validate_mask_shape(mask: np.ndarray) -> bool:
        return len(mask.shape) == 2
    
    @staticmethod
    def check_image_corrupt(image: np.ndarray) -> bool:
        if image.max() == 0:
            return True
        if np.all(image == image[0, 0]):
            return True
        return False


class BRISCPreprocessor:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.image_loader = ImageLoader()
        self.resizer = ImageResizer()
        self.mask_analyzer = MaskAnalyzer()
        self.validator = DataValidator()
        
        self.stats = {
            'total_processed': 0,
            'classification_processed': 0,
            'segmentation_processed': 0,
            'failed': 0,
            'tumor_size_warnings': 0,
            'errors': []
        }
        
        self.tumor_bbox_stats = {
            'glioma': [],
            'meningioma': [],
            'pituitary': []
        }
    
    def load_manifest(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.MANIFEST_PATH)
        
        # First convert backslashes to forward slashes
        df['relative_path'] = df['relative_path'].str.replace('\\', '/', regex=False)
        
        if 'linked_image' in df.columns:
            df['linked_image'] = df['linked_image'].str.replace('\\', '/', regex=False)
        
        # Then convert to system-appropriate separators
        df['relative_path'] = df['relative_path'].str.replace('/', os.sep, regex=False)
        
        if 'linked_image' in df.columns:
            df['linked_image'] = df['linked_image'].str.replace('/', os.sep, regex=False)
        
        # Store locally to avoid lambda scope issues
        brisc_dir = self.config.BRISC_DIR
        df['full_path'] = df['relative_path'].apply(lambda x: os.path.join(brisc_dir, x))
        
        self.logger.info(f"Loaded manifest with {len(df)} entries")
        return df
    
    def process_classification_image(self, image_path: str, row: pd.Series) -> Optional[Dict]:
        try:
            image = self.image_loader.load_and_convert_to_grayscale(image_path)
            
            if not self.validator.validate_image_shape(image):
                raise ValueError("Invalid image shape")
            
            if self.validator.check_image_corrupt(image):
                raise ValueError("Image appears corrupted (all zeros or uniform)")
            
            image_resized = self.resizer.resize_image(image, self.config.TARGET_SIZE)
            
            label = self.config.TUMOR_LABEL_MAP.get(row['tumor_label'], -1)
            view = row['plane_label']
            
            return {
                'image': image_resized,
                'mask': None,
                'label': label,
                'view': view,
                'tumor_label': row['tumor_label'],
                'sample_id': f"classification_{row['split']}_{row['index']:05d}",
                'task': 'classification'
            }
        
        except Exception as e:
            self.logger.warning(f"Failed {row['filename']}: {str(e)}")
            self.stats['errors'].append((row['filename'], str(e)))
            return None
    
    def process_segmentation_pair(self, image_path: str, mask_path: str,
                                  row: pd.Series) -> Optional[Dict]:
        try:
            if not os.path.exists(mask_path):
                raise ValueError(f"Mask not found: {mask_path}")
            
            image = self.image_loader.load_and_convert_to_grayscale(image_path)
            mask = self.image_loader.load_mask(mask_path)
            
            if not self.validator.validate_image_shape(image):
                raise ValueError("Invalid image shape")
            
            if not self.validator.validate_mask_shape(mask):
                raise ValueError("Invalid mask shape")
            
            if image.shape != mask.shape:
                raise ValueError(f"Shape mismatch: {image.shape} vs {mask.shape}")
            
            if self.validator.check_image_corrupt(image):
                raise ValueError("Image appears corrupted")
            
            bbox_before = self.mask_analyzer.get_tumor_bounding_box(mask)
            tumor_size_before = self.mask_analyzer.get_tumor_size_after_resize(
                bbox_before, image.shape, self.config.TARGET_SIZE
            )
            
            if bbox_before and tumor_size_before[0] > 0 and tumor_size_before[1] > 0:
                if tumor_size_before[0] < self.config.MIN_TUMOR_SIZE[0] or \
                   tumor_size_before[1] < self.config.MIN_TUMOR_SIZE[1]:
                    self.logger.warning(
                        f"Small tumor: {row['filename']} -> {tumor_size_before} after resize"
                    )
                    self.stats['tumor_size_warnings'] += 1
                
                if row['tumor_label'] in self.tumor_bbox_stats:
                    self.tumor_bbox_stats[row['tumor_label']].append(tumor_size_before)
            
            image_resized = self.resizer.resize_image(image, self.config.TARGET_SIZE)
            mask_resized = self.resizer.resize_mask(mask, self.config.TARGET_SIZE)
            
            mask_binary = self.mask_analyzer.apply_otsu_binarization(mask_resized)
            
            label = self.config.TUMOR_LABEL_MAP.get(row['tumor_label'], -1)
            view = row['plane_label']
            
            return {
                'image': image_resized,
                'mask': mask_binary,
                'label': label,
                'view': view,
                'tumor_label': row['tumor_label'],
                'sample_id': f"segmentation_{row['split']}_{row['index']:05d}",
                'task': 'segmentation'
            }
        
        except Exception as e:
            self.logger.warning(f"Failed {row['filename']}: {str(e)}")
            self.stats['errors'].append((row['filename'], str(e)))
            return None
    
    def process_dataset(self) -> List[Dict]:
        df = self.load_manifest()
        
        processed = []
        
        classification_df = df[df['task'] == 'classification']
        self.logger.info(f"Processing {len(classification_df)} classification images")
        
        for _, row in tqdm(classification_df.iterrows(), desc="Classification", total=len(classification_df)):
            if not row['is_mask']:
                result = self.process_classification_image(row['full_path'], row)
                if result:
                    processed.append(result)
                    self.stats['classification_processed'] += 1
                self.stats['total_processed'] += 1
        
        segmentation_df = df[df['task'] == 'segmentation']
        seg_masks = segmentation_df[segmentation_df['is_mask'] == True]
        
        self.logger.info(f"Processing {len(seg_masks)} segmentation image-mask pairs")
        
        for _, mask_row in tqdm(seg_masks.iterrows(), desc="Segmentation", total=len(seg_masks)):
            linked = mask_row['linked_image']
            
            if pd.isna(linked):
                self.logger.warning(f"No linked_image for mask {mask_row['filename']}")
                self.stats['failed'] += 1
                continue
            
            img_path = os.path.join(self.config.BRISC_DIR, linked)
            
            if not os.path.exists(img_path):
                self.logger.warning(f"Linked image not found: {linked}")
                self.stats['failed'] += 1
                continue
            
            result = self.process_segmentation_pair(img_path, mask_row['full_path'], mask_row)
            if result:
                processed.append(result)
                self.stats['segmentation_processed'] += 1
            
            self.stats['total_processed'] += 1
        
        self.logger.info(f"\nProcessing complete:")
        self.logger.info(f"  Total processed: {self.stats['total_processed']}")
        self.logger.info(f"  Classification: {self.stats['classification_processed']}")
        self.logger.info(f"  Segmentation: {self.stats['segmentation_processed']}")
        self.logger.info(f"  Failed: {self.stats['failed']}")
        self.logger.info(f"  Tumor size warnings: {self.stats['tumor_size_warnings']}")
        
        return processed
    
    def save_processed(self, samples: List[Dict]):
        self.logger.info("Saving processed data")
        
        os.makedirs(self.config.PROCESSED_DIR, exist_ok=True)
        
        for sample in tqdm(samples, desc="Saving"):
            sample_id = sample['sample_id']
            output_path = os.path.join(self.config.PROCESSED_DIR, f'{sample_id}.npz')
            
            if sample['task'] == 'classification':
                np.savez_compressed(
                    output_path,
                    image=sample['image'].astype(np.uint8),
                    label=sample['label'],
                    view=sample['view'],
                    tumor_label=sample['tumor_label']
                )
            else:
                np.savez_compressed(
                    output_path,
                    image=sample['image'].astype(np.uint8),
                    mask=sample['mask'].astype(np.uint8),
                    label=sample['label'],
                    view=sample['view'],
                    tumor_label=sample['tumor_label']
                )
        
        self.logger.info(f"✓ Saved {len(samples)} files to {self.config.PROCESSED_DIR}")
    
    def report_tumor_statistics(self):
        self.logger.info("\n" + "="*70)
        self.logger.info("TUMOR BOUNDING BOX STATISTICS")
        self.logger.info("="*70)
        
        for tumor_type, sizes in self.tumor_bbox_stats.items():
            if sizes:
                sizes = np.array(sizes)
                self.logger.info(f"\n{tumor_type.upper()}:")
                self.logger.info(f"  Mean size: {sizes.mean(axis=0).astype(int)}")
                self.logger.info(f"  Min size:  {sizes.min(axis=0)}")
                self.logger.info(f"  Max size:  {sizes.max(axis=0)}")
                self.logger.info(f"  Count: {len(sizes)}")
        
        self.logger.info("\n" + "="*70)


if __name__ == "__main__":
    config = Config()
    logger = setup_logging(config.PROCESSED_DIR)
    
    logger.info("="*70)
    logger.info("BRISC PREPROCESSING PIPELINE")
    logger.info("="*70)
    logger.info(f"BRISC_DIR: {config.BRISC_DIR}")
    logger.info(f"PROCESSED_DIR: {config.PROCESSED_DIR}")
    logger.info(f"TARGET_SIZE: {config.TARGET_SIZE}")
    logger.info(f"MIN_TUMOR_SIZE: {config.MIN_TUMOR_SIZE}")
    logger.info("="*70 + "\n")
    
    preprocessor = BRISCPreprocessor(config, logger)
    samples = preprocessor.process_dataset()
    
    if samples:
        preprocessor.save_processed(samples)
        preprocessor.report_tumor_statistics()
        logger.info("\n✓ Preprocessing complete!")
    else:
        logger.error("No samples processed")