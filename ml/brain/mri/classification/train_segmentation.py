"""
FLARE - MRI Brain Tumor Segmentation Training Script
Model: Attention U-Net
Dataset: BRISC 2025 Segmentation Task
Task: Binary segmentation (tumor vs background)
Author: FLARE Team - Huda

Research basis:
- Attention U-Net adds attention gates to standard U-Net skip connections
- Attention gates suppress irrelevant background regions automatically
- Combined Dice + BCE loss handles extreme class imbalance (1% tumor pixels)
- Dice score is the standard metric for medical image segmentation
- 80/20 train/val split from training data, test set untouched
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # --- Paths ---
    PROCESSED_DIR = '/scratch/bckk/flare/mri_brain/classification/processed'
    OUTPUT_DIR    = '/scratch/bckk/flare/mri_brain/segmentation/outputs/training'

    # --- Model ---
    INPUT_SIZE    = 256        # resize to 256x256 for memory efficiency on A40
    IN_CHANNELS   = 1          # grayscale
    OUT_CHANNELS  = 1          # binary mask (tumor vs background)
    FEATURES      = [64, 128, 256, 512]  # encoder feature sizes

    # --- Training ---
    BATCH_SIZE    = 16
    NUM_EPOCHS    = 60
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-5
    PATIENCE      = 15         # early stopping patience

    # --- Loss weights ---
    # Combined Dice + BCE loss handles class imbalance
    # BCE_WEIGHT=0.5 and DICE_WEIGHT=0.5 is standard for medical segmentation
    BCE_WEIGHT    = 0.5
    DICE_WEIGHT   = 0.5

    # --- Data ---
    RANDOM_SEED   = 42
    NUM_WORKERS   = 4
    VAL_SPLIT     = 0.2        # 20% of train -> val

    # --- Tumor types ---
    TUMOR_TYPES   = ['glioma', 'meningioma', 'pituitary']


# ==============================================================================
# LOGGING
# ==============================================================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==============================================================================
# DATASET
# ==============================================================================
class BRISCSegmentationDataset(Dataset):
    """
    Loads preprocessed .npz segmentation files.
    Each file contains:
        - 'image': grayscale MRI slice (H, W) at 512x512
        - 'mask':  binary tumor mask (H, W) at 512x512, values 0 or 1
        - 'tumor_label': string tumor type
        - 'view': axial/coronal/sagittal

    Resize to INPUT_SIZE for memory efficiency.
    Apply augmentation during training for generalization.
    """

    def __init__(self, file_list, processed_dir, split='train',
                 input_size=256, augment=False):
        self.file_list    = file_list
        self.processed_dir = processed_dir
        self.split        = split
        self.input_size   = input_size
        self.augment      = augment

        logging.info(f"Loaded {len(file_list)} {split} segmentation samples")

        # Log tumor type distribution
        from collections import Counter
        tumor_counts = Counter()
        for f in file_list[:100]:  # sample first 100 for speed
            data = np.load(os.path.join(processed_dir, f), allow_pickle=True)
            tumor_counts[str(data['tumor_label'])] += 1
        logging.info(f"  Tumor distribution (sample): {dict(tumor_counts)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.processed_dir, self.file_list[idx])
        data = np.load(filepath, allow_pickle=True)

        image = data['image'].astype(np.float32)
        mask  = data['mask'].astype(np.float32)

        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Resize both image and mask to INPUT_SIZE
        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)
        mask  = torch.tensor(mask).unsqueeze(0)   # (1, H, W)

        image = F.interpolate(image.unsqueeze(0), size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False).squeeze(0)
        mask  = F.interpolate(mask.unsqueeze(0), size=(self.input_size, self.input_size),
                              mode='nearest').squeeze(0)

        # Augmentation (training only)
        # Conservative augmentations that preserve anatomical validity
        if self.augment and torch.rand(1) > 0.5:
            # Random horizontal flip
            image = torch.flip(image, dims=[2])
            mask  = torch.flip(mask,  dims=[2])

        if self.augment and torch.rand(1) > 0.5:
            # Random vertical flip (valid for axial MRI)
            image = torch.flip(image, dims=[1])
            mask  = torch.flip(mask,  dims=[1])

        if self.augment:
            # Gaussian noise for robustness
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0, 1)

        # Normalize with ImageNet-style stats (for consistency)
        mean = torch.tensor([0.485])
        std  = torch.tensor([0.229])
        image = (image - mean.view(1, 1, 1)) / std.view(1, 1, 1)

        return image, mask


# ==============================================================================
# ATTENTION U-NET MODEL
# ==============================================================================

class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU — basic U-Net building block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate from Oktay et al. (2018) - Attention U-Net.

    Takes two inputs:
        g = gating signal from decoder (coarser, semantic)
        x = skip connection from encoder (finer, spatial)

    Produces attention coefficients that highlight relevant tumor regions
    and suppress irrelevant background — critical for small tumor areas.

    This is the key difference from standard U-Net.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Upsample g to match x spatial dimensions
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Add and apply ReLU
        psi = self.relu(g1 + x1)
        # Compute attention map
        psi = self.psi(psi)
        # Apply attention to skip connection
        return x * psi


class UpBlock(nn.Module):
    """Decoder block: upsample + attention gate + conv block."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.attention = AttentionGate(
            F_g=in_ch // 2,
            F_l=skip_ch,
            F_int=skip_ch // 2
        )
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch (padding if needed)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Apply attention gate to skip connection
        skip = self.attention(g=x, x=skip)
        # Concatenate and convolve
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary tumor segmentation.

    Architecture:
    - Encoder: 4 ConvBlocks with MaxPool downsampling
    - Bottleneck: ConvBlock at lowest resolution
    - Decoder: 4 UpBlocks with Attention Gates on skip connections
    - Output: 1-channel sigmoid map (tumor probability per pixel)

    Key improvement over standard U-Net:
    Attention gates learn to focus on tumor regions automatically,
    suppressing the 98.9% background pixels during feature fusion.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.3)

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        # Decoder with Attention Gates
        self.up4 = UpBlock(features[3] * 2, features[3], features[3])
        self.up3 = UpBlock(features[3],     features[2], features[2])
        self.up2 = UpBlock(features[2],     features[1], features[1])
        self.up1 = UpBlock(features[1],     features[0], features[0])

        # Output layer
        self.output = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        # Encoder path (save for skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.dropout(self.pool(e3)))

        # Bottleneck
        b = self.bottleneck(self.dropout(self.pool(e4)))

        # Decoder path with attention gates
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.output(d1)


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Directly optimizes the Dice coefficient (overlap between prediction and mask).
    Critical for class imbalance — ignores the 98.9% background pixels.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss with pos_weight.
    
    pos_weight=10.0 means:
    - Missing a tumor pixel is penalized 10x MORE than missing a background pixel
    - Critical because tumor pixels = only 1% of all pixels
    - Without this, model learns to predict all background and gets 99% "accuracy"
    - Combined with Dice loss = best of both worlds for pixel-level tumor finding
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=10.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        # pos_weight: penalize missing tumor pixels 10x more than background
        self.bce  = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # Move pos_weight to same device as pred
        self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        return (self.bce_weight  * self.bce(pred, target) +
                self.dice_weight * self.dice(pred, target))


# ==============================================================================
# METRICS
# ==============================================================================

def dice_score(pred, target, threshold=0.5, smooth=1.0):
    """Compute Dice coefficient — standard metric for segmentation."""
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) /
            (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """Compute IoU (Intersection over Union) — used in BRISC benchmark."""
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(outputs, masks)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice_score(outputs, masks):.4f}'
        })

    n = len(loader)
    return total_loss / n, total_dice / n


def validate(model, loader, criterion, device, epoch, split='Val'):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    # Per tumor type tracking
    tumor_dice = {'glioma': [], 'meningioma': [], 'pituitary': []}

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{split}]", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
            total_iou  += iou_score(outputs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def save_prediction_samples(model, dataset, device, output_dir, n_samples=6):
    """Save sample predictions vs ground truth for visual inspection."""
    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            output = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(output) > 0.5).float().cpu().squeeze()

            image_np = image.squeeze().numpy()
            mask_np  = mask.squeeze().numpy()
            pred_np  = pred.numpy()

            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title('MRI Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_np, cmap='hot')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_np, cmap='hot')
            dice = dice_score(output.cpu(), mask.unsqueeze(0))
            axes[i, 2].set_title(f'Prediction (Dice: {dice:.3f})')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_samples.png'), dpi=150)
    plt.close()
    logging.info("Saved prediction_samples.png")


def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'r-', label='Val')
    axes[0].set_title('Combined Loss (Dice + BCE)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_dice'],   'r-', label='Val')
    axes[1].set_title('Dice Score')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history['val_iou'], 'g-', label='Val IoU')
    axes[2].set_title('IoU Score (mIoU)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    logging.info("Saved training_curves.png")


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train(config):
    logger = setup_logging(config.OUTPUT_DIR)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- File lists ---
    processed_path = Path(config.PROCESSED_DIR)
    all_train = sorted([f.name for f in processed_path.glob('segmentation_train_*.npz')])
    all_test  = sorted([f.name for f in processed_path.glob('segmentation_test_*.npz')])

    logger.info(f"Total train files: {len(all_train)} | Test files: {len(all_test)}")

    # Split train 80/20 into train/val
    train_files, val_files = train_test_split(
        all_train, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED
    )

    logger.info(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(all_test)}")

    # --- Datasets ---
    train_dataset = BRISCSegmentationDataset(
        train_files, config.PROCESSED_DIR,
        split='train', input_size=config.INPUT_SIZE, augment=True
    )
    val_dataset = BRISCSegmentationDataset(
        val_files, config.PROCESSED_DIR,
        split='val', input_size=config.INPUT_SIZE, augment=False
    )
    test_dataset = BRISCSegmentationDataset(
        all_test, config.PROCESSED_DIR,
        split='test', input_size=config.INPUT_SIZE, augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=config.NUM_WORKERS,
                               pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=False, num_workers=config.NUM_WORKERS,
                               pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=False, num_workers=config.NUM_WORKERS,
                               pin_memory=True)

    # --- Model ---
    model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        features=config.FEATURES
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Attention U-Net params: {total_params:,}")

    # --- Loss & Optimizer ---
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

    # --- Training state ---
    best_val_dice = 0.0
    best_epoch    = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_dice': [],
        'val_loss':   [], 'val_dice':   [], 'val_iou': [],
        'lr': []
    }

    logger.info("=" * 60)
    logger.info("STARTING ATTENTION U-NET TRAINING")
    logger.info("=" * 60)

    for epoch in range(1, config.NUM_EPOCHS + 1):

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, epoch, split='Val'
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)

        logger.info(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch    = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou':  val_iou,
                'config':   config.__dict__
            }, os.path.join(config.OUTPUT_DIR, 'best_model_seg.pth'))
            logger.info(f"  ✓ New best model saved (Dice: {best_val_dice:.4f}, IoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # --- Final Test Evaluation ---
    logger.info("=" * 60)
    logger.info("LOADING BEST MODEL FOR TEST EVALUATION")
    logger.info("=" * 60)

    checkpoint = torch.load(
        os.path.join(config.OUTPUT_DIR, 'best_model_seg.pth'),
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Best model from epoch {best_epoch} (Val Dice: {best_val_dice:.4f})")

    test_loss, test_dice, test_iou = validate(
        model, test_loader, criterion, device, epoch=0, split='Test'
    )

    logger.info(f"\nTEST RESULTS:")
    logger.info(f"  Dice Score: {test_dice:.4f} ({test_dice*100:.2f}%)")
    logger.info(f"  IoU Score:  {test_iou:.4f}  ({test_iou*100:.2f}%)")
    logger.info(f"  Test Loss:  {test_loss:.4f}")
    logger.info(f"\n  BRISC Benchmark comparison:")
    logger.info(f"  U-Net baseline:    75.7% mIoU")
    logger.info(f"  Our Attn U-Net:    {test_iou*100:.1f}% mIoU")
    logger.info(f"  SaberNet target:   80.6% mIoU")

    # --- Save results ---
    results = {
        'best_epoch':  best_epoch,
        'best_val_dice': best_val_dice,
        'test_dice':   test_dice,
        'test_iou':    test_iou,
        'test_loss':   test_loss,
    }
    with open(os.path.join(config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    plot_training_curves(history, config.OUTPUT_DIR)
    save_prediction_samples(model, test_dataset, device, config.OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("SEGMENTATION TRAINING COMPLETE")
    logger.info(f"Best Val Dice: {best_val_dice:.4f}")
    logger.info(f"Test Dice:     {test_dice:.4f}")
    logger.info(f"Test IoU:      {test_iou:.4f}")
    logger.info(f"Outputs saved to: {config.OUTPUT_DIR}")
    logger.info("=" * 60)

    return model, results


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model, results = train(config)