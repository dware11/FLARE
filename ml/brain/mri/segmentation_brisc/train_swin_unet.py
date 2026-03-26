"""
FLARE - MRI Brain Tumor Segmentation Training Script
Model: Swin-T + UNet Decoder (via segmentation_models_pytorch)
Dataset: BRISC 2025 Segmentation Task
Task: Binary segmentation (tumor vs background)

Research basis:
- Swin Transformer encoder captures long-range dependencies via shifted windows
- UNet decoder restores spatial resolution with skip connections
- This matches the Swin-HAFNet architecture reported in BRISC paper (80.6% mIoU)
- Pretrained Swin-T weights from ImageNet for transfer learning
- Same Combined Dice + BCE loss with pos_weight as Attention U-Net for fair comparison
- Separate output folder so Attention U-Net results are preserved
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
import segmentation_models_pytorch as smp
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
    # Separate output folder — does NOT overwrite Attention U-Net results
    OUTPUT_DIR    = '/scratch/bckk/flare/mri_brain/segmentation/outputs/swin_unet'

    # --- Model ---
    INPUT_SIZE    = 256        # Swin-T works well at 256x256
    IN_CHANNELS   = 1          # grayscale MRI
    OUT_CHANNELS  = 1          # binary mask

    # --- Training ---
    BATCH_SIZE    = 8          # Swin-T is larger than U-Net, needs more memory
    NUM_EPOCHS    = 60
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-4
    PATIENCE      = 15

    # --- Loss ---
    BCE_WEIGHT    = 0.5
    DICE_WEIGHT   = 0.5
    POS_WEIGHT    = 10.0       # penalize missing tumor pixels 10x

    # --- Data ---
    RANDOM_SEED   = 42
    NUM_WORKERS   = 4
    VAL_SPLIT     = 0.2


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
# DATASET — identical to Attention U-Net for fair comparison
# ==============================================================================
class BRISCSegmentationDataset(Dataset):
    def __init__(self, file_list, processed_dir, split='train',
                 input_size=256, augment=False):
        self.file_list     = file_list
        self.processed_dir = processed_dir
        self.input_size    = input_size
        self.augment       = augment

        logging.info(f"Loaded {len(file_list)} {split} segmentation samples")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.processed_dir, self.file_list[idx])
        data  = np.load(filepath, allow_pickle=True)
        image = data['image'].astype(np.float32)
        mask  = data['mask'].astype(np.float32)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)
        mask  = torch.tensor(mask).unsqueeze(0)   # (1, H, W)

        # Resize to INPUT_SIZE
        image = F.interpolate(image.unsqueeze(0), size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False).squeeze(0)
        mask  = F.interpolate(mask.unsqueeze(0), size=(self.input_size, self.input_size),
                              mode='nearest').squeeze(0)

        # Augmentation (training only)
        if self.augment and torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
            mask  = torch.flip(mask,  dims=[2])
        if self.augment and torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[1])
            mask  = torch.flip(mask,  dims=[1])
        if self.augment:
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0, 1)

        # Swin-T expects 3-channel input (pretrained on ImageNet RGB)
        # Copy grayscale to 3 channels — same 2.5D strategy as classification
        image = image.repeat(3, 1, 1)  # (3, H, W)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image, mask


# ==============================================================================
# MODEL — Swin-T encoder + UNet decoder from SMP
# ==============================================================================
def build_model(config):
    """
    Swin Transformer (tiny) encoder + UNet decoder.

    Swin-T encoder:
    - Hierarchical feature maps at 4 scales (like CNN encoder)
    - Shifted window self-attention captures long-range dependencies
    - Pretrained on ImageNet for strong feature initialization
    - Much better at capturing global context than CNN encoders

    UNet decoder:
    - Standard skip connections from each encoder scale
    - Bilinear upsampling + conv blocks
    - Outputs full-resolution segmentation mask

    This is the closest publicly available implementation to Swin-HAFNet.
    """
    model = smp.Unet(
        encoder_name='mit_b3',  # Swin-T backbone
        encoder_weights='imagenet',                    # pretrained weights
        in_channels=3,                                 # 3-channel (grayscale copied)
        classes=config.OUT_CHANNELS,                   # 1 = binary segmentation
        activation=None,                               # raw logits (sigmoid in loss)
    )

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Swin-UNet total params:     {total_params:,}")
    logging.info(f"Swin-UNet trainable params: {trainable_params:,}")

    return model


# ==============================================================================
# LOSS FUNCTIONS — identical to Attention U-Net for fair comparison
# ==============================================================================
class DiceLoss(nn.Module):
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
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=10.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.dice = DiceLoss()

    def forward(self, pred, target):
        self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        return (self.bce_weight  * self.bce(pred, target) +
                self.dice_weight * self.dice(pred, target))


# ==============================================================================
# METRICS
# ==============================================================================
def dice_score(pred, target, threshold=0.5, smooth=1.0):
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) /
            (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
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
        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'dice': f'{dice_score(outputs, masks):.4f}'})

    n = len(loader)
    return total_loss / n, total_dice / n


def validate(model, loader, criterion, device, epoch, split='Val'):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{split}]", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()
            total_dice += dice_score(outputs, masks)
            total_iou  += iou_score(outputs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ==============================================================================
# PLOTTING
# ==============================================================================
def save_prediction_samples(model, dataset, device, output_dir, n_samples=6):
    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            output = model(image.unsqueeze(0).to(device))
            pred   = (torch.sigmoid(output) > 0.5).float().cpu().squeeze()

            # Show only first channel for display (all 3 are identical)
            image_np = image[0].numpy()
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
    axes[0].set_title('Loss (Dice + BCE)')
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
    axes[2].set_title('IoU Score')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- File lists ---
    processed_path = Path(config.PROCESSED_DIR)
    all_train = sorted([f.name for f in processed_path.glob('segmentation_train_*.npz')])
    all_test  = sorted([f.name for f in processed_path.glob('segmentation_test_*.npz')])

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
    model = build_model(config).to(device)

    # --- Loss & Optimizer ---
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        pos_weight=config.POS_WEIGHT
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
    best_val_dice    = 0.0
    best_epoch       = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_dice': [],
        'val_loss':   [], 'val_dice':   [], 'val_iou': [],
        'lr': []
    }

    logger.info("=" * 60)
    logger.info("STARTING MiT-B3 + UNET TRAINING")
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

        if val_dice > best_val_dice:
            best_val_dice    = val_dice
            best_epoch       = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou':  val_iou,
            }, os.path.join(config.OUTPUT_DIR, 'best_model_swin.pth'))
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
        os.path.join(config.OUTPUT_DIR, 'best_model_swin.pth'),
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
    logger.info(f"\n  BRISC Benchmark comparison:")
    logger.info(f"  U-Net baseline:      75.7% mIoU")
    logger.info(f"  Attention U-Net:     79.0% mIoU  ← our previous model")
    logger.info(f"  Swin-UNet (ours):    {test_iou*100:.1f}% mIoU  ← this model")
    logger.info(f"  SaberNet/HAFNet:     80.6% mIoU  ← BRISC paper target")

    results = {
        'model': 'Swin-UNet',
        'best_epoch':    best_epoch,
        'best_val_dice': best_val_dice,
        'test_dice':     test_dice,
        'test_iou':      test_iou,
        'test_loss':     test_loss,
    }
    with open(os.path.join(config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_training_curves(history, config.OUTPUT_DIR)
    save_prediction_samples(model, test_dataset, device, config.OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("MiT-B3 UNET TRAINING COMPLETE")
    logger.info(f"Test Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}")
    logger.info(f"Outputs: {config.OUTPUT_DIR}")
    logger.info("=" * 60)

    return model, results


# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model, results = train(config)