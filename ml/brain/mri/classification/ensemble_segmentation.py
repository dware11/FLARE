"""
FLARE - Ensemble Segmentation Script
Models: Attention U-Net + MiT-B3 UNet (averaged predictions)
Dataset: BRISC 2025 Segmentation Task

Method: Soft voting ensemble
- Load both trained models
- Average their sigmoid probability maps
- Apply threshold of 0.5 to get final binary mask
- This cancels out each model's individual errors
- Expected improvement: +2-4% mIoU over best single model
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import logging
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
# ==============================================================================
class Config:
    PROCESSED_DIR  = '/scratch/bckk/flare/mri_brain/classification/processed'
    MODEL1_PATH    = '/scratch/bckk/flare/mri_brain/segmentation/outputs/training/best_model_seg.pth'
    MODEL2_PATH    = '/scratch/bckk/flare/mri_brain/segmentation/outputs/swin_unet/best_model_swin.pth'
    OUTPUT_DIR     = '/scratch/bckk/flare/mri_brain/segmentation/outputs/ensemble'

    INPUT_SIZE     = 256
    BATCH_SIZE     = 8
    NUM_WORKERS    = 4
    RANDOM_SEED    = 42
    VAL_SPLIT      = 0.2

    # Ensemble weights (equal by default, can tune)
    WEIGHT_MODEL1  = 0.5   # Attention U-Net
    WEIGHT_MODEL2  = 0.5   # MiT-B3 UNet
    THRESHOLD      = 0.5   # final binary threshold


# LOGGING
# ==============================================================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'ensemble.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# DATASET
# ==============================================================================
class BRISCSegmentationDataset(Dataset):
    """
    Same dataset as individual models.
    Returns BOTH grayscale (for Attention U-Net) and 3-channel (for MiT-B3).
    """
    def __init__(self, file_list, processed_dir, input_size=256):
        self.file_list     = file_list
        self.processed_dir = processed_dir
        self.input_size    = input_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.processed_dir, self.file_list[idx])
        data  = np.load(filepath, allow_pickle=True)
        image = data['image'].astype(np.float32)
        mask  = data['mask'].astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)
        mask  = torch.tensor(mask).unsqueeze(0)   # (1, H, W)

        # Resize
        image = F.interpolate(image.unsqueeze(0), size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False).squeeze(0)
        mask  = F.interpolate(mask.unsqueeze(0), size=(self.input_size, self.input_size),
                              mode='nearest').squeeze(0)

        # Normalize for Attention U-Net (1-channel)
        mean1 = torch.tensor([0.485])
        std1  = torch.tensor([0.229])
        image_1ch = (image - mean1.view(1, 1, 1)) / std1.view(1, 1, 1)

        # Normalize for MiT-B3 (3-channel)
        image_3ch = image.repeat(3, 1, 1)
        mean3 = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std3  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_3ch = (image_3ch - mean3) / std3

        return image_1ch, image_3ch, mask


# ATTENTION U-NET MODEL 
# ==============================================================================
class ConvBlock(nn.Module):
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
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample  = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.attention = AttentionGate(F_g=in_ch // 2, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv      = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = self.attention(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.enc1      = ConvBlock(in_channels, features[0])
        self.enc2      = ConvBlock(features[0], features[1])
        self.enc3      = ConvBlock(features[1], features[2])
        self.enc4      = ConvBlock(features[2], features[3])
        self.pool      = nn.MaxPool2d(2, 2)
        self.dropout   = nn.Dropout2d(0.3)
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        self.up4       = UpBlock(features[3] * 2, features[3], features[3])
        self.up3       = UpBlock(features[3],     features[2], features[2])
        self.up2       = UpBlock(features[2],     features[1], features[1])
        self.up1       = UpBlock(features[1],     features[0], features[0])
        self.output    = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.dropout(self.pool(e3)))
        b  = self.bottleneck(self.dropout(self.pool(e4)))
        d4 = self.up4(b,  e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.output(d1)


# METRICS
# ==============================================================================
def dice_score(pred, target, threshold=0.5, smooth=1.0):
    pred   = (pred > threshold).float()
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) /
            (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    pred   = (pred > threshold).float()
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


# ENSEMBLE EVALUATION
# ==============================================================================
def evaluate_ensemble(model1, model2, loader, device, config):
    model1.eval()
    model2.eval()

    total_dice_m1  = 0.0
    total_dice_m2  = 0.0
    total_dice_ens = 0.0
    total_iou_m1   = 0.0
    total_iou_m2   = 0.0
    total_iou_ens  = 0.0

    all_images = []
    all_masks  = []
    all_preds  = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Ensemble Evaluation")
        for image_1ch, image_3ch, masks in pbar:
            image_1ch = image_1ch.to(device)
            image_3ch = image_3ch.to(device)
            masks     = masks.to(device)

            # Model 1 predictions (Attention U-Net)
            out1  = torch.sigmoid(model1(image_1ch))

            # Model 2 predictions (MiT-B3)
            out2  = torch.sigmoid(model2(image_3ch))

            # Weighted ensemble
            ensemble = (config.WEIGHT_MODEL1 * out1 +
                        config.WEIGHT_MODEL2 * out2)

            # Metrics
            total_dice_m1  += dice_score(out1,      masks, config.THRESHOLD)
            total_dice_m2  += dice_score(out2,      masks, config.THRESHOLD)
            total_dice_ens += dice_score(ensemble,  masks, config.THRESHOLD)
            total_iou_m1   += iou_score(out1,       masks, config.THRESHOLD)
            total_iou_m2   += iou_score(out2,       masks, config.THRESHOLD)
            total_iou_ens  += iou_score(ensemble,   masks, config.THRESHOLD)

            # Save samples for visualization
            if len(all_images) < 6:
                all_images.append(image_1ch[0].cpu())
                all_masks.append(masks[0].cpu())
                all_preds.append(ensemble[0].cpu())

    n = len(loader)
    return {
        'attn_unet_dice': total_dice_m1  / n,
        'mit_b3_dice':    total_dice_m2  / n,
        'ensemble_dice':  total_dice_ens / n,
        'attn_unet_iou':  total_iou_m1   / n,
        'mit_b3_iou':     total_iou_m2   / n,
        'ensemble_iou':   total_iou_ens  / n,
    }, all_images, all_masks, all_preds


# VISUALIZATION
# ==============================================================================
def save_comparison_plot(images, masks, preds, output_dir, n=6):
    """Save side-by-side comparison: MRI | Ground Truth | Ensemble Prediction."""
    fig, axes = plt.subplots(min(n, len(images)), 3, figsize=(12, min(n, len(images)) * 4))

    for i in range(min(n, len(images))):
        image_np = images[i].squeeze().numpy()
        mask_np  = masks[i].squeeze().numpy()
        pred_np  = (preds[i].squeeze().numpy() > 0.5).astype(float)

        dice = dice_score(
            preds[i].unsqueeze(0),
            masks[i].unsqueeze(0),
            threshold=0.5
        )

        axes[i, 0].imshow(image_np, cmap='gray')
        axes[i, 0].set_title('MRI Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_np, cmap='hot')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_np, cmap='hot')
        axes[i, 2].set_title(f'Ensemble Prediction\n(Dice: {dice:.3f})')
        axes[i, 2].axis('off')

    plt.suptitle('Ensemble: Attention U-Net + MiT-B3 UNet', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_predictions.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    logging.info("Saved ensemble_predictions.png")


def save_model_comparison_bar(results, output_dir):
    """Bar chart comparing all 3 models."""
    models = ['Attention U-Net\n(79.0%)', 'MiT-B3 UNet\n(80.6%)', 'Ensemble']
    ious   = [
        results['attn_unet_iou'] * 100,
        results['mit_b3_iou']    * 100,
        results['ensemble_iou']  * 100,
    ]
    dices  = [
        results['attn_unet_dice'] * 100,
        results['mit_b3_dice']    * 100,
        results['ensemble_dice']  * 100,
    ]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ious,  width, label='IoU (%)',  color='steelblue')
    bars2 = ax.bar(x + width/2, dices, width, label='Dice (%)', color='coral')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Segmentation Model Comparison — BRISC 2025')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(70, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Add SaberNet reference line
    ax.axhline(y=80.6, color='green', linestyle='--', alpha=0.7, label='SaberNet (80.6%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    logging.info("Saved model_comparison.png")


# MAIN
# ==============================================================================
def main():
    config = Config()
    logger = setup_logging(config.OUTPUT_DIR)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Load models ---
    logger.info("Loading Attention U-Net...")
    model1 = AttentionUNet(in_channels=1, out_channels=1).to(device)
    ckpt1  = torch.load(config.MODEL1_PATH, map_location=device, weights_only=False)
    model1.load_state_dict(ckpt1['model_state_dict'])
    model1.eval()
    logger.info(f"  Attention U-Net loaded (Val Dice: {ckpt1['val_dice']:.4f})")

    logger.info("Loading MiT-B3 UNet...")
    model2 = smp.Unet('mit_b3', encoder_weights=None, in_channels=3, classes=1).to(device)
    ckpt2  = torch.load(config.MODEL2_PATH, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt2['model_state_dict'])
    model2.eval()
    logger.info(f"  MiT-B3 loaded (Val Dice: {ckpt2['val_dice']:.4f})")

    # --- Dataset ---
    processed_path = Path(config.PROCESSED_DIR)
    all_test = sorted([f.name for f in processed_path.glob('segmentation_test_*.npz')])
    logger.info(f"Test samples: {len(all_test)}")

    test_dataset = BRISCSegmentationDataset(
        all_test, config.PROCESSED_DIR, input_size=config.INPUT_SIZE
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # --- Ensemble evaluation ---
    logger.info("=" * 60)
    logger.info("RUNNING ENSEMBLE EVALUATION")
    logger.info("=" * 60)

    results, images, masks, preds = evaluate_ensemble(
        model1, model2, test_loader, device, config
    )

    # --- Results ---
    logger.info(f"\nRESULTS COMPARISON:")
    logger.info(f"{'Model':<25} {'IoU':>8} {'Dice':>8}")
    logger.info(f"{'-'*45}")
    logger.info(f"{'Attention U-Net':<25} {results['attn_unet_iou']*100:>7.2f}% {results['attn_unet_dice']*100:>7.2f}%")
    logger.info(f"{'MiT-B3 UNet':<25} {results['mit_b3_iou']*100:>7.2f}% {results['mit_b3_dice']*100:>7.2f}%")
    logger.info(f"{'Ensemble':<25} {results['ensemble_iou']*100:>7.2f}% {results['ensemble_dice']*100:>7.2f}%")
    logger.info(f"{'-'*45}")
    logger.info(f"{'SaberNet (paper)':<25} {'80.6%':>8} {'N/A':>8}")

    improvement = (results['ensemble_iou'] - results['mit_b3_iou']) * 100
    logger.info(f"\nEnsemble improvement over best single model: {improvement:+.2f}% mIoU")

    # --- Save results ---
    with open(os.path.join(config.OUTPUT_DIR, 'ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    save_comparison_plot(images, masks, preds, config.OUTPUT_DIR)
    save_model_comparison_bar(results, config.OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("ENSEMBLE COMPLETE")
    logger.info(f"Final IoU:  {results['ensemble_iou']*100:.2f}%")
    logger.info(f"Final Dice: {results['ensemble_dice']*100:.2f}%")
    logger.info(f"Outputs: {config.OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()