# BraTS SEGMENTATION 

# Imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import gc



DATA_DIR = r"/Users/huda/Desktop/Flare_brain_mri/classification_data_2020/brats2020_processed"
SAVE_DIR = r"/Users/huda/Desktop/Flare_brain_mri/classification_data_2020/checkpoints"  

# TRAINING CONFIGURATION

BATCH_SIZE = 1          
NUM_EPOCHS = 30         
LR = 1e-4              
CHECKPOINT_EVERY = 5    
NUM_WORKERS =4         
STEPS_PER_EPOCH = 100

# SETUP

class Augmenter:
    """Simple augmentations for training."""
    def __call__(self, image, seg):
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            seg = np.flip(seg, axis=0).copy()
        if random.random() < 0.3:
            image = np.flip(image, axis=2).copy()
            seg = np.flip(seg, axis=1).copy()
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            mn, mx = image.min(), image.max()
            if mx > mn:
                image = ((image - mn) / (mx - mn + 1e-8)) ** gamma * (mx - mn) + mn
        return image, seg

class BraTSSegDataset(Dataset):
    """Dataset for segmentation."""
    def __init__(self, data_dir, file_list, val_seg_labels=None, augment=False):
        self.data_dir = data_dir
        self.file_list = file_list
        self.val_seg_labels = val_seg_labels
        self.augmenter = Augmenter() if augment else None
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        fname = self.file_list[idx]
        case_id = fname.replace('.npz', '')
        data = np.load(os.path.join(self.data_dir, fname))
        image = data['image'].astype(np.float32)
        
        if self.val_seg_labels and case_id in self.val_seg_labels:
            seg = self.val_seg_labels[case_id].astype(np.int64)
        else:
            seg = data['seg'].astype(np.int64)
        
        if self.augmenter:
            image, seg = self.augmenter(image, seg)
        
        return torch.from_numpy(image.copy()), torch.from_numpy(seg.copy()), case_id



def main():
    # Create directories
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    VAL_LABELS_PATH = os.path.join(DATA_DIR, 'val_labels.npz')
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_seg_model.pth')
    LATEST_CKPT_PATH = os.path.join(SAVE_DIR, 'latest_checkpoint.pth')
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Verify paths exist
    print("="*70)
    print("CHECKING PATHS")
    print("="*70)
    
    for name, path in [("DATA_DIR", DATA_DIR), ("TRAIN_DIR", TRAIN_DIR), 
                       ("VAL_DIR", VAL_DIR), ("VAL_LABELS", VAL_LABELS_PATH)]:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ NOT FOUND"
        print(f"  {name}: {path} [{status}]")
        if not exists:
            print(f"\n  ERROR: {path} does not exist!")
            print(f"  Please edit the paths at the top of this script.")
            return
    
    print(f"\n  SAVE_DIR: {SAVE_DIR} [✓ Created]")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n  Device: Apple Silicon GPU (MPS) [✓]")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  Device: CUDA GPU [{torch.cuda.get_device_name(0)}]")
    else:
        device = torch.device("cpu")
        print("\n  Device: CPU")
        print("  WARNING: No GPU found! Training will be slow.")
        
    # DATASET
    
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    # Get file lists
    TRAIN_FILES = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith('.npz')])
    VAL_FILES = sorted([f for f in os.listdir(VAL_DIR) if f.endswith('.npz')])
    val_seg_labels = dict(np.load(VAL_LABELS_PATH))
    
    train_dataset = BraTSSegDataset(TRAIN_DIR, TRAIN_FILES, augment=True)
    val_dataset = BraTSSegDataset(VAL_DIR, VAL_FILES, val_seg_labels=val_seg_labels, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    print(f"  Train cases: {len(train_dataset)}")
    print(f"  Val cases: {len(val_dataset)}")
    
    # Verify data shape
    img, seg, _ = train_dataset[0]
    print(f"  Image shape: {img.shape}")
    print(f"  Seg shape: {seg.shape}")
    
    # MODEL
    
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.01, inplace=True),
            )
        def forward(self, x):
            return self.block(x)
    
    class SegmentationUNet(nn.Module):
        def __init__(self, in_channels=4, out_channels=4):
            super().__init__()
            
            # Encoder
            self.enc1 = ConvBlock(in_channels, 32)
            self.pool1 = nn.MaxPool3d(2)
            self.enc2 = ConvBlock(32, 64)
            self.pool2 = nn.MaxPool3d(2)
            self.enc3 = ConvBlock(64, 128)
            self.pool3 = nn.MaxPool3d(2)
            self.enc4 = ConvBlock(128, 256)
            self.pool4 = nn.MaxPool3d(2)
            
            # Bottleneck
            self.bottleneck = ConvBlock(256, 512)
            
            # Decoder
            self.up4 = nn.ConvTranspose3d(512, 256, 2, stride=2)
            self.dec4 = ConvBlock(512, 256)
            self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
            self.dec3 = ConvBlock(256, 128)
            self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
            self.dec2 = ConvBlock(128, 64)
            self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
            self.dec1 = ConvBlock(64, 32)
            
            # Output
            self.seg_head = nn.Conv3d(32, out_channels, 1)
        
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
            bn = self.bottleneck(self.pool4(e4))
            
            d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            
            return self.seg_head(d1)
    
    model = SegmentationUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # LOSS & METRICS
    
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-5):
            super().__init__()
            self.smooth = smooth
        
        def forward(self, logits, target):
            probs = F.softmax(logits, dim=1)
            target_oh = F.one_hot(target, 4).permute(0, 4, 1, 2, 3).float()
            
            dice = 0.0
            for c in range(4):
                p, g = probs[:, c].flatten(), target_oh[:, c].flatten()
                dice += (2 * (p * g).sum() + self.smooth) / (p.sum() + g.sum() + self.smooth)
            return 1.0 - dice / 4
    
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.dice = DiceLoss()
            self.ce = nn.CrossEntropyLoss()
        
        def forward(self, logits, target):
            return 0.5 * self.dice(logits, target) + 0.5 * self.ce(logits, target)
    
    def compute_dice_scores(pred, target):
        smooth = 1e-5
        
        wt_p, wt_g = (pred >= 1).float(), (target >= 1).float()
        wt_dice = ((2 * (wt_p * wt_g).sum() + smooth) / (wt_p.sum() + wt_g.sum() + smooth)).item()
        
        tc_p = ((pred == 1) | (pred == 3)).float()
        tc_g = ((target == 1) | (target == 3)).float()
        tc_dice = ((2 * (tc_p * tc_g).sum() + smooth) / (tc_p.sum() + tc_g.sum() + smooth)).item()
        
        et_p, et_g = (pred == 3).float(), (target == 3).float()
        et_dice = ((2 * (et_p * et_g).sum() + smooth) / (et_p.sum() + et_g.sum() + smooth)).item()
        
        return {'WT': wt_dice, 'TC': tc_dice, 'ET': et_dice}
    
    criterion = CombinedLoss()
    
    # OPTIMIZER + RESUME
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    #scaler = GradScaler()
    
    start_epoch = 1
    best_dice = 0.0
    history = defaultdict(list)
  
    
    # Resume from checkpoint if exists
    if os.path.exists(LATEST_CKPT_PATH):
        print("\n  Found checkpoint, resuming...")
        ckpt = torch.load(LATEST_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt['best_dice']
        history = defaultdict(list, ckpt['history'])
        print(f"  Resumed from epoch {start_epoch - 1}")
    
    elif os.path.exists(BEST_MODEL_PATH):
        print("\n  No checkpoint, loading best model...")
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt['dice']['WT']
        if 'history' in ckpt:
            history = defaultdict(list, ckpt['history'])
        print(f"  Loaded from epoch {ckpt['epoch']}, Dice WT: {best_dice:.4f}")
    
    else:
        print("  Starting fresh")

    # TRAINING LOOP
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()
        
        # TRAIN
        model.train()
        train_loss = 0
        
        for batch_idx, (imgs, segs, _) in enumerate(train_loader):
            if batch_idx >= STEPS_PER_EPOCH:
                break
           
            imgs, segs = imgs.to(device), segs.to(device)
            
            optimizer.zero_grad()
            
            # Direct forward pass (un-indented)
            output = model(imgs)
            loss = criterion(output, segs)
            
            # Standard backward pass
            loss.backward()
            
            # Keep the gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Standard optimizer step
            optimizer.step()
            
            train_loss += loss.item()
            
            # Progress indicator
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch} - Batch {batch_idx+1}/{len(train_loader)}", end='\r')
        
        train_loss /= len(train_loader)
        
        # VALIDATE 
        model.eval()
        val_loss = 0
        val_dice = {'WT': [], 'TC': [], 'ET': []}
        
        with torch.no_grad():
            for imgs, segs, _ in val_loader:
                imgs, segs = imgs.to(device), segs.to(device)
                
                output = model(imgs)
                loss = criterion(output, segs)
                
                val_loss += loss.item()
                
                pred = output.argmax(1)
                dice = compute_dice_scores(pred, segs)
                for k in dice:
                    val_dice[k].append(dice[k])
        
        val_loss /= len(val_loader)
        val_dice = {k: np.mean(v) for k, v in val_dice.items()}
        
        scheduler.step()
        
        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['dice_wt'].append(val_dice['WT'])
        history['dice_tc'].append(val_dice['TC'])
        history['dice_et'].append(val_dice['ET'])
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} ({elapsed:.0f}s) | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Dice WT:{val_dice['WT']:.3f} TC:{val_dice['TC']:.3f} ET:{val_dice['ET']:.3f}")
        
        # Save best
        if val_dice['WT'] > best_dice:
            best_dice = val_dice['WT']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dice': val_dice,
                'history': dict(history)
            }, BEST_MODEL_PATH)
            print(f"  └─ ✓ Best model saved (WT: {best_dice:.4f})")
        
        # Checkpoint
        if epoch % CHECKPOINT_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'history': dict(history)
            }, LATEST_CKPT_PATH)
            print(f"  └─ ✓ Checkpoint saved")
        
        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE! Best WT Dice: {best_dice:.4f}")
    print("="*70)
    
    # SAVE TRAINING CURVES
    
    print("\nSaving training curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['dice_wt'], label='Whole Tumor (WT)', linewidth=2)
    axes[1].plot(history['dice_tc'], label='Tumor Core (TC)', linewidth=2)
    axes[1].plot(history['dice_et'], label='Enhancing (ET)', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    curves_path = os.path.join(SAVE_DIR, 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    print(f"  Saved: {curves_path}")
    
    # VISUALIZE PREDICTIONS
    
    print("\nGenerating sample prediction...")
    
    # Load best model
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Get a sample
    img, seg_gt, case_id = next(iter(val_loader))
    
    with torch.no_grad():
        output = model(img.to(device))
        seg_pred = output.argmax(1)[0].cpu().numpy()
    
    seg_gt = seg_gt[0].numpy()
    img = img[0].numpy()
    
    # Find slice with most tumor
    tumor_per_slice = (seg_gt > 0).sum(axis=(0, 1))
    max_slice = tumor_per_slice.argmax()
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img[1, :, :, max_slice].T, cmap='gray', origin='lower')
    axes[0].set_title('T1c Input')
    axes[0].axis('off')
    
    axes[1].imshow(seg_gt[:, :, max_slice].T, cmap='jet', origin='lower', vmin=0, vmax=3)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(seg_pred[:, :, max_slice].T, cmap='jet', origin='lower', vmin=0, vmax=3)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    axes[3].imshow(img[1, :, :, max_slice].T, cmap='gray', origin='lower')
    axes[3].imshow(seg_pred[:, :, max_slice].T, cmap='jet', alpha=0.5, origin='lower', vmin=0, vmax=3)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.suptitle(f'{case_id[0]} | Slice {max_slice}', fontsize=14)
    plt.tight_layout()
    pred_path = os.path.join(SAVE_DIR, 'prediction_example.png')
    plt.savefig(pred_path, dpi=150)
    print(f"  Saved: {pred_path}")
    
    # SUMMARY
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Best Model:     {BEST_MODEL_PATH}")
    print(f"  Dice WT:        {ckpt['dice']['WT']:.4f}")
    print(f"  Dice TC:        {ckpt['dice']['TC']:.4f}")
    print(f"  Dice ET:        {ckpt['dice']['ET']:.4f}")
    print(f"  Training Curves: {curves_path}")
    print(f"  Sample Prediction: {pred_path}")
    print("="*70)


if __name__ == "__main__":
    main()
