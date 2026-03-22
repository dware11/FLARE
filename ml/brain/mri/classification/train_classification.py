"""
FLARE - MRI Brain Tumor Classification Training Script
Model: EfficientNetB0 (Transfer Learning)
Dataset: BRISC 2025
Classes: glioma, meningioma, pituitary, no_tumor

Research basis:
- EfficientNetB0 achieves 99.2% weighted F1 on BRISC (Fateh et al., 2026)
- 2.5D strategy: grayscale copied to 3 channels for ImageNet transfer learning
- 5-fold cross-validation used in BRISC benchmark
- Cosine annealing LR scheduler for stable convergence
- ReduceLROnPlateau as fallback per medical imaging best practices
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
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
    OUTPUT_DIR    = '/scratch/bckk/flare/mri_brain/classification/outputs/training'

    # --- Model ---
    NUM_CLASSES   = 4          # glioma, meningioma, pituitary, no_tumor
    INPUT_SIZE    = 224        # EfficientNetB0 expects 224x224
    # Note: preprocessed at 512x512, we crop to 224 at load time (tumor-aware)

    # --- Training hyperparameters (based on BRISC benchmark) ---
    BATCH_SIZE    = 32
    NUM_EPOCHS    = 50         # early stopping will kick in before this
    LEARNING_RATE = 1e-4       # initial LR for fine-tuning
    WEIGHT_DECAY  = 1e-4       # L2 regularization
    DROPOUT_RATE  = 0.3        # dropout before final classifier
    PATIENCE      = 10         # early stopping patience

    # --- Phase 1: Freeze backbone, train head only ---
    FREEZE_EPOCHS = 5          # train only classifier head first

    # --- LR Scheduler ---
    # Cosine annealing: smoothly decays LR, proven best for EfficientNet
    LR_MIN        = 1e-6       # minimum LR floor

    # --- Data ---
    RANDOM_SEED   = 42
    NUM_WORKERS   = 4

    # --- Class labels ---
    CLASS_NAMES   = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    LABEL_MAP     = {'glioma': 0, 'meningioma': 1, 'no_tumor': 2, 'pituitary': 3}


# ==============================================================================
# LOGGING SETUP
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
class BRISCClassificationDataset(Dataset):
    """
    Loads preprocessed .npz files for classification.
    
    Each .npz file contains:
        - 'image': grayscale image array (H, W) at 512x512
        - 'label': integer class label
        - 'tumor_label': string label name
    
    2.5D strategy: grayscale channel is copied to 3 channels (R=G=B)
    so we can use ImageNet-pretrained EfficientNetB0 weights.
    Crop from 512 -> 224 is done here at load time (center crop).
    """

    def __init__(self, processed_dir, split='train', transform=None, config=None):
        self.config = config or Config()
        self.transform = transform
        self.samples = []

        processed_path = Path(processed_dir)

        # Load all classification files for this split
        pattern = f'classification_{split}_*.npz'
        files = sorted(processed_path.glob(pattern))

        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found matching {processed_path / pattern}\n"
                f"Make sure preprocessing completed successfully."
            )

        for f in files:
            data = np.load(f, allow_pickle=True)
            label = int(data['label'])
            self.samples.append((str(f), label))

        logging.info(f"Loaded {len(self.samples)} {split} samples")

        # Log class distribution
        from collections import Counter
        label_counts = Counter([s[1] for s in self.samples])
        for label_idx, count in sorted(label_counts.items()):
            name = self.config.CLASS_NAMES[label_idx]
            logging.info(f"  {name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        data = np.load(filepath, allow_pickle=True)
        image = data['image'].astype(np.float32)

        # Normalize to [0, 255] range for EfficientNetB0
        # (EfficientNet handles its own internal normalization)
        if image.max() <= 1.0:
            image = image * 255.0

        # Convert grayscale to 3-channel (2.5D strategy)
        # Copy same values to R, G, B — matches ImageNet pretrained format
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=0)  # (3, H, W)
        elif image.ndim == 3 and image.shape[0] == 1:
            image = np.concatenate([image, image, image], axis=0)
        elif image.ndim == 3 and image.shape[2] in [1, 3]:
            # HWC format -> CHW
            if image.shape[2] == 1:
                image = image[:, :, 0]
                image = np.stack([image, image, image], axis=0)
            else:
                image = image.transpose(2, 0, 1)

        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================================================================
# DATA TRANSFORMS
# ==============================================================================
def get_transforms(config, split='train'):
    """
    Training: augmentation to improve generalization
    Val/Test: only resize + normalize (no augmentation)
    
    Augmentations are conservative for medical imaging:
    - No color jitter (grayscale data)
    - Mild rotation (MRI orientation matters)
    - Horizontal flip only (not vertical — anatomically incorrect)
    """
    input_size = config.INPUT_SIZE  # 224

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet mean
                std=[0.229, 0.224, 0.225]      # ImageNet std
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ==============================================================================
# MODEL
# ==============================================================================
def build_model(config, freeze_backbone=True):
    """
    EfficientNetB0 with ImageNet pretrained weights.
    
    Two-phase training:
    Phase 1 (freeze_backbone=True):  Only train the classifier head
    Phase 2 (freeze_backbone=False): Unfreeze all layers, fine-tune end-to-end
    
    This prevents destroying pretrained features early in training.
    """
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Freeze all backbone layers in phase 1
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        logging.info("Phase 1: Backbone FROZEN — training classifier head only")
    else:
        for param in model.parameters():
            param.requires_grad = True
        logging.info("Phase 2: Full model UNFROZEN — fine-tuning end-to-end")

    # Replace classifier head
    # Original: Linear(1280, 1000) for ImageNet
    # Ours: Linear(1280, 4) for our 4 tumor classes
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=config.DROPOUT_RATE / 2),
        nn.Linear(256, config.NUM_CLASSES)
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model


def unfreeze_model(model, config):
    """Unfreeze all layers for phase 2 fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    logging.info("Phase 2: All layers unfrozen for fine-tuning")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable params now: {trainable_params:,}")
    return model


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device, epoch, split='Val'):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{split}]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1, all_preds, all_labels


# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history['val_f1'], 'g-', label='Val F1')
    axes[2].set_title('Weighted F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    logging.info("Saved training_curves.png")


def plot_confusion_matrix(labels, preds, class_names, output_dir, title='confusion_matrix'):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}.png'), dpi=150)
    plt.close()
    logging.info(f"Saved {title}.png")


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

    # --- Datasets ---
    logger.info("Loading datasets...")

    full_train = BRISCClassificationDataset(
        config.PROCESSED_DIR, split='train',
        transform=get_transforms(config, 'train'), config=config
    )
    # Split train 80/20 into train and val
    train_indices, val_indices = train_test_split(
        range(len(full_train)),
        test_size=0.2,
        random_state=config.RANDOM_SEED
    )
    train_dataset = torch.utils.data.Subset(full_train, train_indices)

    val_full = BRISCClassificationDataset(
        config.PROCESSED_DIR, split='train',
        transform=get_transforms(config, 'val'), config=config
    )
    val_dataset = torch.utils.data.Subset(val_full, val_indices)

    test_dataset = BRISCClassificationDataset(
        config.PROCESSED_DIR, split='test',
        transform=get_transforms(config, 'test'), config=config
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

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    # --- Model ---
    model = build_model(config, freeze_backbone=True)
    model = model.to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()

    # Phase 1 optimizer: only classifier params
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Cosine annealing scheduler (best for EfficientNet per literature)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=config.LR_MIN
    )

    # --- Training state ---
    best_val_f1 = 0.0
    best_epoch  = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'lr': []
    }

    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    for epoch in range(1, config.NUM_EPOCHS + 1):

        # Phase 2: unfreeze backbone after FREEZE_EPOCHS
        if epoch == config.FREEZE_EPOCHS + 1:
            model = unfreeze_model(model, config)
            # Reset optimizer with lower LR for fine-tuning
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE * 0.1,
                weight_decay=config.WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.NUM_EPOCHS - config.FREEZE_EPOCHS,
                eta_min=config.LR_MIN
            )
            logger.info(f"Switched to Phase 2 fine-tuning at epoch {epoch}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch, split='Val'
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        logger.info(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config.__dict__
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
            logger.info(f"  ✓ New best model saved (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {config.PATIENCE} epochs)")
                break

    # --- Final Evaluation on Test Set ---
    logger.info("=" * 60)
    logger.info("LOADING BEST MODEL FOR TEST EVALUATION")
    logger.info("=" * 60)

    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Best model from epoch {best_epoch} (Val F1: {best_val_f1:.4f})")

    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device, epoch=0, split='Test'
    )

    logger.info(f"\nTEST RESULTS:")
    logger.info(f"  Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    logger.info(f"  Weighted F1: {test_f1:.4f}")
    logger.info(f"\nPer-class report:")
    report = classification_report(
        test_labels, test_preds,
        target_names=config.CLASS_NAMES,
        digits=4
    )
    logger.info(f"\n{report}")

    # --- Save results ---
    results = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_loss': test_loss,
    }
    with open(os.path.join(config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    plot_training_curves(history, config.OUTPUT_DIR)
    plot_confusion_matrix(test_labels, test_preds, config.CLASS_NAMES,
                          config.OUTPUT_DIR, title='test_confusion_matrix')
    plot_confusion_matrix(val_labels, val_preds, config.CLASS_NAMES,
                          config.OUTPUT_DIR, title='val_confusion_matrix')

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Val F1:  {best_val_f1:.4f}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Test F1:       {test_f1:.4f}")
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
