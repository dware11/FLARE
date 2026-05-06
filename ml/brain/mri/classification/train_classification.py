"""
FLARE - MRI Brain Tumor Classification Training Script (5-Class)
Model: EfficientNetB0 (Transfer Learning)
Dataset: BRISC (4 classes) + IXI (1 class = NORMAL)
Classes: glioma, meningioma, pituitary, no_tumor, normal

METRICS:
- Accuracy, F1-Score, Confusion Matrix (standard ML)
- Sensitivity, Specificity, PPV, NPV (clinical )
- ROC-AUC curves (discrimination ability)
- Per-class performance breakdown
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
# ==============================================================================

class Config:
    # --- Paths ---
    BRISC_DIR = '/scratch/bckk/flare/mri_brain/data/brisc_processed'
    IXI_DIR = '/scratch/bckk/flare/mri_brain/data/ixi_processed'
    OUTPUT_DIR = '/scratch/bckk/flare/mri_brain/classification/outputs/training_5class'

    # --- Model ---
    NUM_CLASSES = 5
    INPUT_SIZE = 224
    
    # --- Training ---
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.3
    PATIENCE = 10
    FREEZE_EPOCHS = 5
    LR_MIN = 1e-6

    # --- Data Splitting ---
    RANDOM_SEED = 42
    NUM_WORKERS = 4
    TRAIN_RATIO = 0.7    # 70% training
    VAL_RATIO = 0.15     # 15% validation
    TEST_RATIO = 0.15    # 15% testing

    # --- Class labels ---
    CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor', 'normal']

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

# DATASET
# ==============================================================================

class BRISCIXIDataset(Dataset):
    """Load NPZ files from BRISC and IXI directories"""

    def __init__(self, npz_files, transform=None, config=None):
        self.config = config or Config()
        self.transform = transform
        self.npz_files = npz_files
        
        logging.info(f"Loaded {len(self.npz_files)} NPZ files")
        
        from collections import Counter
        labels = []
        for f in self.npz_files:
            data = np.load(f, allow_pickle=True)
            labels.append(int(data['label']))
        
        label_counts = Counter(labels)
        for label_idx in range(self.config.NUM_CLASSES):
            name = self.config.CLASS_NAMES[label_idx]
            count = label_counts.get(label_idx, 0)
            logging.info(f"  {name}: {count} samples")

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        filepath = self.npz_files[idx]
        data = np.load(filepath, allow_pickle=True)
        
        image = data['image'].astype(np.float32)
        label = int(data['label'])

        # Normalize to [0, 255]
        if image.max() <= 1.0:
            image = image * 255.0

        # Convert grayscale to 3-channel
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=0)
        elif image.ndim == 3 and image.shape[0] == 1:
            image = np.concatenate([image, image, image], axis=0)

        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# DATA TRANSFORMS
# ==============================================================================

def get_transforms(config, split='train'):
    input_size = config.INPUT_SIZE

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
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

# MODEL
# ==============================================================================

def build_model(config, freeze_backbone=True):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        logging.info("Phase 1: Backbone FROZEN — training classifier head only")
    else:
        for param in model.parameters():
            param.requires_grad = True
        logging.info("Phase 2: Full model UNFROZEN — fine-tuning end-to-end")

    in_features = model.classifier[1].in_features
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
    for param in model.parameters():
        param.requires_grad = True
    logging.info("Phase 2: All layers unfrozen for fine-tuning")
    return model

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
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{split}]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1)
            
            all_probs.extend(probs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    return avg_loss, accuracy, f1, all_preds, all_labels, all_probs

# METRICS COMPUTATION
# ==============================================================================

def compute_metrics(all_labels, all_preds, all_probs, class_names, output_dir, split='test'):
    """
    Compute comprehensive metrics: standard ML + clinical
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"TEST SET EVALUATION METRICS")
    logging.info(f"{'='*80}")
    
    # STANDARD ML METRICS
    # ==================================================================
    accuracy = np.mean(all_preds == all_labels)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    logging.info(f"\n[STANDARD ML METRICS]")
    logging.info(f"  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    logging.info(f"  Weighted F1:     {f1_weighted:.4f}")
    logging.info(f"  Macro F1:        {f1_macro:.4f}")
    
    # PER-CLASS BREAKDOWN
    # ==================================================================
    logging.info(f"\n[PER-CLASS BREAKDOWN]")
    logging.info(classification_report(all_labels, all_preds, 
                                       target_names=class_names, digits=4))
    
    # CONFUSION MATRIX
    # ==================================================================
    cm = confusion_matrix(all_labels, all_preds)
    logging.info(f"\n[CONFUSION MATRIX]")
    logging.info(f"\nRows=True Label, Cols=Predicted Label:")
    logging.info(f"{cm}")
    
    # Normalized confusion matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    logging.info(f"\n[CONFUSION MATRIX - Normalized (%):]")
    logging.info((cm_norm * 100).astype(int))
    
    # CLINICAL METRICS: SENSITIVITY, SPECIFICITY, PPV, NPV
    # ==================================================================
    logging.info(f"\n[CLINICAL METRICS - Per Class]")
    logging.info(f"(Sensitivity=Recall, Specificity, PPV, NPV)")
    
    sensitivity_per_class = {}
    specificity_per_class = {}
    ppv_per_class = {}
    npv_per_class = {}
    
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        sensitivity_per_class[cls] = round(float(sensitivity), 4)
        specificity_per_class[cls] = round(float(specificity), 4)
        ppv_per_class[cls] = round(float(ppv), 4)
        npv_per_class[cls] = round(float(npv), 4)
        
        logging.info(f"\n  {cls.upper()}:")
        logging.info(f"    Sensitivity (catches true cases):  {sensitivity:.4f}")
        logging.info(f"    Specificity (correct negatives):   {specificity:.4f}")
        logging.info(f"    PPV (confidence when YES):         {ppv:.4f}")
        logging.info(f"    NPV (confidence when NO):          {npv:.4f}")
    
    # ROC-AUC CURVES
    # ==================================================================
    logging.info(f"\n[ROC-AUC SCORES]")
    auc_scores = {}
    try:
        for i, cls in enumerate(class_names):
            binary_labels = (all_labels == i).astype(int)
            class_probs = all_probs[:, i]
            auc = roc_auc_score(binary_labels, class_probs)
            auc_scores[cls] = round(float(auc), 4)
            logging.info(f"  {cls:15s}: AUC = {auc:.4f}")
        
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        logging.info(f"  Macro Average:       {macro_auc:.4f}")
    except Exception as e:
        logging.warning(f"Could not compute ROC-AUC: {e}")
        macro_auc = None
    
    # CONFIDENCE ANALYSIS
    # ==================================================================
    logging.info(f"\n[CONFIDENCE ANALYSIS]")
    correct_mask = all_preds == all_labels
    incorrect_mask = all_preds != all_labels
    
    correct_confidences = np.max(all_probs[correct_mask], axis=1) if correct_mask.sum() > 0 else np.array([])
    incorrect_confidences = np.max(all_probs[incorrect_mask], axis=1) if incorrect_mask.sum() > 0 else np.array([])
    
    if len(correct_confidences) > 0:
        logging.info(f"  Avg confidence when CORRECT: {correct_confidences.mean():.4f}")
    if len(incorrect_confidences) > 0:
        logging.info(f"  Avg confidence when WRONG:   {incorrect_confidences.mean():.4f}")
    
    # ERROR ANALYSIS
    # ==================================================================
    logging.info(f"\n[TOP ERRORS - Most Confused Pairs]")
    errors = []
    for i, true_cls in enumerate(class_names):
        for j, pred_cls in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], true_cls, pred_cls))
    
    errors.sort(reverse=True)
    for count, true_cls, pred_cls in errors[:5]:
        logging.info(f"  {count:3d} cases: {true_cls:12s} → {pred_cls}")
    
    # CLASS DISTRIBUTION
    # ==================================================================
    logging.info(f"\n[CLASS DISTRIBUTION IN TEST SET]")
    unique, counts = np.unique(all_labels, return_counts=True)
    for idx, count in zip(unique, counts):
        logging.info(f"  {class_names[idx]:15s}: {count:4d} samples ({count/len(all_labels)*100:.1f}%)")
    
    # Save all results
    results = {
        'accuracy': round(float(accuracy), 4),
        'f1_weighted': round(float(f1_weighted), 4),
        'f1_macro': round(float(f1_macro), 4),
        'sensitivity_per_class': sensitivity_per_class,
        'specificity_per_class': specificity_per_class,
        'ppv_per_class': ppv_per_class,
        'npv_per_class': npv_per_class,
        'per_class_auc': auc_scores,
        'macro_auc': round(float(macro_auc), 4) if macro_auc else None,
        'n_samples': len(all_labels),
    }
    
    with open(os.path.join(output_dir, f'{split}_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nMetrics saved to: {os.path.join(output_dir, f'{split}_metrics.json')}")
    logging.info(f"{'='*80}\n")
    
    return results

# PLOTTING
# ==============================================================================

def plot_roc_curves(all_labels, all_probs, class_names, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, cls in enumerate(class_names):
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        
        try:
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            auc = roc_auc_score(binary_labels, class_probs)
            ax.plot(fpr, tpr, label=f'{cls} (AUC = {auc:.4f})', linewidth=2)
        except:
            pass
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - 5-Class Classification', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_test.png'), dpi=150)
    plt.close()
    logging.info("Saved: roc_curves_test.png")

def plot_confusion_matrix(labels, preds, class_names, output_dir, title='test_confusion_matrix'):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Raw Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}.png'), dpi=150)
    plt.close()
    logging.info(f"Saved: {title}.png")

# MAIN TRAINING
# ==============================================================================

def main():
    config = Config()
    logger = setup_logging(config.OUTPUT_DIR)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # DATA LOADING & SPLITTING (70/15/15)
    # ==================================================================
    logger.info("\n" + "="*80)
    logger.info("DATA LOADING & SPLITTING")
    logger.info("="*80)
    
    brisc_files = sorted(Path(config.BRISC_DIR).glob("classification_*.npz"))
    ixi_files = sorted(Path(config.IXI_DIR).glob("classification_*.npz"))
    
    logger.info(f"BRISC files: {len(brisc_files)}")
    logger.info(f"IXI files: {len(ixi_files)}")
    
    all_files = list(brisc_files) + list(ixi_files)
    logger.info(f"Total files: {len(all_files)}")

    # SPLIT: 70% train / 15% val / 15% test
    logger.info(f"\nData Split Strategy:")
    logger.info(f"  Train: {config.TRAIN_RATIO*100:.0f}% = {int(len(all_files)*config.TRAIN_RATIO)}")
    logger.info(f"  Val:   {config.VAL_RATIO*100:.0f}% = {int(len(all_files)*config.VAL_RATIO)}")
    logger.info(f"  Test:  {config.TEST_RATIO*100:.0f}% = {int(len(all_files)*config.TEST_RATIO)}")
    
    indices = np.arange(len(all_files))
    train_idx, test_idx = train_test_split(
        indices, test_size=config.TEST_RATIO, random_state=config.RANDOM_SEED
    )
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO),
        random_state=config.RANDOM_SEED
    )
    
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    test_files = [all_files[i] for i in test_idx]
    
    logger.info(f"\nSplit results:")
    logger.info(f"  Train: {len(train_files)} files")
    logger.info(f"  Val:   {len(val_files)} files")
    logger.info(f"  Test:  {len(test_files)} files")

    logger.info("\nCreating datasets...")
    train_dataset = BRISCIXIDataset(
        train_files,
        transform=get_transforms(config, 'train'),
        config=config
    )
    val_dataset = BRISCIXIDataset(
        val_files,
        transform=get_transforms(config, 'val'),
        config=config
    )
    test_dataset = BRISCIXIDataset(
        test_files,
        transform=get_transforms(config, 'test'),
        config=config
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=config.NUM_WORKERS,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS,
                          pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=config.NUM_WORKERS,
                           pin_memory=True)

    # MODEL & TRAINING SETUP
    # ==================================================================
    model = build_model(config, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=config.LR_MIN
    )

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)

    for epoch in range(1, config.NUM_EPOCHS + 1):

        if epoch == config.FREEZE_EPOCHS + 1:
            model = unfreeze_model(model, config)
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

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_loss, val_acc, val_f1, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device, epoch, split='Val'
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

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
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
            logger.info(f"  ✓ New best model saved (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # TEST EVALUATION WITH ALL METRICS
    # ==================================================================
    logger.info("\n" + "="*80)
    logger.info("LOADING BEST MODEL FOR TEST EVALUATION")
    logger.info("="*80)

    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Best model from epoch {best_epoch} (Val F1: {best_val_f1:.4f})")

    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device, epoch=0, split='Test'
    )

    # Compute metrics (standard + clinical)
    test_results = compute_metrics(
        test_labels, test_preds, test_probs,
        config.CLASS_NAMES, config.OUTPUT_DIR, split='test'
    )

    # Generate plots
    plot_confusion_matrix(test_labels, test_preds, config.CLASS_NAMES,
                          config.OUTPUT_DIR, title='test_confusion_matrix')
    plot_roc_curves(test_labels, test_probs, config.CLASS_NAMES, config.OUTPUT_DIR)

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Val F1:      {best_val_f1:.4f}")
    logger.info(f"Test Accuracy:    {test_acc*100:.2f}%")
    logger.info(f"Test Weighted F1: {test_f1:.4f}")
    logger.info(f"Outputs saved to: {config.OUTPUT_DIR}")
    logger.info("="*80)

    return model, test_results

if __name__ == '__main__':
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model, results = main()
