# TEST EVALUATION 

# Run this to evaluate your model on the TRUE held-out test set (202 cases)
# These cases were NEVER seen during training!
#
# python3 test_evaluation.py
#

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import time

# PATHS 
TEST_DIR = "/Users/huda/Desktop/Flare_brain_mri/data/processed/test"
TEST_LABELS_PATH = "/Users/huda/Desktop/Flare_brain_mri/data/processed/test_labels.npz"
MODEL_PATH = "/Users/huda/Desktop/Flare_brain_mri/checkpoints/best_seg_model.pth"
OUTPUT_DIR = "/Users/huda/Desktop/Flare_brain_mri/test_results"

#MODEL DEFINITION
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
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(4, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.up4 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.seg_head = nn.Conv3d(32, 4, 1)
    
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


# HELPER FUNCTIONS

def compute_dice(pred, target):
    """Compute Dice for WT, TC, ET."""
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    
    # Whole Tumor (1,2,3)
    wt_p, wt_t = (pred >= 1).float(), (target >= 1).float()
    wt = (2 * (wt_p * wt_t).sum() + 1e-5) / (wt_p.sum() + wt_t.sum() + 1e-5)
    
    # Tumor Core (1,3)
    tc_p = ((pred == 1) | (pred == 3)).float()
    tc_t = ((target == 1) | (target == 3)).float()
    tc = (2 * (tc_p * tc_t).sum() + 1e-5) / (tc_p.sum() + tc_t.sum() + 1e-5)
    
    # Enhancing (3)
    et_p, et_t = (pred == 3).float(), (target == 3).float()
    et = (2 * (et_p * et_t).sum() + 1e-5) / (et_p.sum() + et_t.sum() + 1e-5)
    
    return {'WT': wt.item(), 'TC': tc.item(), 'ET': et.item()}


# MAIN
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("BRAIN TUMOR SEGMENTATION - TEST EVALUATION")
    print("="*70)
    
    # Check Paths 
    print("\nChecking paths...")
    print(f"   TEST_DIR: {'✓' if os.path.exists(TEST_DIR) else '✗ NOT FOUND'}")
    print(f"   TEST_LABELS: {'✓' if os.path.exists(TEST_LABELS_PATH) else '✗ NOT FOUND'}")
    print(f"   MODEL: {'✓' if os.path.exists(MODEL_PATH) else '✗ NOT FOUND'}")
    
    if not all([os.path.exists(TEST_DIR), os.path.exists(TEST_LABELS_PATH), os.path.exists(MODEL_PATH)]):
        print("\nMissing files! Check paths above.")
        return
    
    #  Device 
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nDevice: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\Device: CUDA GPU")
    else:
        device = torch.device("cpu")
        print(f"\nDevice: CPU")
    
    #  Load Model 
    print("\nLoading model...")
    model = SegmentationUNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
    if 'dice' in checkpoint:
        print(f"   Validation Dice WT: {checkpoint['dice'].get('WT', 0):.4f}")
        print(f"   Validation Dice TC: {checkpoint['dice'].get('TC', 0):.4f}")
        print(f"   Validation Dice ET: {checkpoint['dice'].get('ET', 0):.4f}")
    
    #  Load Test Data 
    print("\n Loading test data...")
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.npz')])
    test_labels = dict(np.load(TEST_LABELS_PATH))
    print(f"   ✓ Found {len(test_files)} test cases (NEVER seen during training!)")
    
    #  Evaluate 
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70 + "\n")
    
    all_dice = {'WT': [], 'TC': [], 'ET': []}
    all_volumes = []
    t_start = time.time()
    
    for i, fname in enumerate(test_files):
        case_id = fname.replace('.npz', '')
        
        # Load
        data = np.load(os.path.join(TEST_DIR, fname))
        image = data['image'].astype(np.float32)
        seg_gt = test_labels[case_id].astype(np.int64)
        
        # Inference
        input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(input_tensor)
            seg_pred = output.argmax(1).squeeze(0).cpu().numpy()
        
        # Dice
        dice = compute_dice(seg_pred, seg_gt)
        all_dice['WT'].append(dice['WT'])
        all_dice['TC'].append(dice['TC'])
        all_dice['ET'].append(dice['ET'])
        
        # Volume
        et_voxels = (seg_pred == 3).sum()
        all_volumes.append(et_voxels)
        
        # Progress
        if (i + 1) % 25 == 0:
            print(f"   Processed {i+1}/{len(test_files)}...")
    
    t_total = time.time() - t_start
    print(f"\nDone in {t_total:.1f}s ({t_total/len(test_files):.2f}s per case)")
    
    #  Results 
    mean_dice = {k: np.mean(v) for k, v in all_dice.items()}
    std_dice = {k: np.std(v) for k, v in all_dice.items()}
    
    print("\n" + "="*70)
    print(f" TEST RESULTS ({len(test_files)} unseen cases)")
    print("="*70)
    print(f"""
   {'Region':<25} {'Mean Dice':>12} {'± Std':>12}
   {'-'*50}
   {'Whole Tumor (WT)':<25} {mean_dice['WT']:>12.4f} {'±':>4} {std_dice['WT']:.4f}
   {'Tumor Core (TC)':<25} {mean_dice['TC']:>12.4f} {'±':>4} {std_dice['TC']:.4f}
   {'Enhancing Tumor (ET)':<25} {mean_dice['ET']:>12.4f} {'±':>4} {std_dice['ET']:.4f}
    """)
    
    # Compare with validation
    if 'dice' in checkpoint:
        print("   Comparison: Validation → Test")
        print("   " + "-"*50)
        for r in ['WT', 'TC', 'ET']:
            val = checkpoint['dice'].get(r, 0)
            test = mean_dice[r]
            diff = test - val
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"   {r}: {val:.4f} → {test:.4f} ({symbol} {abs(diff):.4f})")
    
    print("="*70)
    
    #  Surrogate Classification 
    gbm_count = sum(1 for v in all_volumes if v > 500)
    lgg_count = len(all_volumes) - gbm_count
    
    print(f"""
   🏷️  SURROGATE CLASSIFICATION (ET > 500 → GBM)
   {'-'*50}
   GBM cases:         {gbm_count:>5} ({100*gbm_count/len(all_volumes):.1f}%)
   Lower-Grade cases: {lgg_count:>5} ({100*lgg_count/len(all_volumes):.1f}%)
    """)
    
    # Save Plots 
    print("\nGenerating plots...")
    
    # Plot 1: Bar chart with error bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    regions = ['Whole Tumor\n(WT)', 'Tumor Core\n(TC)', 'Enhancing\n(ET)']
    means = [mean_dice['WT'], mean_dice['TC'], mean_dice['ET']]
    stds = [std_dice['WT'], std_dice['TC'], std_dice['ET']]
    colors = ['#9b59b6', '#e74c3c', '#f1c40f']
    
    bars = axes[0].bar(regions, means, yerr=stds, color=colors, 
                       edgecolor='black', linewidth=2, capsize=5)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Dice Score', fontsize=12)
    axes[0].set_title(f'TEST PERFORMANCE\n({len(test_files)} unseen cases)', fontsize=14, fontweight='bold')
    axes[0].axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='Good (0.8)')
    axes[0].legend()
    for bar, m, s in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                     f'{m:.3f}', ha='center', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plot
    bp = axes[1].boxplot([all_dice['WT'], all_dice['TC'], all_dice['ET']], 
                          labels=['WT', 'TC', 'ET'], patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('DICE DISTRIBUTION', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0.8, color='green', linestyle='--', linewidth=2)
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   ✓ Saved: test_results.png")
    
    #  Visualize Best Case 
    print("\n Visualizing best test case...")
    
    best_idx = np.argmax(all_dice['WT'])
    best_file = test_files[best_idx]
    case_id = best_file.replace('.npz', '')
    
    print(f"   Case: {case_id}")
    print(f"   Dice WT: {all_dice['WT'][best_idx]:.4f}")
    print(f"   Dice TC: {all_dice['TC'][best_idx]:.4f}")
    print(f"   Dice ET: {all_dice['ET'][best_idx]:.4f}")
    
    # Load
    data = np.load(os.path.join(TEST_DIR, best_file))
    image = data['image'].astype(np.float32)
    seg_gt = test_labels[case_id].astype(np.int64)
    
    # Inference
    input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(input_tensor)
        seg_pred = output.argmax(1).squeeze(0).cpu().numpy()
    
    # Best slice
    best_slice = (seg_gt > 0).sum(axis=(0, 1)).argmax()
    
    # Plot
    seg_cmap = ListedColormap(['black', '#3498db', '#2ecc71', '#f1c40f'])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image[1, :, :, best_slice].T, cmap='gray', origin='lower')
    axes[0].set_title('T1c Input', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(seg_gt[:, :, best_slice].T, cmap=seg_cmap, origin='lower', vmin=0, vmax=3)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(seg_pred[:, :, best_slice].T, cmap=seg_cmap, origin='lower', vmin=0, vmax=3)
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(image[1, :, :, best_slice].T, cmap='gray', origin='lower')
    overlay = np.ma.masked_where(seg_pred[:, :, best_slice] == 0, seg_pred[:, :, best_slice])
    axes[3].imshow(overlay.T, cmap=seg_cmap, origin='lower', vmin=0, vmax=3, alpha=0.6)
    axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    patches = [
        mpatches.Patch(color='#3498db', label='Necrotic Core'),
        mpatches.Patch(color='#2ecc71', label='Edema'),
        mpatches.Patch(color='#f1c40f', label='Enhancing Tumor')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=11)
    
    plt.suptitle(f'TEST CASE: {case_id} (Slice {best_slice})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_visualization.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   ✓ Saved: test_visualization.png")
    
    #  Final Summary 
    print("\n" + "="*70)
    print("TEST EVALUATION COMPLETE")
    print("="*70)
    print(f"""
   DATA SPLITS:
      Train:      942 cases (model learned from these)
      Validation: 201 cases (used during training)
      Test:       {len(test_files)} cases (TRUE held-out - NEVER seen!)
   
   TEST DICE SCORES:
      Whole Tumor (WT):  {mean_dice['WT']:.4f} ± {std_dice['WT']:.4f}
      Tumor Core (TC):   {mean_dice['TC']:.4f} ± {std_dice['TC']:.4f}
      Enhancing (ET):    {mean_dice['ET']:.4f} ± {std_dice['ET']:.4f}
   
   CLASSIFICATION:
      GBM:         {gbm_count} cases ({100*gbm_count/len(all_volumes):.1f}%)
      Lower-Grade: {lgg_count} cases ({100*lgg_count/len(all_volumes):.1f}%)
   
   OUTPUT FILES:
      {OUTPUT_DIR}/test_results.png
      {OUTPUT_DIR}/test_visualization.png
    """)
    print("="*70)


if __name__ == "__main__":
    main()
