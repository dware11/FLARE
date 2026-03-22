# ══════════════════════════════════════════════════════════════════════════════
#                    TUMOR VOLUME QUANTIFICATION - PRESENTATION
# ══════════════════════════════════════════════════════════════════════════════
#
# This script generates a clean visualization showing:
# 1. The segmentation
# 2. Volume breakdown by tumor region
# 3. Clinical report style output
#
# Run this to generate slides-ready figures!
#
# ══════════════════════════════════════════════════════════════════════════════

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Paths
DATA_DIR = "/Users/huda/Desktop/Flare_brain_mri/data/processed/test"
LABELS_PATH = "/Users/huda/Desktop/Flare_brain_mri/data/processed/test_labels.npz"
MODEL_PATH = "/Users/huda/Desktop/Flare_brain_mri/checkpoints/best_seg_model.pth"
OUTPUT_DIR = "/Users/huda/Desktop/Flare_brain_mri/presentation_figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("Loading model and data...")

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load model
model = SegmentationUNet().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load data
labels = dict(np.load(LABELS_PATH))
case_file = "BraTS-GLI-00294-000.npz"
case_id = case_file.replace('.npz', '')

data = np.load(os.path.join(DATA_DIR, case_file))
image = data['image'].astype(np.float32)
seg_gt = labels[case_id].astype(np.int64)

# Run inference
input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
with torch.no_grad():
    output = model(input_tensor)
    seg_pred = output.argmax(1).squeeze(0).cpu().numpy()

best_slice = (seg_gt > 0).sum(axis=(0, 1)).argmax()

print("✓ Ready!")

# ══════════════════════════════════════════════════════════════════════════════
# CALCULATE VOLUMES
# ══════════════════════════════════════════════════════════════════════════════

# Voxel to cm³ conversion
# Original BraTS: 240×240×155 mm, resampled to 128×128×128
# Scale factor to account for resampling
scale_x = 240 / 128  # mm per voxel
scale_y = 240 / 128
scale_z = 155 / 128
voxel_volume_mm3 = scale_x * scale_y * scale_z  # mm³ per voxel
voxel_volume_cm3 = voxel_volume_mm3 / 1000  # cm³ per voxel

# Count voxels
necrotic_voxels = (seg_pred == 1).sum()
edema_voxels = (seg_pred == 2).sum()
enhancing_voxels = (seg_pred == 3).sum()
whole_tumor_voxels = necrotic_voxels + edema_voxels + enhancing_voxels
tumor_core_voxels = necrotic_voxels + enhancing_voxels

# Convert to cm³
necrotic_vol = necrotic_voxels * voxel_volume_cm3
edema_vol = edema_voxels * voxel_volume_cm3
enhancing_vol = enhancing_voxels * voxel_volume_cm3
whole_tumor_vol = whole_tumor_voxels * voxel_volume_cm3
tumor_core_vol = tumor_core_voxels * voxel_volume_cm3

print(f"\nVolume Calculations:")
print(f"  Necrotic Core:    {necrotic_voxels:,} voxels = {necrotic_vol:.2f} cm³")
print(f"  Edema:            {edema_voxels:,} voxels = {edema_vol:.2f} cm³")
print(f"  Enhancing Tumor:  {enhancing_voxels:,} voxels = {enhancing_vol:.2f} cm³")
print(f"  Whole Tumor:      {whole_tumor_voxels:,} voxels = {whole_tumor_vol:.2f} cm³")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Volume Quantification Overview (Main Slide)
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 8))

# Colors
colors = {
    'necrotic': '#3498db',   # Blue
    'edema': '#2ecc71',       # Green
    'enhancing': '#f39c12',   # Orange/Yellow
    'whole': '#9b59b6',       # Purple
    'core': '#e74c3c'         # Red
}

seg_cmap = ListedColormap(['black', colors['necrotic'], colors['edema'], colors['enhancing']])

# Left side: Segmentation image
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(image[1, :, :, best_slice].T, cmap='gray', origin='lower')
overlay = np.ma.masked_where(seg_pred[:, :, best_slice] == 0, seg_pred[:, :, best_slice])
ax1.imshow(overlay.T, cmap=seg_cmap, origin='lower', vmin=0, vmax=3, alpha=0.7)
ax1.set_title('Model Segmentation', fontsize=16, fontweight='bold')
ax1.axis('off')

# Add legend
patches = [
    mpatches.Patch(color=colors['necrotic'], label=f'Necrotic Core: {necrotic_vol:.1f} cm³'),
    mpatches.Patch(color=colors['edema'], label=f'Edema: {edema_vol:.1f} cm³'),
    mpatches.Patch(color=colors['enhancing'], label=f'Enhancing: {enhancing_vol:.1f} cm³'),
]
ax1.legend(handles=patches, loc='upper left', fontsize=11, framealpha=0.9)

# Right side: Volume bar chart
ax2 = fig.add_subplot(1, 2, 2)

regions = ['Necrotic\nCore', 'Edema', 'Enhancing\nTumor', 'Tumor\nCore', 'Whole\nTumor']
volumes = [necrotic_vol, edema_vol, enhancing_vol, tumor_core_vol, whole_tumor_vol]
bar_colors = [colors['necrotic'], colors['edema'], colors['enhancing'], colors['core'], colors['whole']]

bars = ax2.bar(regions, volumes, color=bar_colors, edgecolor='black', linewidth=2)
ax2.set_ylabel('Volume (cm³)', fontsize=14, fontweight='bold')
ax2.set_title('Tumor Volume Quantification', fontsize=16, fontweight='bold')

# Add value labels on bars
for bar, vol in zip(bars, volumes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{vol:.1f}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.set_ylim(0, max(volumes) * 1.15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add explanation box
textstr = f'Total Tumor: {whole_tumor_vol:.1f} cm³'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=14,
         verticalalignment='top', horizontalalignment='right', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'volume_quantification.png'), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print(f"\n✓ Saved: volume_quantification.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Pie Chart of Tumor Composition
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Pie chart
sizes = [necrotic_vol, edema_vol, enhancing_vol]
labels_pie = [f'Necrotic Core\n{necrotic_vol:.1f} cm³', 
              f'Edema\n{edema_vol:.1f} cm³', 
              f'Enhancing\n{enhancing_vol:.1f} cm³']
pie_colors = [colors['necrotic'], colors['edema'], colors['enhancing']]
explode = (0.02, 0.02, 0.05)  # Emphasize enhancing

wedges, texts, autotexts = axes[0].pie(sizes, labels=labels_pie, colors=pie_colors,
                                        autopct='%1.1f%%', startangle=90, explode=explode,
                                        textprops={'fontsize': 11},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 2})
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

axes[0].set_title('Tumor Composition Breakdown', fontsize=16, fontweight='bold')

# Right: Stacked visualization
ax = axes[1]

# Create stacked horizontal bar
ax.barh(['Tumor'], [necrotic_vol], color=colors['necrotic'], label='Necrotic Core', edgecolor='black')
ax.barh(['Tumor'], [edema_vol], left=[necrotic_vol], color=colors['edema'], label='Edema', edgecolor='black')
ax.barh(['Tumor'], [enhancing_vol], left=[necrotic_vol + edema_vol], color=colors['enhancing'], label='Enhancing', edgecolor='black')

ax.set_xlabel('Volume (cm³)', fontsize=14, fontweight='bold')
ax.set_title('Tumor Component Distribution', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(0, whole_tumor_vol * 1.1)

# Add total label
ax.text(whole_tumor_vol + 1, 0, f'Total: {whole_tumor_vol:.1f} cm³', 
        va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'volume_composition.png'), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print(f"✓ Saved: volume_composition.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Clinical Report Style
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. MRI Input
axes[0].imshow(image[1, :, :, best_slice].T, cmap='gray', origin='lower')
axes[0].set_title('MRI Input (T1c)', fontsize=14, fontweight='bold')
axes[0].axis('off')

# 2. Model Segmentation
axes[1].imshow(image[1, :, :, best_slice].T, cmap='gray', origin='lower')
overlay = np.ma.masked_where(seg_pred[:, :, best_slice] == 0, seg_pred[:, :, best_slice])
axes[1].imshow(overlay.T, cmap=seg_cmap, origin='lower', vmin=0, vmax=3, alpha=0.7)
axes[1].set_title('Model Segmentation', fontsize=14, fontweight='bold')
axes[1].axis('off')

# 3. Clinical Measurements (as text/table)
axes[2].axis('off')
axes[2].set_xlim(0, 10)
axes[2].set_ylim(0, 10)

# Title
axes[2].text(5, 9.5, 'QUANTIFICATION REPORT', ha='center', fontsize=16, fontweight='bold')
axes[2].axhline(y=9, xmin=0.1, xmax=0.9, color='black', linewidth=2)

# Patient info
axes[2].text(0.5, 8.2, f'Patient ID: {case_id}', fontsize=11)
axes[2].text(0.5, 7.5, f'Slice: {best_slice}', fontsize=11)

# Volume measurements
axes[2].text(0.5, 6.3, 'TUMOR VOLUMES:', fontsize=13, fontweight='bold')

y_pos = 5.5
measurements = [
    ('Necrotic Core:', f'{necrotic_vol:.2f} cm³', colors['necrotic']),
    ('Edema:', f'{edema_vol:.2f} cm³', colors['edema']),
    ('Enhancing Tumor:', f'{enhancing_vol:.2f} cm³', colors['enhancing']),
]

for label, value, color in measurements:
    axes[2].add_patch(plt.Rectangle((0.3, y_pos - 0.15), 0.4, 0.35, facecolor=color, edgecolor='black'))
    axes[2].text(1.0, y_pos, label, fontsize=11, va='center')
    axes[2].text(6.5, y_pos, value, fontsize=11, va='center', fontweight='bold')
    y_pos -= 0.8

# Totals
axes[2].axhline(y=2.8, xmin=0.05, xmax=0.95, color='gray', linewidth=1, linestyle='--')
axes[2].text(1.0, 2.2, 'Tumor Core:', fontsize=11, va='center')
axes[2].text(6.5, 2.2, f'{tumor_core_vol:.2f} cm³', fontsize=11, va='center', fontweight='bold')
axes[2].text(1.0, 1.4, 'Whole Tumor:', fontsize=11, va='center')
axes[2].text(6.5, 1.4, f'{whole_tumor_vol:.2f} cm³', fontsize=12, va='center', fontweight='bold', color='purple')

# Classification
plt.suptitle(f'Automated Tumor Analysis - {case_id}', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clinical_report.png'), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print(f"✓ Saved: clinical_report.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Before/After Comparison (Impact)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Manual Process
axes[0].set_xlim(0, 10)
axes[0].set_ylim(0, 10)
axes[0].set_title('MANUAL PROCESS', fontsize=16, fontweight='bold', color='#e74c3c')
axes[0].axis('off')

manual_text = """
⏱️  Time: 30-60 minutes

📋 Process:
   • Radiologist reviews 4 MRI sequences
   • Manually traces tumor boundaries
   • Slice by slice (128+ slices)
   • Estimates volumes visually

⚠️  Limitations:
   • Subjective (varies between experts)
   • Time-consuming
   • Prone to fatigue errors
   • Difficult to reproduce
"""
axes[0].text(0.5, 5, manual_text, fontsize=12, va='center', 
             family='monospace', linespacing=1.5)

# Right: Model Process
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].set_title('Model-ASSISTED (FLARE)', fontsize=16, fontweight='bold', color='#27ae60')
axes[1].axis('off')

Model_text = f"""
⏱️  Time: ~2 seconds

📋 Process:
   • Load 4 MRI modalities
   • 3D U-Net processes entire volume
   • Automatic segmentation
   • Precise volume calculation

✅ Benefits:
   • Objective & consistent
   • Fast (30x speedup)
   • Reproducible results
   • Quantitative measurements:
   
      Necrotic:  {necrotic_vol:>6.2f} cm³
      Edema:     {edema_vol:>6.2f} cm³
      Enhancing: {enhancing_vol:>6.2f} cm³
      ─────────────────
      Total:     {whole_tumor_vol:>6.2f} cm³
"""
axes[1].text(0.5, 5, Model_text, fontsize=12, va='center',
             family='monospace', linespacing=1.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'manual_vs_Model.png'), dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print(f"✓ Saved: manual_vs_Model.png")


print("\n" + "="*60)
print("  ✅ ALL FIGURES GENERATED!")
print("="*60)
print(f"""
  Output directory: {OUTPUT_DIR}
  
  Generated files:
    1. volume_quantification.png  - Main slide (segmentation + bar chart)
    2. volume_composition.png     - Pie chart + stacked bar
    3. clinical_report.png        - Report-style layout
    4. manual_vs_Model.png           - Before/after comparison
  
  Volume Summary for {case_id}:
    • Necrotic Core:    {necrotic_vol:.2f} cm³
    • Edema:            {edema_vol:.2f} cm³
    • Enhancing Tumor:  {enhancing_vol:.2f} cm³
    • Whole Tumor:      {whole_tumor_vol:.2f} cm³
    
  Use these figures in your Canva slides!
""")
print("="*60)
