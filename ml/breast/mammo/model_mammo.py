"""
3D DBT mammography classifier.

Architecture: 3D Conv encoder (reusing ConvBlock pattern from ml/brain/mri/train_local.py)
with global average pool + linear classification head.

Input:  (B, 1, D=32, H=256, W=256) — single-channel resampled DBT volume
Output: (B, 2) logits (normal / cancer)

Memory at batch=4 on A100 40GB: ~18 GB — feasible for training.
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two Conv3d layers with InstanceNorm3d and LeakyReLU — same pattern as brain MRI U-Net."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DBTClassifier(nn.Module):
    """3D convolutional classifier for Digital Breast Tomosynthesis (DBT) volumes.

    Encoder reduces (1, 32, 256, 256) → (256, 4, 32, 32) via 3× MaxPool3d(2).
    AdaptiveAvgPool3d(1) collapses spatial dims; Linear head maps to num_classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(128, 256)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.enc4(x)
        return self.head(x)


def build_mammo_model(num_classes: int = 2) -> DBTClassifier:
    """Return an initialised DBTClassifier."""
    return DBTClassifier(in_channels=1, num_classes=num_classes)
