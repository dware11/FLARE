"""
3D breast MRI (DCE-MRI) classifier.

Architecture: same Conv3d encoder pattern as DBTClassifier (mammo) and brain MRI U-Net,
but with 3 input channels (pre-contrast + 2 post-contrast DCE phases) and smaller
spatial dimensions (32×128×128 vs 32×256×256 for DBT).

Input:  (B, 3, D=32, H=128, W=128) — 3-channel DCE-MRI volume
Output: (B, 2) logits (normal / cancer)
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two Conv3d + InstanceNorm3d + LeakyReLU — identical to brain MRI ConvBlock."""

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


class BreastMRIClassifier(nn.Module):
    """3D CNN for abbreviated DCE-MRI breast cancer classification.

    Encoder reduces (3, 32, 128, 128) → (256, 4, 16, 16) via 3× MaxPool3d(2).
    Global average pool → Linear(256, 2).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
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


def build_breast_mri_model(in_channels: int = 3, num_classes: int = 2) -> BreastMRIClassifier:
    """Return an initialised BreastMRIClassifier."""
    return BreastMRIClassifier(in_channels=in_channels, num_classes=num_classes)
