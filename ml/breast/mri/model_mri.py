"""3D breast MRI classifier: single logit for BCEWithLogitsLoss."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BreastMRIClassifier3D(nn.Module):
    """Input (B, C, D, H, W) -> logits (B, 1) for BCEWithLogitsLoss."""

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base)
        self.p1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.p2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.p3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(base * 4, base * 4)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(base * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.p1(self.enc1(x))
        x = self.p2(self.enc2(x))
        x = self.p3(self.enc3(x))
        x = self.enc4(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


def build_mri_model(in_channels: int = 3) -> nn.Module:
    return BreastMRIClassifier3D(in_channels=in_channels)
