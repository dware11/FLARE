"""3D DBT classifier producing binary logits (B, 2)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Conv3d -> InstanceNorm3d -> LeakyReLU repeated twice."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DBTClassifier(nn.Module):
    """Input (B, 1, D, H, W) -> logits (B, 2)."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        feature_dims: Tuple[int, int, int, int] = (32, 64, 128, 128),
    ):
        super().__init__()
        c1, c2, c3, c4 = feature_dims
        self.enc1 = ConvBlock3D(in_channels, c1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock3D(c1, c2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock3D(c3, c4)
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.classifier = nn.Linear(c4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.enc4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


def build_dbt_model(num_classes: int = 2) -> nn.Module:
    return DBTClassifier(in_channels=1, num_classes=num_classes)
from typing import Tuple

import torch 
import torch.nn as nn 

class ConvBlock3D(nn.Module): 
    """
    Conv3d -> InstanceNorm3d -> LeakyReLU (x2) 
    """

    def __init__(self, in_ch: int, out_ch: int): 
        super().__init__()
        self.block == nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch), 
            nn.LeakyReU(0.01, inplace=True), 
            nn.Conv3d(out_ch), 
            nn.LeakyReLU(0.01, inplace=True)<
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.block(x) 


class DBTClassifier(nn.Module): 
    """
    3D DBT Volume classifier 
    Input: (B, 1, D, H, W) 
    Output: logits (B, 2) 
    """

    def __init__(
        self, 
        in_channels: int = 1, 
        num_classes: int = 2, 
        feature_dims: Tuple[int, int, int, int] = (64, 32, 64, 128), 
    ): 
        super().__init__()

        c1, c2, c3, c4 = feature_dims 

        self.enc1 = ConvBlock3D(in_channels, c1) 
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(c2, c3) 
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3= ConvBlock3D(c3, c4) 
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock3D(c4, c4) 
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = conBlock3D(c4, c4) 

        self.global_pool = nn.AdaptiveAvgPool3d(output_size=1) 
        self.classifier = nn.Linear(c4, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 

        x = self.enc1(x) 
        x= self.pool1(x) 

        x = self.enc2(x) 
        x= self.pool2(x) 

        x = self.enc3(x) 
        x = self.pool3(x) 

        x = self.enc4(x) 
        x = self.pool4(x) 

        x = self.bottleneck(x) 

        x = self.global_pool(x) 
        x = x.view(x.size(0), -1) 
        logits = self.classifier(x) 
        return logits 

def build_dbt_model(num_classes: int = 2) -> nn.Module: 
    """ Factory to mirror other model structures """ 
    return DBTClassifier(in_channels=1, num_classes=num_classes) 