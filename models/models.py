"""Model definitions: persistence, linear baseline, CNN, and a ResNet-style encoder-decoder.

These are lightweight PyTorch modules suitable as starting points.
"""
from typing import Tuple

import torch
import torch.nn as nn


class PersistenceModel(nn.Module):
    """Predict future as current observation (identity for target variable)."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (B, C, H, W). Persistence assumes target is same as one of input channels.
        # Here we return the first channel as a naive persistence baseline.
        return x[:, :1, ...]


class LinearBaseline(nn.Module):
    """A simple 1x1 conv linear model mapping inputs to target channels."""
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SimpleCNN(nn.Module):
    """A small convolutional encoder-decoder for image-to-image regression."""
    def __init__(self, in_ch: int, out_ch: int = 1, features: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.dec = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features, out_ch, 1)
        )

    def forward(self, x):
        h = self.enc(x)
        h = self.pool(h)
        out = self.dec(h)
        return out


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(x + self.conv(x))


class ResNetEncoderDecoder(nn.Module):
    """A ResNet-like encoder-decoder for image-to-image tasks."""
    def __init__(self, in_ch: int, out_ch: int = 1, base_filters: int = 32, nblocks: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base_filters, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(base_filters) for _ in range(nblocks)])
        self.head = nn.Conv2d(base_filters, out_ch, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h)
