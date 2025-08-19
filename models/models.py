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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, ch, dropout: float = 0.0, use_se: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(ch)

    def forward(self, x):
        r = self.conv(x)
        if self.use_se:
            r = self.se(r)
        return nn.ReLU(inplace=True)(x + r)


class ResNetEncoderDecoder(nn.Module):
    """A deeper ResNet-like encoder-decoder with optional down/up sampling.

    This model keeps resolution the same (no complex U-Net) but adds multiple
    residual blocks and uses a final conv to map to output channels. For larger
    grids you may want to add pooling/upsampling stages.
    """
    def __init__(self, in_ch: int, out_ch: int = 1, base_filters: int = 64,
                 nblocks: int = 6, dropout: float = 0.1, use_se: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(nblocks):
            blocks.append(ResBlock(base_filters, dropout=dropout, use_se=use_se))
        self.blocks = nn.Sequential(*blocks)
        # optional bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(base_filters, out_ch, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.bottleneck(h)
        return self.head(h)
