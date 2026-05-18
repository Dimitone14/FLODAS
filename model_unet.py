# models_unet.py
"""
UNet model definition for 8-band PlanetScope segmentation.

This file is intentionally minimal:
- Only imports torch / torch.nn
- No training-time dependencies (albumentations, sklearn, matplotlib, etc.)

Use this in both training and inference, e.g.:

    from models_unet import UNet
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # bias=True (default)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # bias=True (default)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    UNet encoder-decoder with skip connections.

    - n_channels: number of input channels (8 for PlanetScope)
    - n_classes:  number of output channels (1 for binary logits)
    """
    def __init__(self, n_channels: int = 8, n_classes: int = 1):
        super().__init__()

        self.in_conv = DoubleConv(n_channels, 64)

        self.down1 = self.down_block(64, 128)
        self.down2 = self.down_block(128, 256)
        self.down3 = self.down_block(256, 512)
        self.down4 = self.down_block(512, 512)

        self.up1 = self.up_block(1024, 256)
        self.up2 = self.up_block(512, 128)
        self.up3 = self.up_block(256, 64)
        self.up4 = self.up_block(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def down_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Down block: MaxPool2d(2) -> DoubleConv(in_channels, out_channels)
        """
        return nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    @staticmethod
    def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Up block used in your training script:

        - ConvTranspose2d on decoder features (in_channels // 2)
        - DoubleConv on concatenated [encoder_skip, upsampled_decoder]
          with total in_channels.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2,
                kernel_size=2, stride=2
            ),
            DoubleConv(in_channels, out_channels)
        )

    @staticmethod
    def crop(enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        """
        Center-crop encoder feature map to match decoder spatial size (H, W).
        Assumes enc_feat is at least as large as dec_feat.
        """
        _, _, H, W = dec_feat.shape
        return enc_feat[:, :, :H, :W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.in_conv(x)   # (B, 64, H,   W)
        x2 = self.down1(x1)    # (B, 128, H/2, W/2)
        x3 = self.down2(x2)    # (B, 256, H/4, W/4)
        x4 = self.down3(x3)    # (B, 512, H/8, W/8)
        x5 = self.down4(x4)    # (B, 512, H/16,W/16)

        # Decoder with skip connections
        x5_up = self.up1[0](x5)  # upsample
        x = self.up1[1](torch.cat([self.crop(x4, x5_up), x5_up], dim=1))

        x_up = self.up2[0](x)
        x = self.up2[1](torch.cat([self.crop(x3, x_up), x_up], dim=1))

        x_up = self.up3[0](x)
        x = self.up3[1](torch.cat([self.crop(x2, x_up), x_up], dim=1))

        x_up = self.up4[0](x)
        x = self.up4[1](torch.cat([self.crop(x1, x_up), x_up], dim=1))

        return self.out_conv(x)


__all__ = ["UNet", "DoubleConv"]
