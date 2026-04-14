"""
U-Net Generator for Pix2Pix Floor Plan Renderer

This module implements the U-Net generator architecture for the Pix2Pix model.
The generator takes a semantic segmentation map (simple colored boxes) and
transforms it into a realistic floor plan rendering.

Architecture:
- Encoder: 8 downsampling blocks (64→128→256→512→512→512→512→512)
- Decoder: 8 upsampling blocks with skip connections (512→512→512→512→256→128→64→3)
- Skip connections from encoder to decoder for preserving spatial information
- Tanh activation at output for normalized RGB images
"""

import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    """Single U-Net block with Conv-BatchNorm-Activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down: bool = True,
        use_dropout: bool = False,
        use_batchnorm: bool = True
    ):
        super().__init__()

        if down:
            # Encoder block: Conv(stride=2) for downsampling
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            )
        else:
            # Decoder block: ConvTranspose for upsampling
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            )

        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        self.activation = nn.LeakyReLU(0.2) if down else nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class UNetGenerator(nn.Module):
    """
    U-Net Generator for Pix2Pix

    Transforms simple layout (semantic map) into realistic floor plan image.

    Input: (B, 3, 256, 256) - Semantic segmentation map
           - Different colors for different room types
           - Simple colored rectangles

    Output: (B, 3, 256, 256) - Realistic floor plan image
            - Proper walls, doors, windows
            - Textures and architectural symbols
            - Professional rendering quality

    Architecture:
        Encoder (Downsampling):
            3 → 64 → 128 → 256 → 512 → 512 → 512 → 512 → 512
            256x256 → 128x128 → 64x64 → 32x32 → 16x16 → 8x8 → 4x4 → 2x2 → 1x1

        Decoder (Upsampling with Skip Connections):
            512 → 512 → 512 → 512 → 256 → 128 → 64 → 3
            1x1 → 2x2 → 4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128 → 256x256
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encoder (Downsampling path)
        # No batch norm on first layer
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.down2 = UNetBlock(64, 128, down=True)         # 128x128 → 64x64
        self.down3 = UNetBlock(128, 256, down=True)        # 64x64 → 32x32
        self.down4 = UNetBlock(256, 512, down=True)        # 32x32 → 16x16
        self.down5 = UNetBlock(512, 512, down=True)        # 16x16 → 8x8
        self.down6 = UNetBlock(512, 512, down=True)        # 8x8 → 4x4
        self.down7 = UNetBlock(512, 512, down=True)        # 4x4 → 2x2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 2x2 → 1x1
            nn.ReLU()
        )

        # Decoder (Upsampling path with skip connections)
        # Dropout on first 3 layers for regularization
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)      # 1x1 → 2x2
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)     # 2x2 → 4x4 (1024 = 512 + 512 skip)
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)     # 4x4 → 8x8
        self.up4 = UNetBlock(1024, 512, down=False)                       # 8x8 → 16x16
        self.up5 = UNetBlock(1024, 256, down=False)                       # 16x16 → 32x32
        self.up6 = UNetBlock(512, 128, down=False)                        # 32x32 → 64x64
        self.up7 = UNetBlock(256, 64, down=False)                         # 64x64 → 128x128

        # Final layer: 128x128 → 256x256, no batch norm
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        """
        Forward pass with skip connections

        Args:
            x: Input tensor (B, 3, 256, 256) - semantic segmentation map

        Returns:
            Generated image (B, 3, 256, 256) - realistic floor plan
        """
        # Encoder
        d1 = self.down1(x)      # (B, 64, 128, 128)
        d2 = self.down2(d1)     # (B, 128, 64, 64)
        d3 = self.down3(d2)     # (B, 256, 32, 32)
        d4 = self.down4(d3)     # (B, 512, 16, 16)
        d5 = self.down5(d4)     # (B, 512, 8, 8)
        d6 = self.down6(d5)     # (B, 512, 4, 4)
        d7 = self.down7(d6)     # (B, 512, 2, 2)

        # Bottleneck
        bottleneck = self.bottleneck(d7)  # (B, 512, 1, 1)

        # Decoder with skip connections
        u1 = self.up1(bottleneck)                        # (B, 512, 2, 2)
        u2 = self.up2(torch.cat([u1, d7], dim=1))        # (B, 512, 4, 4)
        u3 = self.up3(torch.cat([u2, d6], dim=1))        # (B, 512, 8, 8)
        u4 = self.up4(torch.cat([u3, d5], dim=1))        # (B, 512, 16, 16)
        u5 = self.up5(torch.cat([u4, d4], dim=1))        # (B, 256, 32, 32)
        u6 = self.up6(torch.cat([u5, d3], dim=1))        # (B, 128, 64, 64)
        u7 = self.up7(torch.cat([u6, d2], dim=1))        # (B, 64, 128, 128)

        # Final layer
        output = self.final(torch.cat([u7, d1], dim=1))  # (B, 3, 256, 256)

        return output


def test_generator():
    """Test the U-Net generator with dummy input"""
    print("Testing U-Net Generator...")

    # Create model
    generator = UNetGenerator(in_channels=3, out_channels=3)

    # Count parameters
    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test with dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = generator(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Verify output is in correct range
    assert output.shape == (batch_size, 3, 256, 256), "Output shape mismatch"
    assert output.min() >= -1.0 and output.max() <= 1.0, "Output not in [-1, 1] range"

    print("\n✓ Generator test passed!")

    return generator


if __name__ == "__main__":
    test_generator()
