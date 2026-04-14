"""
PatchGAN Discriminator for Pix2Pix Floor Plan Renderer

This module implements the PatchGAN discriminator that classifies whether
overlapping image patches are real or fake.

Instead of outputting a single scalar (real/fake for entire image),
PatchGAN outputs an N×N matrix where each value classifies an image patch.

This approach:
- Encourages sharp, high-frequency details (textures, edges)
- Runs faster than full-image discriminator
- Has fewer parameters
- Penalizes structure at the scale of image patches
"""

import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """
    Single convolutional block for discriminator

    Architecture: Conv → BatchNorm → LeakyReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_batchnorm: bool = True
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, padding=1, bias=False
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (70×70 receptive field)

    Classifies overlapping 70×70 patches as real or fake.

    Input: (B, 6, 256, 256) - Concatenated input and target images
           - First 3 channels: input semantic map
           - Last 3 channels: target/generated realistic image

    Output: (B, 1, 30, 30) - Patch predictions
            - Each value is the probability that the corresponding
              70×70 patch in the input is real (not generated)

    Architecture:
        Input (6 channels) → Conv blocks → Output (1 channel)
        6 → 64 → 128 → 256 → 512 → 1
        256×256 → 128×128 → 64×64 → 32×32 → 31×31 → 30×30

    Receptive Field Calculation:
    - Each layer with stride=2 doubles the receptive field
    - Final receptive field: 70×70 pixels
    """

    def __init__(self, in_channels: int = 3):
        """
        Args:
            in_channels: Number of channels in each image (usually 3 for RGB)
                        Input will be concatenated condition + target = 2 * in_channels
        """
        super().__init__()

        # First layer: no batch norm
        self.initial = DiscriminatorBlock(
            in_channels * 2,  # Concatenate input and target
            64,
            stride=2,
            use_batchnorm=False
        )

        # Middle layers with batch norm
        self.down1 = DiscriminatorBlock(64, 128, stride=2)    # 128x128 → 64x64
        self.down2 = DiscriminatorBlock(128, 256, stride=2)   # 64x64 → 32x32
        self.down3 = DiscriminatorBlock(256, 512, stride=1)   # 32x32 → 31x31

        # Final layer: reduce to 1 channel (real/fake prediction per patch)
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # 31x31 → 30x30
            # No sigmoid here - we'll use BCEWithLogitsLoss which applies sigmoid internally
        )

    def forward(self, condition, target):
        """
        Forward pass

        Args:
            condition: Input semantic map (B, 3, 256, 256)
            target: Real or generated image (B, 3, 256, 256)

        Returns:
            Patch predictions (B, 1, 30, 30) - logits for each 70x70 patch
        """
        # Concatenate input and target along channel dimension
        x = torch.cat([condition, target], dim=1)  # (B, 6, 256, 256)

        # Apply discriminator blocks
        x = self.initial(x)  # (B, 64, 128, 128)
        x = self.down1(x)    # (B, 128, 64, 64)
        x = self.down2(x)    # (B, 256, 32, 32)
        x = self.down3(x)    # (B, 512, 31, 31)
        x = self.final(x)    # (B, 1, 30, 30)

        return x


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that operates at multiple image resolutions

    This is an optional enhancement to PatchGAN that:
    - Runs 3 discriminators at different scales (1x, 0.5x, 0.25x)
    - Encourages realistic appearance at multiple resolutions
    - Improves quality for large images

    Can be used as drop-in replacement for PatchGANDiscriminator.
    """

    def __init__(self, in_channels: int = 3, num_scales: int = 3):
        """
        Args:
            in_channels: Number of channels per image
            num_scales: Number of discriminators at different scales (typically 3)
        """
        super().__init__()

        self.num_scales = num_scales

        # Create discriminators for each scale
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels)
            for _ in range(num_scales)
        ])

        # Downsampling layer (for creating coarser scales)
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, condition, target):
        """
        Forward pass at multiple scales

        Args:
            condition: Input semantic map (B, 3, 256, 256)
            target: Real or generated image (B, 3, 256, 256)

        Returns:
            List of patch predictions at different scales:
            [
                (B, 1, 30, 30),   # Full resolution
                (B, 1, 14, 14),   # 0.5x resolution
                (B, 1, 6, 6)      # 0.25x resolution
            ]
        """
        outputs = []

        condition_scaled = condition
        target_scaled = target

        for i, discriminator in enumerate(self.discriminators):
            # Apply discriminator at current scale
            output = discriminator(condition_scaled, target_scaled)
            outputs.append(output)

            # Downsample for next scale (except for last scale)
            if i < self.num_scales - 1:
                condition_scaled = self.downsample(condition_scaled)
                target_scaled = self.downsample(target_scaled)

        return outputs


def test_discriminator():
    """Test the PatchGAN discriminator with dummy input"""
    print("Testing PatchGAN Discriminator...")

    # Create model
    discriminator = PatchGANDiscriminator(in_channels=3)

    # Count parameters
    num_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test with dummy input
    batch_size = 2
    condition = torch.randn(batch_size, 3, 256, 256)  # Input semantic map
    target = torch.randn(batch_size, 3, 256, 256)     # Real/fake image

    print(f"\nCondition shape: {condition.shape}")
    print(f"Target shape: {target.shape}")

    # Forward pass
    with torch.no_grad():
        output = discriminator(condition, target)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Verify output shape
    assert output.shape == (batch_size, 1, 30, 30), "Output shape mismatch"

    print("\n✓ Discriminator test passed!")

    return discriminator


def test_multiscale_discriminator():
    """Test the multi-scale discriminator"""
    print("\n" + "="*60)
    print("Testing Multi-Scale Discriminator...")

    # Create model
    discriminator = MultiScaleDiscriminator(in_channels=3, num_scales=3)

    # Count parameters
    num_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test with dummy input
    batch_size = 2
    condition = torch.randn(batch_size, 3, 256, 256)
    target = torch.randn(batch_size, 3, 256, 256)

    print(f"\nCondition shape: {condition.shape}")
    print(f"Target shape: {target.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = discriminator(condition, target)

    print(f"\nNumber of scales: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Scale {i} output shape: {output.shape}")

    # Verify outputs
    assert len(outputs) == 3, "Should have 3 scales"
    assert outputs[0].shape == (batch_size, 1, 30, 30), "Scale 0 shape mismatch"

    print("\n✓ Multi-scale discriminator test passed!")

    return discriminator


if __name__ == "__main__":
    test_discriminator()
    test_multiscale_discriminator()
