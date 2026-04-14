"""
Pix2Pix Model for Floor Plan Rendering

This module combines the U-Net generator and PatchGAN discriminator
into a complete Pix2Pix conditional GAN for transforming simple layout
sketches into realistic floor plan renderings.

Paper: "Image-to-Image Translation with Conditional Adversarial Networks"
       Isola et al., CVPR 2017
       https://arxiv.org/abs/1611.07004

Key Features:
- Conditional GAN: Generator conditioned on input layout
- U-Net architecture with skip connections
- PatchGAN discriminator for local realism
- Combined L1 + adversarial loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .generator_unet import UNetGenerator
from .discriminator import PatchGANDiscriminator, MultiScaleDiscriminator


class Pix2PixModel(nn.Module):
    """
    Complete Pix2Pix model for floor plan rendering

    Architecture:
        Generator: U-Net that transforms semantic map → realistic image
        Discriminator: PatchGAN that distinguishes real vs generated

    Loss Function:
        L_total = λ_GAN * L_GAN + λ_L1 * L_L1

        Where:
        - L_GAN: Adversarial loss (make generated images fool discriminator)
        - L_L1: Pixel-wise reconstruction loss (match ground truth)
        - λ_GAN = 1.0 (default)
        - λ_L1 = 100.0 (strong weight on reconstruction)

    The high L1 weight ensures generated images are structurally similar
    to ground truth, while GAN loss adds realistic details.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        use_multiscale: bool = False,
        lambda_l1: float = 100.0
    ):
        """
        Args:
            in_channels: Input image channels (3 for RGB semantic map)
            out_channels: Output image channels (3 for RGB rendered image)
            use_multiscale: Use multi-scale discriminator (slower but better quality)
            lambda_l1: Weight for L1 reconstruction loss (default: 100.0)
        """
        super().__init__()

        self.lambda_l1 = lambda_l1

        # Generator: semantic map → realistic rendering
        self.generator = UNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels
        )

        # Discriminator: classify real vs fake
        if use_multiscale:
            self.discriminator = MultiScaleDiscriminator(in_channels=in_channels)
        else:
            self.discriminator = PatchGANDiscriminator(in_channels=in_channels)

        self.use_multiscale = use_multiscale

        # Loss functions
        # BCEWithLogitsLoss = sigmoid + binary cross entropy
        # More numerically stable than separate sigmoid + BCE
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Generate realistic floor plan from semantic map

        Args:
            condition: Input semantic map (B, 3, 256, 256)

        Returns:
            Generated realistic image (B, 3, 256, 256)
        """
        return self.generator(condition)

    def compute_generator_loss(
        self,
        condition: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator losses

        Args:
            condition: Input semantic map (B, 3, 256, 256)
            target: Real realistic image (B, 3, 256, 256)

        Returns:
            Dictionary with loss components:
            - 'gen_gan': Adversarial loss (fool discriminator)
            - 'gen_l1': L1 reconstruction loss
            - 'gen_total': Total generator loss
        """
        # Generate fake image
        fake = self.generator(condition)

        # Adversarial loss: make discriminator classify fake as real
        if self.use_multiscale:
            # Multi-scale discriminator returns list of outputs
            pred_fake = self.discriminator(condition, fake)
            loss_gan = 0
            for pred in pred_fake:
                # Target: all ones (want discriminator to think it's real)
                target_real = torch.ones_like(pred)
                loss_gan += self.criterion_gan(pred, target_real)
            loss_gan /= len(pred_fake)  # Average across scales
        else:
            pred_fake = self.discriminator(condition, fake)
            target_real = torch.ones_like(pred_fake)
            loss_gan = self.criterion_gan(pred_fake, target_real)

        # L1 reconstruction loss: match ground truth pixel-wise
        loss_l1 = self.criterion_l1(fake, target)

        # Total generator loss
        loss_total = loss_gan + self.lambda_l1 * loss_l1

        return {
            'gen_gan': loss_gan,
            'gen_l1': loss_l1,
            'gen_total': loss_total,
            'fake_images': fake  # Return for visualization
        }

    def compute_discriminator_loss(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
        fake: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator losses

        Args:
            condition: Input semantic map (B, 3, 256, 256)
            target: Real realistic image (B, 3, 256, 256)
            fake: Pre-generated fake image (optional, will generate if not provided)

        Returns:
            Dictionary with loss components:
            - 'disc_real': Loss on real images (should classify as real)
            - 'disc_fake': Loss on fake images (should classify as fake)
            - 'disc_total': Total discriminator loss
        """
        # Generate fake image if not provided
        if fake is None:
            with torch.no_grad():  # Don't need gradients for generator here
                fake = self.generator(condition)

        if self.use_multiscale:
            # Multi-scale discriminator
            pred_real = self.discriminator(condition, target)
            pred_fake = self.discriminator(condition, fake.detach())

            loss_real = 0
            loss_fake = 0

            for pred_r, pred_f in zip(pred_real, pred_fake):
                # Real images should be classified as real (ones)
                target_real = torch.ones_like(pred_r)
                loss_real += self.criterion_gan(pred_r, target_real)

                # Fake images should be classified as fake (zeros)
                target_fake = torch.zeros_like(pred_f)
                loss_fake += self.criterion_gan(pred_f, target_fake)

            loss_real /= len(pred_real)
            loss_fake /= len(pred_fake)

        else:
            # Single-scale discriminator
            # Real images should be classified as real
            pred_real = self.discriminator(condition, target)
            target_real = torch.ones_like(pred_real)
            loss_real = self.criterion_gan(pred_real, target_real)

            # Fake images should be classified as fake
            pred_fake = self.discriminator(condition, fake.detach())
            target_fake = torch.zeros_like(pred_fake)
            loss_fake = self.criterion_gan(pred_fake, target_fake)

        # Total discriminator loss (average of real and fake)
        loss_total = (loss_real + loss_fake) * 0.5

        return {
            'disc_real': loss_real,
            'disc_fake': loss_fake,
            'disc_total': loss_total
        }

    def generate(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Generate realistic floor plan (inference mode)

        Args:
            condition: Input semantic map (B, 3, 256, 256)
                      Values should be in range [-1, 1]

        Returns:
            Generated realistic image (B, 3, 256, 256)
            Values in range [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            fake = self.generator(condition)
        return fake


class Pix2PixLoss(nn.Module):
    """
    Combined loss function for Pix2Pix training

    This wraps the generator and discriminator losses for convenient training.
    """

    def __init__(self, lambda_l1: float = 100.0):
        super().__init__()
        self.lambda_l1 = lambda_l1

    def forward(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        condition: torch.Tensor,
        target: torch.Tensor,
        mode: str = 'generator'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for generator or discriminator

        Args:
            generator: Generator network
            discriminator: Discriminator network
            condition: Input semantic map
            target: Real realistic image
            mode: 'generator' or 'discriminator'

        Returns:
            Dictionary of losses
        """
        if mode == 'generator':
            # Generate fake image
            fake = generator(condition)

            # Adversarial loss
            pred_fake = discriminator(condition, fake)
            target_real = torch.ones_like(pred_fake)
            loss_gan = F.binary_cross_entropy_with_logits(pred_fake, target_real)

            # L1 loss
            loss_l1 = F.l1_loss(fake, target)

            # Total loss
            loss_total = loss_gan + self.lambda_l1 * loss_l1

            return {
                'gen_gan': loss_gan,
                'gen_l1': loss_l1,
                'gen_total': loss_total,
                'fake_images': fake
            }

        elif mode == 'discriminator':
            # Generate fake image (detached)
            with torch.no_grad():
                fake = generator(condition)

            # Real loss
            pred_real = discriminator(condition, target)
            target_real = torch.ones_like(pred_real)
            loss_real = F.binary_cross_entropy_with_logits(pred_real, target_real)

            # Fake loss
            pred_fake = discriminator(condition, fake.detach())
            target_fake = torch.zeros_like(pred_fake)
            loss_fake = F.binary_cross_entropy_with_logits(pred_fake, target_fake)

            # Total loss
            loss_total = (loss_real + loss_fake) * 0.5

            return {
                'disc_real': loss_real,
                'disc_fake': loss_fake,
                'disc_total': loss_total
            }

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'generator' or 'discriminator'")


def test_pix2pix_model():
    """Test the complete Pix2Pix model"""
    print("Testing Pix2Pix Model...")

    # Create model
    model = Pix2PixModel(
        in_channels=3,
        out_channels=3,
        use_multiscale=False,
        lambda_l1=100.0
    )

    # Count parameters
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    total_params = gen_params + disc_params

    print(f"\nGenerator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {total_params:,}")

    # Test with dummy data
    batch_size = 2
    condition = torch.randn(batch_size, 3, 256, 256)
    target = torch.randn(batch_size, 3, 256, 256)

    print(f"\nInput shape: {condition.shape}")
    print(f"Target shape: {target.shape}")

    # Test generator forward pass
    with torch.no_grad():
        fake = model(condition)
    print(f"\nGenerated output shape: {fake.shape}")

    # Test generator loss
    gen_losses = model.compute_generator_loss(condition, target)
    print(f"\nGenerator losses:")
    for key, value in gen_losses.items():
        if key != 'fake_images':
            print(f"  {key}: {value.item():.4f}")

    # Test discriminator loss
    disc_losses = model.compute_discriminator_loss(condition, target)
    print(f"\nDiscriminator losses:")
    for key, value in disc_losses.items():
        print(f"  {key}: {value.item():.4f}")

    print("\n✓ Pix2Pix model test passed!")

    return model


if __name__ == "__main__":
    test_pix2pix_model()
