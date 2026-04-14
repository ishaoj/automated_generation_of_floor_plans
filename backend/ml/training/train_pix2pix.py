"""
Training Script for Pix2Pix Floor Plan Renderer

This script trains the Pix2Pix model to transform simple semantic layouts
into realistic, professional-quality floor plan renderings.

Training Process:
1. Load CubiCasa5k dataset (preprocessed semantic maps + real renderings)
2. Alternately train discriminator and generator
3. Validate on held-out test set
4. Save checkpoints and visualizations
5. Track metrics (FID, LPIPS, L1 error)

Usage:
    python backend/ml/training/train_pix2pix.py --config configs/pix2pix_config.yaml

    Or with command-line arguments:
    python backend/ml/training/train_pix2pix.py \
        --data_dir data/cubicasa5k \
        --output_dir ml/models/saved/pix2pix \
        --epochs 100 \
        --batch_size 4 \
        --lr 0.0002
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.pix2pix_renderer import Pix2PixModel
from ml.data.cubicasa_loader import CubiCasaDataset
from ml.training.metrics import compute_fid, compute_lpips, compute_ssim
from ml.utils.visualization import create_comparison_grid


class Pix2PixTrainer:
    """
    Trainer for Pix2Pix floor plan renderer

    Handles:
    - Alternating discriminator and generator training
    - Learning rate scheduling
    - Checkpointing and resuming
    - Validation and metric computation
    - Visualization and logging
    """

    def __init__(
        self,
        model: Pix2PixModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cpu',
        output_dir: str = 'ml/models/saved/pix2pix',
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizers
        # Use separate optimizers for generator and discriminator
        # Adam with beta1=0.5 is standard for GANs
        self.optimizer_g = optim.Adam(
            self.model.generator.parameters(),
            lr=config['learning_rate'],
            betas=(0.5, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config['learning_rate'],
            betas=(0.5, 0.999)
        )

        # Learning rate schedulers (optional)
        self.scheduler_g = optim.lr_scheduler.StepLR(
            self.optimizer_g,
            step_size=config.get('lr_decay_epochs', 50),
            gamma=config.get('lr_decay_gamma', 0.5)
        )
        self.scheduler_d = optim.lr_scheduler.StepLR(
            self.optimizer_d,
            step_size=config.get('lr_decay_epochs', 50),
            gamma=config.get('lr_decay_gamma', 0.5)
        )

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Wandb logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="vastu-ai-pix2pix",
                config=config,
                name=config.get('run_name', 'pix2pix_training')
            )

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        epoch_losses = {
            'gen_gan': 0.0,
            'gen_l1': 0.0,
            'gen_total': 0.0,
            'disc_real': 0.0,
            'disc_fake': 0.0,
            'disc_total': 0.0
        }

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config['epochs']}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            condition = batch['semantic_map'].to(self.device)  # Input layout
            target = batch['rendered_image'].to(self.device)   # Real floor plan

            # ========================================
            # Train Discriminator
            # ========================================
            self.optimizer_d.zero_grad()

            # Compute discriminator loss
            disc_losses = self.model.compute_discriminator_loss(
                condition, target
            )
            disc_loss = disc_losses['disc_total']

            # Backward pass
            disc_loss.backward()
            self.optimizer_d.step()

            # ========================================
            # Train Generator
            # ========================================
            self.optimizer_g.zero_grad()

            # Compute generator loss
            gen_losses = self.model.compute_generator_loss(
                condition, target
            )
            gen_loss = gen_losses['gen_total']

            # Backward pass
            gen_loss.backward()
            self.optimizer_g.step()

            # ========================================
            # Update statistics
            # ========================================
            for key in epoch_losses.keys():
                if key in gen_losses:
                    epoch_losses[key] += gen_losses[key].item()
                elif key in disc_losses:
                    epoch_losses[key] += disc_losses[key].item()

            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f"{gen_loss.item():.4f}",
                'D_loss': f"{disc_loss.item():.4f}",
                'L1': f"{gen_losses['gen_l1'].item():.4f}"
            })

            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/gen_gan': gen_losses['gen_gan'].item(),
                    'train/gen_l1': gen_losses['gen_l1'].item(),
                    'train/gen_total': gen_loss.item(),
                    'train/disc_real': disc_losses['disc_real'].item(),
                    'train/disc_fake': disc_losses['disc_fake'].item(),
                    'train/disc_total': disc_loss.item(),
                    'global_step': self.global_step
                })

            self.global_step += 1

        # Average losses over epoch
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()

        val_losses = {
            'gen_gan': 0.0,
            'gen_l1': 0.0,
            'gen_total': 0.0,
            'disc_real': 0.0,
            'disc_fake': 0.0,
            'disc_total': 0.0
        }

        all_fake_images = []
        all_real_images = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            condition = batch['semantic_map'].to(self.device)
            target = batch['rendered_image'].to(self.device)

            # Compute losses
            gen_losses = self.model.compute_generator_loss(condition, target)
            disc_losses = self.model.compute_discriminator_loss(condition, target)

            # Accumulate losses
            for key in val_losses.keys():
                if key in gen_losses:
                    val_losses[key] += gen_losses[key].item()
                elif key in disc_losses:
                    val_losses[key] += disc_losses[key].item()

            # Collect images for metrics
            fake_images = gen_losses['fake_images']
            all_fake_images.append(fake_images.cpu())
            all_real_images.append(target.cpu())

        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        # Compute additional metrics (FID, LPIPS, SSIM)
        # Note: These are expensive, so we compute them less frequently
        if self.epoch % self.config.get('metric_frequency', 5) == 0:
            fake_images = torch.cat(all_fake_images, dim=0)
            real_images = torch.cat(all_real_images, dim=0)

            # FID score (lower is better)
            # fid_score = compute_fid(real_images, fake_images)
            # val_losses['fid'] = fid_score

            # LPIPS (lower is better)
            # lpips_score = compute_lpips(real_images, fake_images)
            # val_losses['lpips'] = lpips_score

            # SSIM (higher is better)
            ssim_score = compute_ssim(real_images, fake_images)
            val_losses['ssim'] = ssim_score

        return val_losses

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Save epoch checkpoint
        if (self.epoch + 1) % self.config.get('save_frequency', 10) == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{self.epoch + 1}.pt'
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch + 1}")

    def save_sample_images(self, num_samples: int = 4):
        """Save sample generated images for visualization"""
        self.model.eval()

        # Get a batch from validation set
        batch = next(iter(self.val_loader))
        condition = batch['semantic_map'][:num_samples].to(self.device)
        target = batch['rendered_image'][:num_samples].to(self.device)

        # Generate fake images
        with torch.no_grad():
            fake = self.model.generate(condition)

        # Create comparison grid
        comparison = create_comparison_grid(condition, fake, target)

        # Save image
        output_path = self.output_dir / f'samples_epoch_{self.epoch + 1}.png'
        save_image(comparison, output_path, normalize=True)

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'samples': wandb.Image(comparison),
                'epoch': self.epoch + 1
            })

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(self.epoch, self.config['epochs']):
            self.epoch = epoch

            # Train for one epoch
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']} Summary:")
            print(f"  Train - G_total: {train_losses['gen_total']:.4f}, "
                  f"D_total: {train_losses['disc_total']:.4f}, "
                  f"L1: {train_losses['gen_l1']:.4f}")
            print(f"  Val   - G_total: {val_losses['gen_total']:.4f}, "
                  f"D_total: {val_losses['disc_total']:.4f}, "
                  f"L1: {val_losses['gen_l1']:.4f}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_gen_total': train_losses['gen_total'],
                    'train/epoch_disc_total': train_losses['disc_total'],
                    'val/gen_total': val_losses['gen_total'],
                    'val/disc_total': val_losses['disc_total'],
                    'val/gen_l1': val_losses['gen_l1'],
                    'learning_rate': self.optimizer_g.param_groups[0]['lr']
                })

            # Save sample images
            if (epoch + 1) % self.config.get('sample_frequency', 5) == 0:
                self.save_sample_images()

            # Save checkpoint
            is_best = val_losses['gen_total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['gen_total']

            self.save_checkpoint(is_best=is_best)

            # Update learning rate
            self.scheduler_g.step()
            self.scheduler_d.step()

        print("\nTraining completed!")
        if self.use_wandb:
            wandb.finish()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix floor plan renderer")
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='data/cubicasa5k')
    parser.add_argument('--output_dir', type=str, default='ml/models/saved/pix2pix')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--use_multiscale', action='store_true')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Load config or use command-line arguments
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'lambda_l1': args.lambda_l1,
            'use_multiscale': args.use_multiscale,
            'lr_decay_epochs': 50,
            'lr_decay_gamma': 0.5,
            'save_frequency': 10,
            'sample_frequency': 5,
            'metric_frequency': 5
        }

    # Create datasets
    print("Loading datasets...")
    train_dataset = CubiCasaDataset(
        data_dir=config['data_dir'],
        split='train',
        transform=None  # Add transforms if needed
    )
    val_dataset = CubiCasaDataset(
        data_dir=config['data_dir'],
        split='val',
        transform=None
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    print("\nCreating model...")
    model = Pix2PixModel(
        in_channels=3,
        out_channels=3,
        use_multiscale=config['use_multiscale'],
        lambda_l1=config['lambda_l1']
    )

    # Count parameters
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")

    # Create trainer
    trainer = Pix2PixTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        output_dir=args.output_dir,
        use_wandb=args.wandb
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
