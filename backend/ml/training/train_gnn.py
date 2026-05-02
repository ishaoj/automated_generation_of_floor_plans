#!/usr/bin/env python3
"""
Train GNN Layout Refiner

Trains a Graph Attention Network to refine constraint-solver layouts
based on patterns learned from real floor plans (CubiCasa5k).

Usage:
    python backend/ml/training/train_gnn.py --data_dir data/cubicasa5k --epochs 50
"""

import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.cubicasa_loader import CubiCasaDataset, collate_fn
from ml.data.preprocessor import FloorPlanPreprocessor, GraphBatchCollator
from ml.data.augmentation import FloorPlanAugmentation, NoAugmentation
from ml.models.gnn_refiner import GraphAttentionRefiner
from ml.training.losses import LayoutRefinementLoss, VastuAwareLoss
from ml.training.metrics import LayoutMetrics


def _add_solver_noise(
    positions: torch.Tensor,
    pos_std: float = 0.08,
    size_std: float = 0.04,
) -> torch.Tensor:
    """
    Add noise to clean CubiCasa5k positions to simulate constraint-solver
    imprecision. The GNN is trained to recover the clean target from this
    noisy input (denoising objective).

    pos_std  — std of position jitter as fraction of plot dimension (~8%)
    size_std — std of size jitter (~4%)
    """
    noisy = positions.clone()
    device = positions.device

    pos_noise  = torch.randn(positions.shape[0], 2, device=device) * pos_std
    size_noise = torch.randn(positions.shape[0], 2, device=device) * size_std

    noisy[:, :2] = torch.clamp(positions[:, :2] + pos_noise,  0.0, 0.95)
    noisy[:, 2:] = torch.clamp(positions[:, 2:] + size_noise, 0.05, 1.0)
    return noisy


class GNNTrainer:
    """Trainer for GNN layout refiner"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        lr: float = 1e-3,
        use_vastu_loss: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        if use_vastu_loss:
            self.criterion = VastuAwareLoss()
        else:
            self.criterion = LayoutRefinementLoss()

        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Metrics tracker
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()

        total_losses = {
            'total': 0.0,
            'position': 0.0,
            'overlap': 0.0,
            'area': 0.0,
            'adjacency': 0.0,
        }
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # Move to device
            node_features = batch['node_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device)
            positions = batch['positions'].to(self.device)
            batch_idx = batch['batch'].to(self.device)

            # Denoising training: add solver-like noise to clean CubiCasa positions,
            # train GNN to recover the original clean target.
            target_positions = positions
            noisy_positions  = _add_solver_noise(positions)

            # Forward pass
            pred_positions = self.model(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                positions=noisy_positions,
                batch=batch_idx,
            )

            if isinstance(self.criterion, VastuAwareLoss):
                losses = self.criterion(
                    pred_positions, target_positions, node_features, edge_index, batch_idx
                )
            else:
                losses = self.criterion(
                    pred_positions, target_positions, edge_index, batch_idx
                )

            loss = losses['total']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track losses
            for key in total_losses.keys():
                if key in losses:
                    total_losses[key] += losses[key].item()

            n_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses

    def validate(self, epoch: int) -> dict:
        """Validate model"""
        self.model.eval()

        total_losses = {
            'total': 0.0,
            'position': 0.0,
            'overlap': 0.0,
            'area': 0.0,
            'adjacency': 0.0,
        }
        total_metrics = {
            'position_mse': 0.0,
            'overlap_pct': 0.0,
            'area_error': 0.0,
        }
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                node_features = batch['node_features'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_attr = batch['edge_attr'].to(self.device)
                positions = batch['positions'].to(self.device)
                batch_idx = batch['batch'].to(self.device)

                # Use a fixed noise level during validation for consistent metrics
                target_positions = positions
                noisy_positions  = _add_solver_noise(positions)

                # Forward pass
                pred_positions = self.model(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    positions=noisy_positions,
                    batch=batch_idx,
                )

                if isinstance(self.criterion, VastuAwareLoss):
                    losses = self.criterion(
                        pred_positions, target_positions, node_features, edge_index, batch_idx
                    )
                else:
                    losses = self.criterion(
                        pred_positions, target_positions, edge_index, batch_idx
                    )

                # Track losses
                for key in total_losses.keys():
                    if key in losses:
                        total_losses[key] += losses[key].item()

                # Compute metrics
                metrics = LayoutMetrics.compute_metrics(
                    pred_positions, target_positions, batch_idx
                )

                for key in total_metrics.keys():
                    if key in metrics:
                        total_metrics[key] += metrics[key]

                n_batches += 1

                # Update progress bar
                pbar.set_postfix({'loss': losses['total'].item()})

        # Average losses and metrics
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

        return {**avg_losses, **avg_metrics}

    def train(self, num_epochs: int, save_dir: str):
        """Full training loop"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nTraining for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Save directory: {save_path}\n")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)

            # Validate
            val_results = self.validate(epoch)
            self.val_losses.append(val_results)

            # Learning rate scheduling
            self.scheduler.step(val_results['total'])

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss:   {val_results['total']:.4f}")
            print(f"  Val Overlap: {val_results.get('overlap_pct', 0):.2f}%")
            print(f"  Val Position MSE: {val_results.get('position_mse', 0):.4f}")

            # Save best model
            if val_results['total'] < self.best_val_loss:
                self.best_val_loss = val_results['total']
                self.save_checkpoint(
                    save_path / 'best_model.pt',
                    epoch,
                    val_results['total']
                )
                print(f"  ✓ Saved best model (loss: {self.best_val_loss:.4f})")

            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(
                    save_path / f'checkpoint_epoch_{epoch}.pt',
                    epoch,
                    val_results['total']
                )

        # Save final model
        self.save_checkpoint(
            save_path / 'final_model.pt',
            num_epochs,
            val_results['total']
        )

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n✓ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }, path)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    augment: bool = True,
) -> tuple:
    """Create train and validation dataloaders"""

    # Preprocessor
    preprocessor = FloorPlanPreprocessor(normalize_coords=True)

    # Augmentation
    if augment:
        augmentation = FloorPlanAugmentation(
            flip_prob=0.5,
            rotate_prob=0.5,
            jitter_prob=0.3,
        )
    else:
        augmentation = NoAugmentation()

    # Create datasets
    print("Loading training data...")
    train_dataset = CubiCasaDataset(
        data_dir=data_dir,
        split='train',
        include_images=False,
        filter_non_residential=True,
        min_rooms=2,
        max_rooms=10,
    )

    print("Loading validation data...")
    val_dataset = CubiCasaDataset(
        data_dir=data_dir,
        split='val',
        include_images=False,
        filter_non_residential=True,
        min_rooms=2,
        max_rooms=10,
    )

    # Custom collate function
    def collate_and_preprocess(batch):
        """Collate and preprocess batch"""
        # Preprocess each sample
        processed = []
        for sample in batch:
            graph = preprocessor.preprocess(sample)

            # Apply augmentation (only to training data)
            if augment:
                graph = augmentation(graph)

            processed.append(graph)

        # Batch graphs
        collator = GraphBatchCollator(preprocessor)
        return collator(processed)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_and_preprocess,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_and_preprocess,
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train GNN Layout Refiner")

    # Data
    parser.add_argument('--data_dir', type=str, default='data/cubicasa5k',
                        help='Path to CubiCasa5k dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GAT layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cpu, cuda, or auto')
    parser.add_argument('--use_vastu_loss', action='store_true',
                        help='Use Vastu-aware loss function')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')

    # Output
    parser.add_argument('--save_dir', type=str, default='ml/models/saved',
                        help='Directory to save trained models')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    # node_features_dim = 11 room-type one-hot + 1 area + 4 normalised (x,y,w,h) = 16
    model = GraphAttentionRefiner(
        node_features_dim=16,
        edge_features_dim=4,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    # Create trainer
    trainer = GNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        use_vastu_loss=args.use_vastu_loss,
    )

    # Train
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
