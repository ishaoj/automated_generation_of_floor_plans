"""
Model checkpointing utilities
"""

import torch
from pathlib import Path
from typing import Dict, Optional
import shutil


class ModelCheckpointer:
    """
    Handles model checkpointing during training

    Features:
    - Save best model based on validation metric
    - Save periodic checkpoints
    - Keep only N best checkpoints
    """

    def __init__(
        self,
        save_dir: str,
        keep_n_best: int = 3,
        metric_name: str = 'val_loss',
        mode: str = 'min',
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            keep_n_best: Number of best checkpoints to keep
            metric_name: Metric to track for best model
            mode: 'min' or 'max' (lower/higher is better)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_n_best = keep_n_best
        self.metric_name = metric_name
        self.mode = mode

        self.checkpoints = []  # List of (metric_value, path)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        extra_data: Optional[Dict] = None,
    ) -> Path:
        """
        Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dict of metric values
            scheduler: LR scheduler (optional)
            extra_data: Additional data to save (optional)

        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_data is not None:
            checkpoint.update(extra_data)

        # Save checkpoint
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = self.save_dir / checkpoint_name

        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint
        metric_value = metrics.get(self.metric_name, float('inf'))
        self.checkpoints.append((metric_value, checkpoint_path))

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        return checkpoint_path

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Optional[Path]:
        """
        Save checkpoint if it's the best so far

        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        metric_value = metrics.get(self.metric_name, float('inf'))

        # Check if this is the best
        if not self._is_best(metric_value):
            return None

        # Save as best model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        best_path = self.save_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)

        print(f"✓ Saved best model (epoch {epoch}, {self.metric_name}={metric_value:.4f})")

        return best_path

    def _is_best(self, metric_value: float) -> bool:
        """Check if metric value is the best so far"""
        if not self.checkpoints:
            return True

        current_best = min if self.mode == 'min' else max
        best_value = current_best(v for v, _ in self.checkpoints)

        if self.mode == 'min':
            return metric_value < best_value
        else:
            return metric_value > best_value

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only N best"""
        if len(self.checkpoints) <= self.keep_n_best:
            return

        # Sort by metric (best first)
        reverse = (self.mode == 'max')
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Remove worst checkpoints
        to_remove = self.checkpoints[self.keep_n_best:]
        self.checkpoints = self.checkpoints[:self.keep_n_best]

        for _, path in to_remove:
            if path.exists() and 'best' not in path.name:
                path.unlink()

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
    ) -> Dict:
        """
        Load checkpoint

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Path to checkpoint (if None, loads best)
            device: Device to map tensors to

        Returns:
            Checkpoint dict with metadata
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / 'best_model.pt'
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"✓ Loaded checkpoint from {checkpoint_path}")

        return checkpoint


def save_model_for_production(
    model: torch.nn.Module,
    save_path: str,
    metadata: Optional[Dict] = None,
):
    """
    Save model in production-ready format

    Args:
        model: Trained model
        save_path: Path to save model
        metadata: Optional metadata (training config, metrics, etc.)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if metadata is not None:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, save_path)
    print(f"✓ Saved production model to {save_path}")
