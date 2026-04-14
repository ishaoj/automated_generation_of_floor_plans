"""
Visualization Utilities for ML Training

This module provides utilities for visualizing:
- Floor plan layouts
- Training progress
- Model predictions
- Comparison grids
"""

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from PIL import Image


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1]

    Args:
        tensor: Normalized tensor in range [-1, 1]

    Returns:
        Denormalized tensor in range [0, 1]
    """
    return (tensor + 1.0) / 2.0


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image

    Args:
        tensor: Image tensor (C, H, W) in range [-1, 1] or [0, 1]

    Returns:
        Numpy array (H, W, C) in range [0, 255]
    """
    # Denormalize if needed
    if tensor.min() < 0:
        tensor = denormalize(tensor)

    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy
    img = tensor.cpu().numpy()

    # Transpose from (C, H, W) to (H, W, C)
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    # Scale to [0, 255]
    img = (img * 255).astype(np.uint8)

    return img


def create_comparison_grid(
    condition: torch.Tensor,
    generated: torch.Tensor,
    target: torch.Tensor,
    max_images: int = 4
) -> torch.Tensor:
    """
    Create comparison grid showing input, generated, and target images

    Args:
        condition: Input semantic maps (B, 3, H, W)
        generated: Generated images (B, 3, H, W)
        target: Target real images (B, 3, H, W)
        max_images: Maximum number of images to show

    Returns:
        Grid tensor suitable for visualization
    """
    # Limit number of images
    n = min(max_images, condition.size(0))
    condition = condition[:n]
    generated = generated[:n]
    target = target[:n]

    # Denormalize all tensors
    condition = denormalize(condition)
    generated = denormalize(generated)
    target = denormalize(target)

    # Create grid: each row shows [input, generated, target]
    all_images = []
    for i in range(n):
        all_images.extend([condition[i], generated[i], target[i]])

    # Stack into grid
    grid = make_grid(
        all_images,
        nrow=3,  # 3 columns: input, generated, target
        padding=2,
        normalize=False,
        value_range=(0, 1)
    )

    return grid


def create_training_visualization(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Create training/validation loss curves

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curve to {save_path}")

    plt.close()


def create_gan_loss_visualization(
    gen_losses: List[float],
    disc_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Create GAN training curves (generator and discriminator)

    Args:
        gen_losses: List of generator losses per epoch
        disc_losses: List of discriminator losses per epoch
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(gen_losses) + 1)

    plt.plot(epochs, gen_losses, 'g-', label='Generator Loss', linewidth=2)
    plt.plot(epochs, disc_losses, 'orange', linestyle='--', label='Discriminator Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GAN Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved GAN loss curve to {save_path}")

    plt.close()


def visualize_layout_graph(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    positions: torch.Tensor,
    room_types: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize floor plan as a graph

    Args:
        node_features: Node feature tensor (N, F)
        edge_index: Edge connectivity (2, E)
        positions: Room positions (N, 4) - [x, y, w, h]
        room_types: List of room type names
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 10))

    # Convert to numpy
    positions = positions.cpu().numpy()
    edge_index = edge_index.cpu().numpy()

    # Draw rooms as rectangles
    for i, (x, y, w, h) in enumerate(positions):
        rect = plt.Rectangle(
            (x, y), w, h,
            fill=True, alpha=0.3, edgecolor='black', linewidth=2
        )
        plt.gca().add_patch(rect)

        # Add room label at center
        cx, cy = x + w/2, y + h/2
        room_type = room_types[i] if i < len(room_types) else f"Room {i}"
        plt.text(cx, cy, room_type, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        x1, y1 = positions[src, 0] + positions[src, 2]/2, positions[src, 1] + positions[src, 3]/2
        x2, y2 = positions[dst, 0] + positions[dst, 2]/2, positions[dst, 1] + positions[dst, 3]/2

        plt.plot([x1, x2], [y1, y2], 'r--', alpha=0.5, linewidth=1)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.title('Floor Plan Layout Graph', fontsize=14, fontweight='bold')
    plt.xlabel('X Position (normalized)', fontsize=11)
    plt.ylabel('Y Position (normalized)', fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved graph visualization to {save_path}")

    plt.close()


def create_side_by_side_comparison(
    images: List[torch.Tensor],
    titles: List[str],
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Create side-by-side comparison of multiple images

    Args:
        images: List of image tensors (each is (3, H, W))
        titles: List of titles for each image
        save_path: Path to save figure (optional)

    Returns:
        PIL Image of the comparison
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))

    if n == 1:
        axes = [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        # Convert tensor to numpy
        img_np = tensor_to_image(img)

        axes[i].imshow(img_np)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    # Convert to PIL Image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    pil_img = Image.fromarray(img_array)

    plt.close()

    return pil_img


def plot_metric_comparison(
    metrics_dict: dict,
    save_path: Optional[str] = None
):
    """
    Plot comparison of multiple metrics over epochs

    Args:
        metrics_dict: Dictionary mapping metric names to lists of values
                     e.g., {'FID': [50, 45, 40], 'LPIPS': [0.5, 0.4, 0.35]}
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(12, 6))

    for metric_name, values in metrics_dict.items():
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, marker='o', label=metric_name, linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Metric Comparison Over Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metric comparison to {save_path}")

    plt.close()


def save_model_predictions(
    model,
    dataloader,
    device: str,
    num_samples: int = 8,
    save_dir: str = 'output/predictions'
):
    """
    Save model predictions on validation data

    Args:
        model: Trained model
        dataloader: DataLoader with validation data
        device: Device to run inference on
        num_samples: Number of samples to save
        save_dir: Directory to save predictions
    """
    import os
    from pathlib import Path

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    samples_saved = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_saved >= num_samples:
                break

            condition = batch['semantic_map'].to(device)
            target = batch['rendered_image'].to(device)

            # Generate predictions
            generated = model.generate(condition)

            # Save each sample
            batch_size = condition.size(0)
            for i in range(min(batch_size, num_samples - samples_saved)):
                # Create comparison
                comparison = create_side_by_side_comparison(
                    images=[condition[i].cpu(), generated[i].cpu(), target[i].cpu()],
                    titles=['Input', 'Generated', 'Target'],
                    save_path=save_dir / f'prediction_{samples_saved:03d}.png'
                )

                samples_saved += 1

    print(f"Saved {samples_saved} predictions to {save_dir}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")

    # Test denormalization
    tensor = torch.randn(3, 256, 256)
    denorm = denormalize(tensor)
    print(f"Original range: [{tensor.min():.2f}, {tensor.max():.2f}]")
    print(f"Denormalized range: [{denorm.min():.2f}, {denorm.max():.2f}]")

    # Test comparison grid
    condition = torch.randn(2, 3, 256, 256)
    generated = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)

    grid = create_comparison_grid(condition, generated, target)
    print(f"\nComparison grid shape: {grid.shape}")

    print("\n✓ Visualization utilities test passed!")
