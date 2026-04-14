"""
Pix2Pix Renderer Inference Wrapper

This module provides a production-ready inference wrapper for the Pix2Pix
floor plan renderer. It handles model loading, preprocessing, postprocessing,
and graceful fallback to the default renderer.

Usage:
    from ml.inference_renderer import Pix2PixRendererInference

    renderer = Pix2PixRendererInference(
        model_path='ml/models/saved/pix2pix/best_model.pt',
        device='cpu',
        enable=True
    )

    # Generate realistic floor plan from layout
    semantic_map = create_semantic_map(layout)  # PIL Image or tensor
    rendered_image = renderer.render(semantic_map)
"""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np

import torch
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class Pix2PixRendererInference:
    """
    Inference wrapper for Pix2Pix floor plan renderer

    This class provides a simple interface for rendering realistic
    floor plans from semantic layouts using a trained Pix2Pix model.

    Features:
    - Automatic model loading and caching
    - Preprocessing and postprocessing
    - Graceful error handling with fallback
    - Both PIL Image and tensor input/output support
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        enable: bool = True,
        image_size: int = 256
    ):
        """
        Initialize the renderer

        Args:
            model_path: Path to trained Pix2Pix checkpoint
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            enable: Whether to enable ML rendering (if False, acts as passthrough)
            image_size: Input/output image size (default: 256x256)
        """
        self.model_path = model_path
        self.device = device
        self.enable = enable
        self.image_size = image_size
        self.model = None

        # Initialize model if enabled
        if self.enable and self.model_path:
            self._load_model()
        elif self.enable and not self.model_path:
            logger.warning(
                "Pix2Pix renderer enabled but no model_path provided. "
                "Rendering will fall back to default."
            )
            self.enable = False

        # Image transforms
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])

        self.to_pil = transforms.ToPILImage()

    def _load_model(self):
        """Load the Pix2Pix model from checkpoint"""
        try:
            from ml.models.pix2pix_renderer import Pix2PixModel

            logger.info(f"Loading Pix2Pix model from {self.model_path}...")

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract config
            config = checkpoint.get('config', {})
            lambda_l1 = config.get('lambda_l1', 100.0)
            use_multiscale = config.get('use_multiscale', False)

            # Create model
            self.model = Pix2PixModel(
                in_channels=3,
                out_channels=3,
                use_multiscale=use_multiscale,
                lambda_l1=lambda_l1
            )

            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Successfully loaded Pix2Pix model (epoch {checkpoint.get('epoch', '?')})")

        except Exception as e:
            logger.error(f"Failed to load Pix2Pix model: {e}")
            logger.warning("Disabling ML rendering, will use default renderer")
            self.enable = False
            self.model = None

    def render(
        self,
        semantic_map: Union[Image.Image, torch.Tensor, np.ndarray],
        return_tensor: bool = False
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Render realistic floor plan from semantic map

        Args:
            semantic_map: Input semantic map
                         - PIL Image (RGB)
                         - Tensor (C, H, W) in [0, 1] or [-1, 1]
                         - NumPy array (H, W, C) in [0, 255]
            return_tensor: If True, return tensor instead of PIL Image

        Returns:
            Rendered floor plan as PIL Image or tensor
        """
        # If ML rendering is disabled, return input as-is
        if not self.enable or self.model is None:
            logger.debug("ML rendering disabled, returning input")
            if return_tensor and isinstance(semantic_map, Image.Image):
                return self.to_tensor(semantic_map).unsqueeze(0)
            elif not return_tensor and isinstance(semantic_map, torch.Tensor):
                return self.to_pil(semantic_map)
            return semantic_map

        try:
            # Preprocess input
            input_tensor = self._preprocess(semantic_map)

            # Run inference
            with torch.no_grad():
                output_tensor = self.model.generate(input_tensor)

            # Postprocess output
            if return_tensor:
                return output_tensor
            else:
                return self._postprocess(output_tensor)

        except Exception as e:
            logger.error(f"Error during Pix2Pix rendering: {e}")
            logger.warning("Falling back to input image")

            # Fallback: return input
            if return_tensor and isinstance(semantic_map, Image.Image):
                return self.to_tensor(semantic_map).unsqueeze(0)
            elif not return_tensor and isinstance(semantic_map, torch.Tensor):
                return self.to_pil(semantic_map.squeeze(0) if semantic_map.dim() == 4 else semantic_map)
            return semantic_map

    def _preprocess(
        self,
        semantic_map: Union[Image.Image, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess input to model format

        Args:
            semantic_map: Input in various formats

        Returns:
            Tensor (1, 3, H, W) in range [-1, 1]
        """
        # Convert to PIL Image if needed
        if isinstance(semantic_map, np.ndarray):
            # NumPy array (H, W, C) in [0, 255]
            semantic_map = Image.fromarray(semantic_map.astype(np.uint8))

        if isinstance(semantic_map, Image.Image):
            # PIL Image → Tensor
            tensor = self.to_tensor(semantic_map)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
        elif isinstance(semantic_map, torch.Tensor):
            # Already tensor
            tensor = semantic_map

            # Add batch dimension if needed
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            # Resize if needed
            if tensor.size(2) != self.image_size or tensor.size(3) != self.image_size:
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                )

            # Normalize to [-1, 1] if in [0, 1]
            if tensor.min() >= 0 and tensor.max() <= 1:
                tensor = tensor * 2 - 1
        else:
            raise ValueError(f"Unsupported input type: {type(semantic_map)}")

        return tensor.to(self.device)

    def _postprocess(self, output_tensor: torch.Tensor) -> Image.Image:
        """
        Postprocess model output to PIL Image

        Args:
            output_tensor: Model output (1, 3, H, W) in [-1, 1]

        Returns:
            PIL Image
        """
        # Remove batch dimension
        if output_tensor.dim() == 4:
            output_tensor = output_tensor.squeeze(0)

        # Denormalize from [-1, 1] to [0, 1]
        output_tensor = (output_tensor + 1) / 2

        # Clamp to [0, 1]
        output_tensor = torch.clamp(output_tensor, 0, 1)

        # Convert to PIL Image
        output_tensor = output_tensor.cpu()
        image = self.to_pil(output_tensor)

        return image

    def render_batch(
        self,
        semantic_maps: torch.Tensor,
        return_tensor: bool = False
    ) -> Union[list[Image.Image], torch.Tensor]:
        """
        Render multiple floor plans in a batch

        Args:
            semantic_maps: Batch of semantic maps (B, 3, H, W)
            return_tensor: If True, return tensor instead of list of images

        Returns:
            List of PIL Images or tensor (B, 3, H, W)
        """
        if not self.enable or self.model is None:
            logger.debug("ML rendering disabled")
            if return_tensor:
                return semantic_maps
            else:
                return [self.to_pil(img) for img in semantic_maps]

        try:
            # Preprocess
            input_tensor = self._preprocess(semantic_maps)

            # Run inference
            with torch.no_grad():
                output_tensor = self.model.generate(input_tensor)

            # Postprocess
            if return_tensor:
                return output_tensor
            else:
                images = []
                for i in range(output_tensor.size(0)):
                    img = self._postprocess(output_tensor[i:i+1])
                    images.append(img)
                return images

        except Exception as e:
            logger.error(f"Error during batch rendering: {e}")
            logger.warning("Falling back to input images")

            if return_tensor:
                return semantic_maps
            else:
                return [self.to_pil(img) for img in semantic_maps]

    def is_available(self) -> bool:
        """Check if ML renderer is available and loaded"""
        return self.enable and self.model is not None


def create_semantic_map_from_layout(layout: dict, image_size: int = 256) -> Image.Image:
    """
    Create a semantic segmentation map from a floor plan layout

    This function converts the layout dictionary (with room positions and types)
    into a colored semantic map suitable for Pix2Pix rendering.

    Args:
        layout: Layout dictionary with 'rooms' key containing room data
                Each room should have: position, dimensions, room_type
        image_size: Output image size (default: 256x256)

    Returns:
        PIL Image with colored semantic map
    """
    from PIL import ImageDraw

    # Color mapping for different room types
    ROOM_COLORS = {
        'bedroom': (135, 206, 250),      # Light blue
        'living_room': (255, 218, 185),  # Peach
        'kitchen': (255, 160, 122),      # Light salmon
        'bathroom': (176, 224, 230),     # Powder blue
        'dining_room': (255, 228, 196),  # Bisque
        'puja_room': (255, 215, 0),      # Gold
        'balcony': (152, 251, 152),      # Pale green
        'study': (221, 160, 221),        # Plum
        'storage': (211, 211, 211),      # Light gray
        'entrance': (255, 255, 224),     # Light yellow
        'corridor': (245, 245, 220),     # Beige
    }

    # Create blank white image
    img = Image.new('RGB', (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Get plot dimensions from layout
    plot_width = layout.get('plot_width', 50.0)
    plot_height = layout.get('plot_height', 50.0)

    # Draw each room
    rooms = layout.get('rooms', [])
    for room in rooms:
        # Get room data
        x = room['position']['x']
        y = room['position']['y']
        w = room['dimensions']['width']
        h = room['dimensions']['height']
        room_type = room.get('room_type', 'bedroom')

        # Convert to pixel coordinates
        x_px = int((x / plot_width) * image_size)
        y_px = int((y / plot_height) * image_size)
        w_px = int((w / plot_width) * image_size)
        h_px = int((h / plot_height) * image_size)

        # Get room color
        color = ROOM_COLORS.get(room_type, (200, 200, 200))

        # Draw filled rectangle
        draw.rectangle(
            [x_px, y_px, x_px + w_px, y_px + h_px],
            fill=color,
            outline=(0, 0, 0),
            width=2
        )

    return img


# Example usage
if __name__ == "__main__":
    # Test the renderer
    print("Testing Pix2Pix Renderer Inference...")

    # Create renderer (will fail gracefully if model not available)
    renderer = Pix2PixRendererInference(
        model_path='ml/models/saved/pix2pix/best_model.pt',
        device='cpu',
        enable=True
    )

    print(f"Renderer available: {renderer.is_available()}")

    # Test with dummy image
    dummy_image = Image.new('RGB', (256, 256), (255, 200, 200))

    output = renderer.render(dummy_image)
    print(f"Output type: {type(output)}")

    print("\n✓ Pix2Pix renderer inference test complete!")
