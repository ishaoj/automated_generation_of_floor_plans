"""
Production inference wrappers for ML models

Provides easy-to-use interfaces for GNN refiner and visual renderer.
"""

import torch
from typing import Dict, Optional, List
from pathlib import Path


class GNNRefinerInference:
    """
    Production-ready GNN layout refiner

    Usage:
        refiner = GNNRefinerInference(model_path='ml/models/saved/best_model.pt')
        refined_layout = refiner.refine(layout, request)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        enable: bool = True,
    ):
        """
        Args:
            model_path: Path to trained model weights
            device: Device to run on ('cpu' or 'cuda')
            enable: Whether to enable refinement (if False, returns input unchanged)
        """
        self.enable = enable
        self.device = torch.device(device)
        self.model = None
        self.encoder = None

        if enable and model_path and Path(model_path).exists():
            try:
                self._load_model(model_path)
                print(f"✓ Loaded GNN refiner from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load GNN model: {e}")
                print("Falling back to constraint-only mode")
                self.enable = False
        else:
            if enable:
                print("Warning: GNN model not found. Using constraint-only mode.")
            self.enable = False

    def _load_model(self, model_path: str):
        """Load trained model"""
        from ml.models.gnn_refiner import GraphAttentionRefiner
        from ml.models.graph_encoder import GraphEncoder

        # Initialize model
        # node_features_dim=12: 11 room-type one-hot + 1 area (matches training)
        self.model = GraphAttentionRefiner(
            node_features_dim=12,
            edge_features_dim=4,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
            dropout=0.0,  # No dropout at inference
        ).to(self.device)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Initialize encoder
        self.encoder = GraphEncoder(normalize_coords=True)

    def refine(
        self,
        layout: Dict,
        request: Optional[Dict] = None,
        max_adjustment: float = 0.15,
        min_room_size: float = 0.05,
    ) -> Dict:
        """
        Refine layout using GNN

        Args:
            layout: Layout dict from constraint solver
            request: Original FloorPlanRequest (optional)
            max_adjustment: Maximum position adjustment (fraction)
            min_room_size: Minimum room size (fraction)

        Returns:
            Refined layout dict (or original if disabled/failed)
        """
        # If disabled or not loaded, return original
        if not self.enable or self.model is None:
            return layout

        try:
            # Encode layout as graph
            graph = self.encoder.encode_layout(layout)

            # Move to device
            # Trained model uses 12 features (11 room-type + 1 area); strip zone encoding
            node_features = graph['node_features'][:, :12].to(self.device)
            edge_index = graph['edge_index'].to(self.device)
            edge_attr = graph['edge_attr'].to(self.device)
            positions = graph['positions'].to(self.device)

            # Run refinement
            with torch.no_grad():
                refined_positions = self.model.refine_with_constraints(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    positions=positions,
                    min_room_size=min_room_size,
                    max_adjustment=max_adjustment,
                )

            # Decode back to layout
            refined_layout = self.encoder.decode_graph(
                node_features=node_features.cpu(),
                positions=refined_positions.cpu(),
                plot_dims=graph['plot_dims'],
                original_layout=layout,
            )

            return refined_layout

        except Exception as e:
            print(f"Warning: GNN refinement failed: {e}")
            print("Returning original layout")
            return layout

    def batch_refine(self, layouts: List[Dict]) -> List[Dict]:
        """
        Refine multiple layouts

        Args:
            layouts: List of layout dicts

        Returns:
            List of refined layouts
        """
        if not self.enable:
            return layouts

        refined = []
        for layout in layouts:
            refined.append(self.refine(layout))

        return refined


class VisualRendererInference:
    """
    Production-ready visual renderer (Pix2Pix or Diffusion)

    Generates photorealistic floor plan images from layouts.

    Note: This is a placeholder. Full implementation requires
    training the Pix2Pix or Diffusion model first.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        enable: bool = False,
    ):
        """
        Args:
            model_path: Path to trained renderer model
            device: Device to run on
            enable: Whether to enable ML rendering
        """
        self.enable = enable
        self.device = device
        self.model = None

        if enable and model_path:
            try:
                self._load_model(model_path)
                print(f"✓ Loaded visual renderer from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load renderer model: {e}")
                self.enable = False

    def _load_model(self, model_path: str):
        """Load trained renderer model"""
        # TODO: Implement when Pix2Pix/Diffusion model is ready
        raise NotImplementedError("Visual renderer not yet implemented")

    def render(self, layout: Dict, size: int = 512) -> Optional[bytes]:
        """
        Render layout as photorealistic image

        Args:
            layout: Layout dict
            size: Output image size

        Returns:
            PNG image bytes (or None if disabled/failed)
        """
        if not self.enable or self.model is None:
            return None

        # TODO: Implement rendering
        raise NotImplementedError("Visual renderer not yet implemented")


def create_inference_pipeline(
    gnn_model_path: Optional[str] = None,
    renderer_model_path: Optional[str] = None,
    device: str = "cpu",
    enable_gnn: bool = True,
    enable_renderer: bool = False,
) -> tuple:
    """
    Create complete ML inference pipeline

    Args:
        gnn_model_path: Path to GNN model
        renderer_model_path: Path to renderer model
        device: Device to run on
        enable_gnn: Enable GNN refinement
        enable_renderer: Enable visual rendering

    Returns:
        (gnn_refiner, visual_renderer) tuple
    """
    gnn_refiner = GNNRefinerInference(
        model_path=gnn_model_path,
        device=device,
        enable=enable_gnn,
    )

    visual_renderer = VisualRendererInference(
        model_path=renderer_model_path,
        device=device,
        enable=enable_renderer,
    )

    return gnn_refiner, visual_renderer
