"""
GNN Layout Refiner

Graph Neural Network that refines constraint-solver layouts to make them
more realistic based on patterns learned from real floor plans.

Architecture:
- Input: Layout from constraint solver (as graph)
- Processing: Graph Attention Network (GAT) layers
- Output: Refined positions and dimensions for each room
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


try:
    from torch_geometric.nn import GATConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed. GNN models will not be available.")


class GraphAttentionRefiner(nn.Module):
    """
    Graph Attention Network for layout refinement

    Architecture:
        Input Graph → GAT Layers → MLP Decoder → Position Adjustments

    The model predicts:
        - Position adjustments (Δx, Δy)
        - Dimension refinements (Δw, Δh)
    """

    def __init__(
        self,
        node_features_dim: int = 16,  # 11 room types + 1 area + 4 normalised (x,y,w,h)
        edge_features_dim: int = 4,   # distance, dx, dy, adjacency
        hidden_dim: int = 128,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            node_features_dim: Dimension of node feature vectors
            edge_features_dim: Dimension of edge feature vectors
            hidden_dim: Hidden dimension for GNN layers
            num_gat_layers: Number of GAT layers
            num_heads: Number of attention heads per layer
            dropout: Dropout rate
        """
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch-geometric"
            )

        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.dropout = dropout

        # Input projection
        self.node_encoder = nn.Linear(node_features_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_gat_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * num_heads

            # Last layer outputs hidden_dim, others output hidden_dim * num_heads
            out_dim = hidden_dim if i == num_gat_layers - 1 else hidden_dim

            self.gat_layers.append(
                GATConv(
                    in_dim,
                    out_dim,
                    heads=num_heads if i < num_gat_layers - 1 else 1,
                    dropout=dropout,
                    concat=True if i < num_gat_layers - 1 else False,
                )
            )

            self.layer_norms.append(nn.LayerNorm(hidden_dim * (num_heads if i < num_gat_layers - 1 else 1)))

        # Position refinement decoder
        self.position_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # Output: Δx, Δy, Δw, Δh
            nn.Tanh(),  # Bounded adjustments in [-1, 1]
        )

        # Max adjustment per room: ~20% of normalized plot dimension.
        # Covers ~2.5σ of the training noise (pos_std=0.08, size_std=0.04).
        self.adjustment_scale = 0.20

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            node_features: (N, F) node feature matrix
            edge_index: (2, E) edge indices
            edge_attr: (E, A) edge attributes
            positions: (N, 4) current positions (x, y, w, h)
            batch: (N,) batch assignment for each node (optional)

        Returns:
            refined_positions: (N, 4) refined positions
        """
        # Encode node and edge features
        x = self.node_encoder(node_features)  # (N, hidden_dim)

        # Process through GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_in = x
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if i > 0 and x.shape == x_in.shape:
                x = x + x_in

        # Decode position adjustments
        adjustments = self.position_decoder(x)  # (N, 4)
        adjustments = adjustments * self.adjustment_scale  # Scale down adjustments

        # Apply adjustments to original positions
        refined_positions = positions + adjustments

        # Clamp all to [0, 1], then separately enforce min size for w/h
        # Use torch.cat to avoid in-place ops that break autograd
        xy = torch.clamp(refined_positions[:, :2], 0.0, 1.0)
        wh = torch.clamp(refined_positions[:, 2:], 0.05, 1.0)
        refined_positions = torch.cat([xy, wh], dim=1)

        return refined_positions

    def refine_with_constraints(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        positions: torch.Tensor,
        min_room_size: float = 0.05,
        max_adjustment: float = 0.15,
    ) -> torch.Tensor:
        """
        Refine layout with hard constraints enforced

        Args:
            node_features: (N, F) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, A) edge attributes
            positions: (N, 4) current positions
            min_room_size: Minimum room dimension (normalized)
            max_adjustment: Maximum position adjustment allowed

        Returns:
            refined_positions: (N, 4) constrained refined positions
        """
        # Get raw refinement
        refined = self.forward(node_features, edge_index, edge_attr, positions)

        # Enforce maximum adjustment constraint
        max_delta = max_adjustment
        delta = refined - positions
        delta = torch.clamp(delta, -max_delta, max_delta)
        refined = positions + delta

        # Enforce minimum room size
        refined[:, 2:] = torch.clamp(refined[:, 2:], min_room_size, 1.0)

        # Ensure rooms stay within plot bounds
        refined[:, 0] = torch.clamp(refined[:, 0], 0.0, 1.0 - refined[:, 2])
        refined[:, 1] = torch.clamp(refined[:, 1], 0.0, 1.0 - refined[:, 3])

        return refined


class GNNRefiner:
    """
    Production-ready wrapper for GNN layout refinement

    Handles:
    - Model loading
    - Layout encoding/decoding
    - Inference
    - Constraint enforcement
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        # Initialize model
        self.model = GraphAttentionRefiner(
            node_features_dim=21,
            edge_features_dim=4,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
            dropout=0.0,  # No dropout at inference
        ).to(self.device)

        # Load weights if provided
        if model_path:
            self.load_model(model_path)

        self.model.eval()

        # Import encoder
        from .graph_encoder import GraphEncoder
        self.encoder = GraphEncoder(normalize_coords=True)

    def load_model(self, model_path: str):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Loaded model from {model_path}")

    def refine(self, layout: Dict, request: Optional[Dict] = None) -> Dict:
        """
        Refine layout to make it more realistic

        Args:
            layout: Layout dict from constraint solver
            request: Original request (optional, for context)

        Returns:
            Refined layout dict
        """
        # Encode layout as graph
        graph = self.encoder.encode_layout(layout)

        # Move to device
        node_features = graph['node_features'].to(self.device)
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
                min_room_size=0.05,
                max_adjustment=0.15,
            )

        # Decode back to layout
        refined_layout = self.encoder.decode_graph(
            node_features=node_features.cpu(),
            positions=refined_positions.cpu(),
            plot_dims=graph['plot_dims'],
            original_layout=layout,
        )

        return refined_layout

    def batch_refine(self, layouts: list[Dict]) -> list[Dict]:
        """
        Refine multiple layouts in batch

        Args:
            layouts: List of layout dicts

        Returns:
            List of refined layouts
        """
        refined = []
        for layout in layouts:
            refined.append(self.refine(layout))
        return refined
