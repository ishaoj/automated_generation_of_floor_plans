"""
Custom loss functions for floor plan generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayoutRefinementLoss(nn.Module):
    """
    Multi-objective loss for layout refinement

    Components:
    1. Position loss: MSE between predicted and target positions
    2. Overlap penalty: Penalize room overlaps
    3. Area preservation: Keep total area similar
    4. Adjacency preservation: Maintain room connectivity from target
    """

    def __init__(
        self,
        position_weight: float = 5.0,
        overlap_weight: float = 1.0,
        area_weight: float = 0.3,
        adjacency_weight: float = 0.2,
    ):
        """
        Args:
            position_weight: Weight for position refinement loss
            overlap_weight: Weight for overlap penalty
            area_weight: Weight for area preservation loss
            adjacency_weight: Weight for adjacency preservation loss
        """
        super().__init__()

        self.position_weight = position_weight
        self.overlap_weight = overlap_weight
        self.area_weight = area_weight
        self.adjacency_weight = adjacency_weight

    def forward(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute loss

        Args:
            pred_positions: (N, 4) predicted positions (x, y, w, h)
            target_positions: (N, 4) target positions
            edge_index: (2, E) edge indices (optional, for adjacency loss)
            batch: (N,) batch assignment (optional)

        Returns:
            Dict with total loss and individual components
        """
        # 1. Position refinement loss (MSE)
        position_loss = F.mse_loss(pred_positions, target_positions)

        # 2. Overlap penalty
        overlap_loss = self._compute_overlap_penalty(pred_positions, batch)

        # 3. Area preservation loss
        area_loss = self._compute_area_loss(pred_positions, target_positions)

        # 4. Adjacency preservation loss
        if edge_index is not None:
            adjacency_loss = self._compute_adjacency_loss(
                pred_positions, target_positions, edge_index
            )
        else:
            adjacency_loss = torch.tensor(0.0, device=pred_positions.device)

        # Total loss
        total_loss = (
            self.position_weight * position_loss +
            self.overlap_weight * overlap_loss +
            self.area_weight * area_loss +
            self.adjacency_weight * adjacency_loss
        )

        return {
            'total': total_loss,
            'position': position_loss,
            'overlap': overlap_loss,
            'area': area_loss,
            'adjacency': adjacency_loss,
        }

    def _compute_overlap_penalty(
        self,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute penalty for room overlaps

        Args:
            positions: (N, 4) room positions
            batch: (N,) batch assignment

        Returns:
            Overlap penalty (0 if no overlaps)
        """
        N = len(positions)
        device = positions.device

        if N == 0:
            return torch.tensor(0.0, device=device)

        # If batch is provided, only check overlaps within same graph
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        total_overlap = 0.0
        n_pairs = 0

        # Check each pair of rooms
        for i in range(N):
            for j in range(i + 1, N):
                # Skip if from different graphs
                if batch[i] != batch[j]:
                    continue

                # Compute overlap
                overlap = self._box_overlap(positions[i], positions[j])
                total_overlap += overlap
                n_pairs += 1

        if n_pairs == 0:
            return torch.tensor(0.0, device=device)

        return total_overlap / n_pairs

    def _box_overlap(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Compute overlap area between two boxes

        Args:
            box1, box2: (4,) tensors of (x, y, w, h)

        Returns:
            Overlap area (0 if no overlap)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Compute intersection
        x_left = torch.max(x1, x2)
        y_top = torch.max(y1, y2)
        x_right = torch.min(x1 + w1, x2 + w2)
        y_bottom = torch.min(y1 + h1, y2 + h2)

        # Check if boxes intersect
        if x_right > x_left and y_bottom > y_top:
            intersection = (x_right - x_left) * (y_bottom - y_top)
            return intersection
        else:
            return torch.tensor(0.0, device=box1.device)

    def _compute_area_loss(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize deviation in total area

        Args:
            pred_positions: (N, 4) predicted positions
            target_positions: (N, 4) target positions

        Returns:
            Area preservation loss
        """
        # Compute areas
        pred_areas = pred_positions[:, 2] * pred_positions[:, 3]
        target_areas = target_positions[:, 2] * target_positions[:, 3]

        # Total area should be preserved
        pred_total = pred_areas.sum()
        target_total = target_areas.sum()

        return F.l1_loss(pred_total, target_total)

    def _compute_adjacency_loss(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Preserve adjacency relationships from target layout

        Args:
            pred_positions: (N, 4) predicted positions
            target_positions: (N, 4) target positions
            edge_index: (2, E) edge indices

        Returns:
            Adjacency preservation loss
        """
        if len(edge_index[0]) == 0:
            return torch.tensor(0.0, device=pred_positions.device)

        # Get source and target nodes
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # Compute distances between connected rooms
        pred_distances = self._compute_distances(pred_positions, src_nodes, dst_nodes)
        target_distances = self._compute_distances(target_positions, src_nodes, dst_nodes)

        # Adjacency loss: preserve relative distances
        return F.mse_loss(pred_distances, target_distances)

    def _compute_distances(
        self,
        positions: torch.Tensor,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances between pairs of rooms"""
        src_pos = positions[src_nodes]
        dst_pos = positions[dst_nodes]

        # Centroids
        src_centroids = src_pos[:, :2] + src_pos[:, 2:] / 2
        dst_centroids = dst_pos[:, :2] + dst_pos[:, 2:] / 2

        # Euclidean distances
        distances = torch.norm(src_centroids - dst_centroids, dim=1)

        return distances


class VastuAwareLoss(nn.Module):
    """
    Loss function that incorporates Vastu principles

    Penalizes layouts that violate Vastu rules
    """

    def __init__(self, base_loss_weight: float = 1.0, vastu_weight: float = 0.5):
        super().__init__()
        self.base_loss = LayoutRefinementLoss()
        self.base_loss_weight = base_loss_weight
        self.vastu_weight = vastu_weight

    def forward(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute Vastu-aware loss

        Args:
            pred_positions: (N, 4) predicted positions
            target_positions: (N, 4) target positions
            node_features: (N, F) node features (includes room types)
            edge_index: (2, E) edge indices
            batch: (N,) batch assignment

        Returns:
            Dict with total loss and components
        """
        # Base refinement loss
        base_losses = self.base_loss(pred_positions, target_positions, edge_index, batch)

        # Vastu compliance penalty
        vastu_penalty = self._compute_vastu_penalty(pred_positions, node_features, batch)

        # Total loss
        total_loss = (
            self.base_loss_weight * base_losses['total'] +
            self.vastu_weight * vastu_penalty
        )

        return {
            'total': total_loss,
            'base': base_losses['total'],
            'vastu': vastu_penalty,
            **base_losses,
        }

    def _compute_vastu_penalty(
        self,
        positions: torch.Tensor,
        node_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute penalty for Vastu violations

        Basic Vastu rules:
        - Kitchen should be in southeast
        - Master bedroom in southwest
        - Puja room in northeast
        - Bathrooms not in northeast
        """
        N = len(positions)
        device = positions.device

        if N == 0:
            return torch.tensor(0.0, device=device)

        # Extract room types (first 11 features are room type one-hot)
        # 0: bedroom, 1: living_room, 2: kitchen, 3: bathroom, 4: dining, 5: puja_room
        room_type_idx = torch.argmax(node_features[:, :11], dim=1)

        penalty = torch.tensor(0.0, device=device)

        # Check each room
        for i in range(N):
            x, y, w, h = positions[i]
            cx, cy = x + w/2, y + h/2  # Centroid

            # Determine zone (rough approximation)
            # Northeast: x > 0.6, y < 0.4
            # Southeast: x > 0.6, y > 0.6
            # Southwest: x < 0.4, y > 0.6
            # Northwest: x < 0.4, y < 0.4

            room_type = room_type_idx[i].item()

            # Kitchen should be in southeast (x > 0.5, y > 0.5)
            if room_type == 2:  # kitchen
                if cx < 0.5 or cy < 0.5:
                    penalty += 0.5

            # Puja room should be in northeast (x > 0.5, y < 0.5)
            if room_type == 5:  # puja_room
                if cx < 0.5 or cy > 0.5:
                    penalty += 0.5

            # Bathroom not in northeast
            if room_type == 3:  # bathroom
                if cx > 0.5 and cy < 0.5:
                    penalty += 0.5

        return penalty / N if N > 0 else torch.tensor(0.0, device=device)
