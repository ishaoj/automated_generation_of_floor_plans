"""
Data Augmentation for Floor Plans

Augments floor plan graphs to improve model generalization.
"""

import torch
import numpy as np
import random
from typing import Dict, List
import copy


class FloorPlanAugmentation:
    """
    Applies data augmentation to floor plan graphs

    Augmentations:
    1. Horizontal/vertical flipping
    2. 90-degree rotations
    3. Small position jitter (realistic noise)
    4. Room size variations (±5%)
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        jitter_prob: float = 0.3,
        jitter_std: float = 0.02,  # 2% of plot size
        size_var_prob: float = 0.3,
        size_var_range: float = 0.05,  # ±5%
    ):
        """
        Args:
            flip_prob: Probability of flipping
            rotate_prob: Probability of rotating
            jitter_prob: Probability of adding position jitter
            jitter_std: Standard deviation of jitter (fraction of plot size)
            size_var_prob: Probability of size variation
            size_var_range: Range of size variation (fraction)
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.jitter_prob = jitter_prob
        self.jitter_std = jitter_std
        self.size_var_prob = size_var_prob
        self.size_var_range = size_var_range

    def __call__(self, graph: Dict) -> Dict:
        """
        Apply augmentation to a preprocessed graph

        Args:
            graph: Dict from FloorPlanPreprocessor.preprocess()

        Returns:
            Augmented graph dict
        """
        graph = copy.deepcopy(graph)  # Don't modify original

        positions = graph['positions'].clone()  # (N, 4): x, y, w, h

        # 1. Horizontal flip
        if random.random() < self.flip_prob:
            positions = self._flip_horizontal(positions)

        # 2. Vertical flip
        if random.random() < self.flip_prob:
            positions = self._flip_vertical(positions)

        # 3. Rotate 90/180/270 degrees
        if random.random() < self.rotate_prob:
            angle = random.choice([90, 180, 270])
            positions = self._rotate(positions, angle)

        # 4. Add position jitter
        if random.random() < self.jitter_prob:
            positions = self._add_jitter(positions)

        # 5. Size variation
        if random.random() < self.size_var_prob:
            positions = self._vary_size(positions)

        # Clamp positions to valid range [0, 1] if normalized
        positions = torch.clamp(positions, 0.0, 1.0)

        graph['positions'] = positions
        return graph

    def _flip_horizontal(self, positions: torch.Tensor) -> torch.Tensor:
        """Flip horizontally (mirror across vertical axis)"""
        positions = positions.clone()
        # x_new = 1 - x - w
        positions[:, 0] = 1.0 - positions[:, 0] - positions[:, 2]
        return positions

    def _flip_vertical(self, positions: torch.Tensor) -> torch.Tensor:
        """Flip vertically (mirror across horizontal axis)"""
        positions = positions.clone()
        # y_new = 1 - y - h
        positions[:, 1] = 1.0 - positions[:, 1] - positions[:, 3]
        return positions

    def _rotate(self, positions: torch.Tensor, angle: int) -> torch.Tensor:
        """
        Rotate by 90, 180, or 270 degrees

        Note: For normalized coordinates, rotation affects both position and dimensions
        """
        positions = positions.clone()

        if angle == 90:
            # (x, y, w, h) -> (y, 1-x-w, h, w)
            new_positions = torch.zeros_like(positions)
            new_positions[:, 0] = positions[:, 1]  # x = y
            new_positions[:, 1] = 1.0 - positions[:, 0] - positions[:, 2]  # y = 1-x-w
            new_positions[:, 2] = positions[:, 3]  # w = h
            new_positions[:, 3] = positions[:, 2]  # h = w
            positions = new_positions

        elif angle == 180:
            # (x, y, w, h) -> (1-x-w, 1-y-h, w, h)
            positions[:, 0] = 1.0 - positions[:, 0] - positions[:, 2]
            positions[:, 1] = 1.0 - positions[:, 1] - positions[:, 3]

        elif angle == 270:
            # (x, y, w, h) -> (1-y-h, x, h, w)
            new_positions = torch.zeros_like(positions)
            new_positions[:, 0] = 1.0 - positions[:, 1] - positions[:, 3]  # x = 1-y-h
            new_positions[:, 1] = positions[:, 0]  # y = x
            new_positions[:, 2] = positions[:, 3]  # w = h
            new_positions[:, 3] = positions[:, 2]  # h = w
            positions = new_positions

        return positions

    def _add_jitter(self, positions: torch.Tensor) -> torch.Tensor:
        """Add small random noise to positions"""
        positions = positions.clone()
        noise = torch.randn_like(positions) * self.jitter_std
        # Only jitter x, y (not width, height)
        noise[:, 2:] = 0
        positions += noise
        return positions

    def _vary_size(self, positions: torch.Tensor) -> torch.Tensor:
        """Randomly vary room sizes slightly"""
        positions = positions.clone()
        # Random scale factors for width and height
        scale = 1.0 + torch.randn(len(positions), 2) * self.size_var_range
        scale = torch.clamp(scale, 1.0 - self.size_var_range, 1.0 + self.size_var_range)

        # Apply to width and height
        positions[:, 2] *= scale[:, 0]
        positions[:, 3] *= scale[:, 1]

        return positions


class NoAugmentation:
    """Dummy augmentation that returns the input unchanged"""

    def __call__(self, graph: Dict) -> Dict:
        return graph
