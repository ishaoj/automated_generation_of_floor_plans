"""
Graph Encoder

Converts floor plan layouts to graph representations for GNN processing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class GraphEncoder:
    """
    Encodes floor plan layouts as graphs

    Converts from our Layout schema to PyTorch Geometric compatible graphs.
    """

    # Room type to index mapping (same as preprocessor)
    ROOM_TYPES = [
        'bedroom',
        'living_room',
        'kitchen',
        'bathroom',
        'dining',
        'puja_room',
        'storage',
        'utility',
        'balcony',
        'parking',
        'staircase',
    ]

    ROOM_TYPE_TO_IDX = {rt: i for i, rt in enumerate(ROOM_TYPES)}

    def __init__(self, normalize_coords: bool = True):
        """
        Args:
            normalize_coords: Whether to normalize coordinates to [0, 1]
        """
        self.normalize_coords = normalize_coords

    def encode_layout(self, layout: Dict) -> Dict:
        """
        Convert Layout to graph representation

        Args:
            layout: Layout dict with 'rooms', 'plot_width', 'plot_length'

        Returns:
            Graph dict with:
                - 'node_features': (N, F) tensor
                - 'edge_index': (2, E) tensor
                - 'edge_attr': (E, A) tensor
                - 'positions': (N, 4) tensor of (x, y, w, h)
                - 'plot_dims': (width, height)
        """
        rooms = layout['rooms']
        plot_width = layout.get('plot_width', layout.get('plot_length'))  # handle both names
        plot_height = layout.get('plot_length', layout.get('plot_height'))

        # Encode node features
        node_features = []
        positions = []

        for room in rooms:
            # Room type one-hot
            room_type = room.get('room_type', room.get('type', 'bedroom'))
            room_type_idx = self.ROOM_TYPE_TO_IDX.get(room_type, 0)
            room_type_onehot = [0.0] * len(self.ROOM_TYPES)
            room_type_onehot[room_type_idx] = 1.0

            # Position and dimensions
            x = room['x']
            y = room['y']
            w = room['width']
            h = room['height']

            if self.normalize_coords:
                x_norm = x / plot_width
                y_norm = y / plot_height
                w_norm = w / plot_width
                h_norm = h / plot_height
            else:
                x_norm, y_norm, w_norm, h_norm = x, y, w, h

            # Area (normalized)
            area = room.get('area', w * h)
            area_norm = area / (plot_width * plot_height)

            # Zone encoding (optional, for Vastu awareness)
            zone = room.get('zone', 'center')
            zone_encoding = self._encode_zone(zone)

            # Node features: [room_type (11), area (1), zone (9)]
            node_feat = room_type_onehot + [area_norm] + zone_encoding
            node_features.append(node_feat)

            positions.append([x_norm, y_norm, w_norm, h_norm])

        # Build adjacency graph
        edge_index, edge_attr = self._build_edges(rooms, plot_width, plot_height)

        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'plot_dims': (plot_width, plot_height),
            'num_rooms': len(rooms),
        }

    def decode_graph(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        plot_dims: Tuple[float, float],
        original_layout: Optional[Dict] = None
    ) -> Dict:
        """
        Convert graph representation back to Layout

        Args:
            node_features: (N, F) room features
            positions: (N, 4) room positions
            plot_dims: (width, height)
            original_layout: Original layout to preserve metadata (optional)

        Returns:
            Layout dict
        """
        plot_width, plot_height = plot_dims
        n_rooms = len(positions)

        rooms = []
        for i in range(n_rooms):
            # Decode room type
            room_type_onehot = node_features[i, :len(self.ROOM_TYPES)]
            room_type_idx = torch.argmax(room_type_onehot).item()
            room_type = self.ROOM_TYPES[room_type_idx]

            # Get position
            x, y, w, h = positions[i].tolist()

            if self.normalize_coords:
                x *= plot_width
                y *= plot_height
                w *= plot_width
                h *= plot_height

            # Preserve original room metadata if available
            room_data = {
                'id': f"{room_type}_{i}",
                'room_type': room_type,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': w * h,
            }

            # Copy additional metadata from original if available
            if original_layout and i < len(original_layout.get('rooms', [])):
                orig_room = original_layout['rooms'][i]
                for key in ['zone', 'doors', 'windows']:
                    if key in orig_room:
                        room_data[key] = orig_room[key]

            rooms.append(room_data)

        layout = {
            'rooms': rooms,
            'plot_width': plot_width,
            'plot_length': plot_height,
        }

        # Preserve other layout metadata
        if original_layout:
            for key in ['facing_direction', 'entrance_position']:
                if key in original_layout:
                    layout[key] = original_layout[key]

        return layout

    def _encode_zone(self, zone: str) -> List[float]:
        """
        Encode Vastu zone as one-hot vector

        Zones: north, south, east, west, northeast, northwest, southeast, southwest, center
        """
        zones = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'center']
        zone_encoding = [0.0] * len(zones)

        if zone in zones:
            zone_encoding[zones.index(zone)] = 1.0

        return zone_encoding

    def _build_edges(
        self,
        rooms: List[Dict],
        plot_width: float,
        plot_height: float
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Build graph edges based on room adjacency and proximity

        Strategy:
        1. Connect rooms that share walls (adjacent)
        2. Connect rooms within distance threshold (nearby)
        3. If isolated rooms, connect to nearest neighbor
        """
        n_rooms = len(rooms)
        edge_index = [[], []]
        edge_attr = []

        # Distance threshold for considering rooms "nearby"
        proximity_threshold = 0.3  # 30% of plot diagonal

        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                # Check adjacency
                is_adjacent, edge_features = self._compute_edge_features(
                    rooms[i], rooms[j], plot_width, plot_height
                )

                # Distance-based connection
                distance = edge_features[0]
                is_nearby = distance < proximity_threshold

                if is_adjacent or is_nearby:
                    # Add bidirectional edges
                    edge_index[0].extend([i, j])
                    edge_index[1].extend([j, i])
                    edge_attr.extend([edge_features, edge_features])

        # Handle isolated nodes - connect to nearest neighbor
        connected_nodes = set(edge_index[0] + edge_index[1])
        for i in range(n_rooms):
            if i not in connected_nodes:
                # Find nearest neighbor
                min_dist = float('inf')
                nearest = 0
                for j in range(n_rooms):
                    if i != j:
                        _, edge_features = self._compute_edge_features(
                            rooms[i], rooms[j], plot_width, plot_height
                        )
                        dist = edge_features[0]
                        if dist < min_dist:
                            min_dist = dist
                            nearest = j

                # Connect to nearest
                edge_index[0].extend([i, nearest])
                edge_index[1].extend([nearest, i])
                _, edge_features = self._compute_edge_features(
                    rooms[i], rooms[nearest], plot_width, plot_height
                )
                edge_attr.extend([edge_features, edge_features])

        return edge_index, edge_attr

    def _compute_edge_features(
        self,
        room1: Dict,
        room2: Dict,
        plot_width: float,
        plot_height: float
    ) -> Tuple[bool, List[float]]:
        """
        Compute edge features between two rooms

        Returns:
            (is_adjacent, features)

        Features: [distance, dx, dy, adjacency_flag]
        """
        x1, y1 = room1['x'], room1['y']
        w1, h1 = room1['width'], room1['height']
        x2, y2 = room2['x'], room2['y']
        w2, h2 = room2['width'], room2['height']

        # Centroids
        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2

        # Normalized distance
        dx = (cx2 - cx1) / plot_width
        dy = (cy2 - cy1) / plot_height
        distance = np.sqrt(dx**2 + dy**2)

        # Check adjacency (bounding boxes overlap or touch)
        threshold = 2.0  # feet
        h_overlap = (x1 < x2 + w2 + threshold and x2 < x1 + w1 + threshold)
        v_overlap = (y1 < y2 + h2 + threshold and y2 < y1 + h1 + threshold)
        is_adjacent = h_overlap and v_overlap

        edge_features = [
            distance,
            dx,
            dy,
            float(is_adjacent),
        ]

        return is_adjacent, edge_features
