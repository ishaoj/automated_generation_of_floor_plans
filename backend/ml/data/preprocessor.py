"""
Floor Plan Preprocessing

Converts raw CubiCasa floor plans into graph representations
compatible with our GNN models.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class FloorPlanPreprocessor:
    """
    Preprocesses floor plans for ML training

    Converts floor plans into:
    1. Graph representation (nodes = rooms, edges = adjacency/connections)
    2. Normalized coordinates and dimensions
    3. One-hot encoded room types
    """

    # Room type to index mapping
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

    def preprocess(self, floor_plan: Dict) -> Dict:
        """
        Convert raw floor plan to model-ready format

        Args:
            floor_plan: Floor plan dict from CubiCasaDataset

        Returns:
            Dict with:
                - 'node_features': (N, F) tensor of room features
                - 'edge_index': (2, E) tensor of edges
                - 'edge_attr': (E, A) tensor of edge attributes
                - 'positions': (N, 4) tensor of room positions (x, y, w, h)
                - 'plot_dims': (width, height) tuple
        """
        rooms = floor_plan['rooms']
        plot_width, plot_height = floor_plan['plot_dims']

        # Build node features
        node_features = []
        positions = []

        for room in rooms:
            # Room type one-hot encoding
            room_type_idx = self.ROOM_TYPE_TO_IDX.get(room['type'], 0)
            room_type_onehot = [0.0] * len(self.ROOM_TYPES)
            room_type_onehot[room_type_idx] = 1.0

            # Position and dimensions (normalized or absolute)
            x, y, w, h = room['bbox']

            if self.normalize_coords:
                x_norm = x / plot_width
                y_norm = y / plot_height
                w_norm = w / plot_width
                h_norm = h / plot_height
            else:
                x_norm, y_norm, w_norm, h_norm = x, y, w, h

            # Area
            area_norm = room['area'] / (plot_width * plot_height)

            # Node feature vector: [room_type_onehot (11), area (1), x_n, y_n, w_n, h_n (4)] = 16-D
            # Including normalised position lets GNN condition corrections on each room's location.
            node_feat = room_type_onehot + [area_norm, x_norm, y_norm, w_norm, h_norm]
            node_features.append(node_feat)

            # Store position separately for prediction target
            positions.append([x_norm, y_norm, w_norm, h_norm])

        # Build adjacency graph
        edge_index, edge_attr = self._build_adjacency_graph(rooms, plot_width, plot_height)

        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'plot_dims': (plot_width, plot_height),
            'num_rooms': len(rooms),
        }

    def _build_adjacency_graph(
        self,
        rooms: List[Dict],
        plot_width: float,
        plot_height: float
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Build graph edges based on room adjacency

        Two rooms are adjacent if they share a wall (bounding boxes overlap)
        """
        n_rooms = len(rooms)
        edge_index = [[], []]  # [source_nodes, target_nodes]
        edge_attr = []  # Edge attributes

        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                # Check if rooms i and j are adjacent
                is_adjacent, edge_features = self._rooms_adjacent(
                    rooms[i], rooms[j], plot_width, plot_height
                )

                if is_adjacent:
                    # Add bidirectional edges
                    edge_index[0].extend([i, j])
                    edge_index[1].extend([j, i])
                    edge_attr.extend([edge_features, edge_features])

        # If no edges found, create fully connected graph
        if len(edge_index[0]) == 0:
            for i in range(n_rooms):
                for j in range(n_rooms):
                    if i != j:
                        edge_index[0].append(i)
                        edge_index[1].append(j)
                        # Default edge features: [distance, 0, 0, 0]
                        dist = self._room_distance(rooms[i], rooms[j])
                        edge_attr.append([dist, 0, 0, 0])

        return edge_index, edge_attr

    def _rooms_adjacent(
        self,
        room1: Dict,
        room2: Dict,
        plot_width: float,
        plot_height: float
    ) -> Tuple[bool, List[float]]:
        """
        Check if two rooms are adjacent and compute edge features

        Edge features:
            - Distance between centroids (normalized)
            - Relative position: dx, dy (normalized)
            - Overlap indicator (0 or 1)
        """
        x1, y1, w1, h1 = room1['bbox']
        x2, y2, w2, h2 = room2['bbox']

        # Check for bounding box overlap/adjacency (within threshold)
        threshold = 2.0  # feet

        # Horizontal overlap
        h_overlap = (x1 < x2 + w2 + threshold and x2 < x1 + w1 + threshold)

        # Vertical overlap
        v_overlap = (y1 < y2 + h2 + threshold and y2 < y1 + h1 + threshold)

        is_adjacent = h_overlap and v_overlap

        # Compute edge features
        cx1, cy1 = room1['centroid']
        cx2, cy2 = room2['centroid']

        dx = (cx2 - cx1) / plot_width
        dy = (cy2 - cy1) / plot_height
        distance = np.sqrt(dx**2 + dy**2)

        edge_features = [
            distance,  # Distance
            dx,  # Relative x
            dy,  # Relative y
            float(is_adjacent),  # Adjacency flag
        ]

        return is_adjacent, edge_features

    def _room_distance(self, room1: Dict, room2: Dict) -> float:
        """Calculate normalized distance between room centroids"""
        cx1, cy1 = room1['centroid']
        cx2, cy2 = room2['centroid']
        return np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

    def graph_to_layout(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        plot_dims: Tuple[float, float]
    ) -> Dict:
        """
        Convert graph representation back to layout format

        Args:
            node_features: (N, F) room features
            positions: (N, 4) room positions (x, y, w, h) - normalized or absolute
            plot_dims: (width, height) plot dimensions

        Returns:
            Layout dict compatible with our system
        """
        plot_width, plot_height = plot_dims
        n_rooms = len(positions)

        rooms = []
        for i in range(n_rooms):
            # Decode room type
            room_type_onehot = node_features[i, :len(self.ROOM_TYPES)]
            room_type_idx = torch.argmax(room_type_onehot).item()
            room_type = self.ROOM_TYPES[room_type_idx]

            # Get position (denormalize if needed)
            x, y, w, h = positions[i].tolist()

            if self.normalize_coords:
                x *= plot_width
                y *= plot_height
                w *= plot_width
                h *= plot_height

            rooms.append({
                'id': f"{room_type}_{i}",
                'room_type': room_type,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': w * h,
            })

        return {
            'rooms': rooms,
            'plot_width': plot_width,
            'plot_height': plot_height,
        }


class GraphBatchCollator:
    """
    Collates multiple graphs into a batch for PyTorch Geometric

    Handles variable-sized graphs by creating a single large graph
    with disconnected components.
    """

    def __init__(self, preprocessor: FloorPlanPreprocessor):
        self.preprocessor = preprocessor

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of preprocessed floor plans

        Args:
            batch: List of dicts from FloorPlanPreprocessor.preprocess()

        Returns:
            Batched dict with:
                - 'node_features': (total_nodes, F) stacked features
                - 'edge_index': (2, total_edges) stacked edges with offset
                - 'edge_attr': (total_edges, A) stacked edge attributes
                - 'positions': (total_nodes, 4) stacked positions
                - 'batch': (total_nodes,) batch assignment for each node
                - 'num_graphs': number of graphs in batch
        """
        node_features_list = []
        edge_index_list = []
        edge_attr_list = []
        positions_list = []
        batch_list = []
        plot_dims_list = []

        node_offset = 0
        for graph_idx, graph in enumerate(batch):
            # Append node features and positions
            node_features_list.append(graph['node_features'])
            positions_list.append(graph['positions'])
            plot_dims_list.append(graph['plot_dims'])

            # Append edges with offset
            edge_index = graph['edge_index']
            edge_index_offset = edge_index + node_offset
            edge_index_list.append(edge_index_offset)
            edge_attr_list.append(graph['edge_attr'])

            # Create batch assignment
            num_nodes = len(graph['node_features'])
            batch_list.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))

            node_offset += num_nodes

        return {
            'node_features': torch.cat(node_features_list, dim=0),
            'edge_index': torch.cat(edge_index_list, dim=1),
            'edge_attr': torch.cat(edge_attr_list, dim=0),
            'positions': torch.cat(positions_list, dim=0),
            'batch': torch.cat(batch_list, dim=0),
            'num_graphs': len(batch),
            'plot_dims': plot_dims_list,
        }
