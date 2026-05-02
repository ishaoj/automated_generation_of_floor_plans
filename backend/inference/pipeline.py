"""
MLInferencePipeline — GNN-only layout refinement.

Pipeline:
  Constraint Solver layout dict
    ↓
  GNN (GraphAttentionRefiner, 16-D position-aware)
    ↓
  Refined room positions (used by FloorPlanRenderer for the primary image)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_BACK = _HERE.parent.parent          # backend/
_ROOT = _BACK.parent                 # project root
for _p in [str(_ROOT), str(_BACK)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.models.gnn_refiner import GraphAttentionRefiner

logger = logging.getLogger(__name__)

# ── Room types the GNN was trained on (must match FloorPlanPreprocessor) ─────
_GNN_ROOM_TYPES = [
    'bedroom', 'living_room', 'kitchen', 'bathroom', 'dining',
    'puja_room', 'storage', 'utility', 'balcony', 'parking', 'staircase',
]
_ROOM_TYPE_IDX = {rt: i for i, rt in enumerate(_GNN_ROOM_TYPES)}


# ── Graph encoding (matches FloorPlanPreprocessor used during GNN training) ──

def _layout_to_graph(layout: Dict) -> Dict[str, torch.Tensor]:
    """
    Convert constraint-solver layout dict → GNN-ready tensors.

    Node features (16-D): [11-D room-type one-hot | 1-D area | 4-D normalised (x,y,w,h)]
    Positions in the node features let the GNN condition corrections on each
    room's current location, preventing the identity/collapsed-bias failure mode.

    Edge features (4-D): [distance, Δx, Δy, is_adjacent]
    Positions (N×4): normalised (x, y, w, h)  — still passed separately for
    the residual add at the end of GraphAttentionRefiner.forward()
    """
    rooms = layout["rooms"]
    pw    = float(layout["plot_width"])
    pl    = float(layout["plot_length"])
    plot_area = pw * pl

    node_features = []
    positions     = []
    centroids     = []

    for r in rooms:
        rt  = r.get("room_type", "")
        idx = _ROOM_TYPE_IDX.get(rt, 0)
        oh  = [0.0] * len(_GNN_ROOM_TYPES)
        oh[idx] = 1.0

        x, y, w, h = float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])
        area_norm = (w * h) / max(plot_area, 1.0)
        xn, yn, wn, hn = x / pw, y / pl, w / pw, h / pl
        # 16-D: room-type one-hot (11) + area (1) + normalised pos+size (4)
        node_features.append(oh + [area_norm, xn, yn, wn, hn])

        positions.append([x / pw, y / pl, w / pw, h / pl])
        centroids.append(((x + w / 2) / pw, (y + h / 2) / pl))

    N = len(rooms)
    src, dst, edge_feats = [], [], []
    THRESHOLD = 2.0   # feet — rooms touching or within 2 ft are adjacent

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cx1, cy1 = centroids[i]
            cx2, cy2 = centroids[j]
            dx   = cx2 - cx1
            dy   = cy2 - cy1
            dist = float(np.sqrt(dx ** 2 + dy ** 2))

            r1, r2 = rooms[i], rooms[j]
            h_ov = (r1["x"] < r2["x"] + r2["width"]  + THRESHOLD and
                    r2["x"] < r1["x"] + r1["width"]  + THRESHOLD)
            v_ov = (r1["y"] < r2["y"] + r2["height"] + THRESHOLD and
                    r2["y"] < r1["y"] + r1["height"] + THRESHOLD)

            # Sparse adjacency: only add edge if rooms are actually adjacent.
            # This matches the FloorPlanPreprocessor used during GNN training.
            # A dense (fully-connected) graph causes GAT oversmoothing: all
            # nodes collapse to the same representation after 1 layer.
            if h_ov and v_ov:
                src.append(i); dst.append(j)
                edge_feats.append([dist, dx, dy, 1.0])

    # Fallback: if no adjacency found (rooms too spread out), connect each
    # room only to its two nearest neighbours to preserve node identity.
    if not src:
        centroids_arr = [(c[0], c[1]) for c in centroids]
        for i in range(N):
            dists = [(float(np.sqrt((centroids_arr[i][0]-centroids_arr[j][0])**2 +
                                    (centroids_arr[i][1]-centroids_arr[j][1])**2)), j)
                     for j in range(N) if j != i]
            dists.sort()
            for d, j in dists[:2]:
                dx = centroids_arr[j][0] - centroids_arr[i][0]
                dy = centroids_arr[j][1] - centroids_arr[i][1]
                src.append(i); dst.append(j)
                edge_feats.append([d, dx, dy, 0.0])

    if not src:
        src, dst = [0], [0]
        edge_feats = [[0.0, 0.0, 0.0, 0.0]]

    return {
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "edge_index":    torch.tensor([src, dst],    dtype=torch.long),
        "edge_attr":     torch.tensor(edge_feats,    dtype=torch.float32),
        "positions":     torch.tensor(positions,     dtype=torch.float32),
    }


def _apply_refined_positions(refined_pos: torch.Tensor, layout: Dict) -> Dict:
    """Write GNN-refined normalised positions back to the layout dict."""
    pw = layout["plot_width"]
    pl = layout["plot_length"]
    pos = refined_pos.detach().cpu().numpy()
    new_rooms = []
    for i, r in enumerate(layout["rooms"]):
        if i >= len(pos):
            new_rooms.append(r)
            continue
        nx, ny, nw, nh = pos[i]
        new_rooms.append({
            **r,
            "x":      float(nx * pw),
            "y":      float(ny * pl),
            "width":  float(nw * pw),
            "height": float(nh * pl),
            "area":   float(nw * pw * nh * pl),
        })
    return {**layout, "rooms": new_rooms}


# ── Main pipeline class ───────────────────────────────────────────────────────

class MLInferencePipeline:
    """
    GNN-only layout refinement pipeline.

    Args:
        gnn_path : path to GNN best_model.pt (gnn_v3 or later)
        device   : 'auto' | 'cuda' | 'mps' | 'cpu'
    """

    def __init__(
        self,
        gnn_path: str,
        device:   str = "auto",
        **kwargs,               # absorb unused config keys (pix2pix_path, etc.)
    ):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        gnn_path_obj = Path(gnn_path)
        # If a relative path is given, resolve it from the project root
        if not gnn_path_obj.is_absolute():
            gnn_path_obj = (_ROOT / gnn_path).resolve()
        if not gnn_path_obj.exists():
            raise RuntimeError(
                f"[PIPELINE] GNN model not found at: {gnn_path_obj}\n"
                "Train with: python backend/ml/training/train_gnn.py"
            )
        gnn_path = str(gnn_path_obj)
        ckpt = torch.load(gnn_path, map_location=device, weights_only=False)
        self.gnn = GraphAttentionRefiner(
            node_features_dim=16,
            edge_features_dim=4,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
        ).to(device)
        self.gnn.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.gnn.eval()
        logger.info(f"[PIPELINE] GNN loaded from {gnn_path}  device={device}")

    # ── Main entry point ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, layout: Dict) -> Dict:
        """
        Run GNN refinement on a constraint-solver layout dict.

        Returns:
            {"refined_rooms": list[dict]}  — GNN-adjusted room positions
        """
        graph = _layout_to_graph(layout)
        refined = self.gnn(
            node_features=graph["node_features"].to(self.device),
            edge_index=graph["edge_index"].to(self.device),
            edge_attr=graph["edge_attr"].to(self.device),
            positions=graph["positions"].to(self.device),
        )
        refined_layout = _apply_refined_positions(refined, layout)
        return {"refined_rooms": refined_layout["rooms"]}

    @torch.no_grad()
    def generate_batch(self, layouts: List[Dict]) -> List[Dict]:
        return [self.generate(layout) for layout in layouts]

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str) -> "MLInferencePipeline":
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        return cls(
            gnn_path = cfg["gnn"]["model_path"],
            device   = cfg["inference"].get("device", "auto"),
        )
