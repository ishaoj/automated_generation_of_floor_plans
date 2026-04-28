"""
MLInferencePipeline — end-to-end mandatory ML inference.

Pipeline:
  Constraint Solver layout dict
    ↓
  1. GNN (GraphAttentionRefiner, 12-D)  → refined room positions
    ↓
  2. SemanticRenderer                   → 256×256 RGB semantic image
    ↓
  3. Pix2Pix UNetGenerator (ngf=32)     → realistic floor plan PNG

Both GNN and Pix2Pix are required. Missing checkpoints raise RuntimeError.
"""

import base64
import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter, ImageEnhance

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_BACK = _HERE.parent.parent          # backend/
_ROOT = _BACK.parent                 # project root
for _p in [str(_ROOT), str(_BACK)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.models.gnn_refiner      import GraphAttentionRefiner
from ml.models.generator_unet   import UNetGenerator
from preprocessing.semantic_renderer import SemanticRenderer

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

    Node features (12-D): [11-D room-type one-hot | 1-D normalised area]
    Edge features  (4-D): [distance, Δx, Δy, is_adjacent]
    Positions      (N×4): normalised (x, y, w, h)
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
        node_features.append(oh + [area_norm])

        positions.append([x / pw, y / pl, w / pw, h / pl])
        centroids.append(((x + w / 2) / pw, (y + h / 2) / pl))

    N = len(rooms)
    src, dst, edge_feats = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cx1, cy1 = centroids[i]
            cx2, cy2 = centroids[j]
            dx   = cx2 - cx1
            dy   = cy2 - cy1
            dist = float(np.sqrt(dx ** 2 + dy ** 2))

            # Adjacency: rooms are neighbours if bboxes are within 2 ft
            r1, r2   = rooms[i], rooms[j]
            threshold = 2.0
            h_ov = (r1["x"] < r2["x"] + r2["width"]  + threshold and
                    r2["x"] < r1["x"] + r1["width"]  + threshold)
            v_ov = (r1["y"] < r2["y"] + r2["height"] + threshold and
                    r2["y"] < r1["y"] + r1["height"] + threshold)
            adj  = float(h_ov and v_ov)

            src.append(i); dst.append(j)
            edge_feats.append([dist, dx, dy, adj])

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
    End-to-end ML inference: GNN refinement + Pix2Pix rendering.

    Args:
        gnn_path        : path to GNN best_model.pt
        pix2pix_path    : path to Pix2Pix best_model.pt (full Pix2PixModel ckpt)
        device          : 'auto' | 'cuda' | 'mps' | 'cpu'
        image_size      : semantic image resolution (square)
        ngf             : Pix2Pix generator base channels (must match training)
    """

    def __init__(
        self,
        gnn_path:     str,
        pix2pix_path: str,
        device:       str = "auto",
        image_size:   int = 256,
        ngf:          int = 32,
        **kwargs,                 # absorb unused config keys
    ):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device     = device
        self.image_size = image_size

        # ── Load GNN ──────────────────────────────────────────────────────
        gnn_path_obj = Path(gnn_path)
        if not gnn_path_obj.exists():
            raise RuntimeError(
                f"[PIPELINE] GNN model not found at: {gnn_path}\n"
                "Train with: python backend/ml/training/train_gnn.py"
            )
        ckpt = torch.load(gnn_path, map_location=device, weights_only=False)
        self.gnn = GraphAttentionRefiner(
            node_features_dim=12,
            edge_features_dim=4,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
        ).to(device)
        sd = ckpt.get("model_state_dict", ckpt)
        self.gnn.load_state_dict(sd)
        self.gnn.eval()
        logger.info(f"[PIPELINE] GNN loaded from {gnn_path}")

        # ── Load Pix2Pix generator ─────────────────────────────────────────
        p2p_path_obj = Path(pix2pix_path)
        if not p2p_path_obj.exists():
            raise RuntimeError(
                f"[PIPELINE] Pix2Pix model not found at: {pix2pix_path}\n"
                "Train with: python backend/ml/training/train_pix2pix.py"
            )
        p2p_ckpt = torch.load(pix2pix_path, map_location=device, weights_only=False)
        raw_sd   = p2p_ckpt.get("model_state_dict", p2p_ckpt)

        # Strip torch.compile() prefix and extract just generator weights
        gen_sd = {}
        for k, v in raw_sd.items():
            k = k.replace("_orig_mod.", "")          # torch.compile prefix
            if k.startswith("generator."):
                gen_sd[k[len("generator."):]] = v    # drop "generator." prefix

        self.generator = UNetGenerator(
            in_channels=3, out_channels=3, ngf=ngf
        ).to(device)
        self.generator.load_state_dict(gen_sd)
        self.generator.eval()
        logger.info(f"[PIPELINE] Pix2Pix generator (ngf={ngf}) loaded from {pix2pix_path}")

        # ── Semantic renderer ──────────────────────────────────────────────
        self.renderer = SemanticRenderer(image_size=image_size)

        logger.info(f"[PIPELINE] Ready  device={device}  image_size={image_size}")

    # ── Main entry point ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, layout: Dict) -> str:
        """
        Run GNN + Pix2Pix on a constraint-solver layout dict.

        Returns base64-encoded PNG string.
        """
        # 1. GNN refinement
        graph     = _layout_to_graph(layout)
        nf        = graph["node_features"].to(self.device)
        ei        = graph["edge_index"].to(self.device)
        ea        = graph["edge_attr"].to(self.device)
        pos       = graph["positions"].to(self.device)

        refined   = self.gnn(
            node_features=nf,
            edge_index=ei,
            edge_attr=ea,
            positions=pos,
        )                                        # (N, 4) normalised

        refined_layout = _apply_refined_positions(refined, layout)

        # 2. Semantic image
        rooms_for_renderer = [
            {
                "type":     r["room_type"],
                "zone":     r.get("zone"),
                "bbox_norm": (
                    r["x"] / layout["plot_width"],
                    r["y"] / layout["plot_length"],
                    r["width"]  / layout["plot_width"],
                    r["height"] / layout["plot_length"],
                ),
            }
            for r in refined_layout["rooms"]
        ]
        sem_arr = self.renderer.render_from_normalized(rooms_for_renderer)  # H×W×3

        sem_tensor = (
            torch.from_numpy(sem_arr)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 127.5 - 1.0
        ).to(self.device)                        # (1, 3, H, W)

        # 3. Pix2Pix
        fake      = self.generator(sem_tensor)   # (1, 3, H, W) in [-1, 1]
        img_arr   = ((fake.squeeze(0).clamp(-1, 1) + 1) / 2 * 255).byte()
        img_arr   = img_arr.permute(1, 2, 0).cpu().numpy()
        img       = Image.fromarray(img_arr, "RGB")

        # Post-process: upscale + sharpen to compensate for soft early output
        out_size  = self.image_size * 2
        img = img.resize((out_size, out_size), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(2.5)
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Brightness(img).enhance(1.1)

        # 4. Base64 encode
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @torch.no_grad()
    def generate_batch(self, layouts: List[Dict]) -> List[str]:
        return [self.generate(layout) for layout in layouts]

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str) -> "MLInferencePipeline":
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        return cls(
            gnn_path     = cfg["gnn"]["model_path"],
            pix2pix_path = cfg["pix2pix"]["model_path"],
            device       = cfg["inference"].get("device", "auto"),
            image_size   = cfg["inference"].get("semantic_image_size", 256),
            ngf          = cfg["pix2pix"].get("ngf", 32),
        )
