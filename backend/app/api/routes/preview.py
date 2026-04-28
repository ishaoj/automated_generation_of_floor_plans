"""
Preview endpoints for Pix2Pix training progress.

GET  /api/preview
  Validation-set samples: semantic-map / generated / target triples.

POST /api/preview/from-request
  User-supplied room requirements → constraint solver → semantic map → Pix2Pix.
  Returns semantic map + generated image so you can see output quality for
  your specific room layout without waiting for full pipeline training.
"""

import base64
import io
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Query
from PIL import Image, ImageDraw
from pydantic import BaseModel
from torchvision import transforms

# Ensure backend/ is on sys.path
_BACKEND = Path(__file__).resolve().parent.parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from ml.data.cubicasa_loader import CubiCasaDataset
from ml.models.pix2pix_renderer import Pix2PixModel
from preprocessing.semantic_renderer import SemanticRenderer
from app.models.schemas import FloorPlanRequest
from app.services.graph_builder import RoomGraphBuilder
from app.services.constraint_solver import VastuConstraintSolver
from app.services.openings_placer import OpeningsPlacer

router = APIRouter()

# ── Constants (match train_pix2pix.py for the GET /preview endpoint) ─────────
_PIXEL_TO_FEET = 0.033

_ROOM_COLORS_TRAIN: Dict[str, Tuple[int, int, int]] = {
    "bedroom":     (255, 228, 225),
    "living_room": (224, 240, 255),
    "kitchen":     (255, 228, 181),
    "bathroom":    (230, 230, 250),
    "dining":      (240, 255, 240),
    "storage":     (245, 245, 220),
    "utility":     (245, 245, 220),
    "balcony":     (224, 255, 255),
    "parking":     (220, 220, 220),
    "corridor":    (255, 248, 220),
}

_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

_DEFAULT_CKPT = "ml/models/saved/pix2pix_v1/latest_checkpoint.pt"
_DEFAULT_DATA = "data/cubicasa5k_extracted/cubicasa5k"

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PreviewSample(BaseModel):
    semantic_map: str
    generated:    str
    target:       Optional[str] = None


class PreviewResponse(BaseModel):
    epoch:         int
    best_val_loss: float
    checkpoint:    str
    num_samples:   int
    samples:       List[PreviewSample]


class GeneratePreviewResponse(BaseModel):
    epoch:         int
    best_val_loss: float
    semantic_map:  str   # base64 PNG — colour-coded room layout
    generated:     str   # base64 PNG — Pix2Pix output
    rooms:         List[Dict]  # room metadata for legend


# ── Shared helpers ────────────────────────────────────────────────────────────

def _tensor_to_b64(t: torch.Tensor) -> str:
    arr = ((t.cpu().float().clamp(-1, 1) + 1) / 2 * 255).byte()
    arr = arr.permute(1, 2, 0).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _load_generator(ngf: int = 32, checkpoint: str = _DEFAULT_CKPT):
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _BACKEND.parent / checkpoint
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt["model_state_dict"]
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in raw_sd.items()
    }
    model = Pix2PixModel(in_channels=3, out_channels=3, ngf=ngf)
    model.load_state_dict(state_dict)
    model.eval()
    return model.generator, ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("nan"))


# ── GET /api/preview — validation dataset samples ─────────────────────────────

def _make_semantic_from_sample(sample: Dict, image_size: int) -> torch.Tensor:
    plot_w_ft, plot_h_ft = sample["plot_dims"]
    plot_w_px = plot_w_ft / _PIXEL_TO_FEET
    plot_h_px = plot_h_ft / _PIXEL_TO_FEET
    sz = image_size
    scale = min(sz / max(plot_w_px, 1), sz / max(plot_h_px, 1))

    img = Image.new("RGB", (sz, sz), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for room in sample["rooms"]:
        color = _ROOM_COLORS_TRAIN.get(room["type"], (255, 255, 255))
        pts = [(x * scale, y * scale) for x, y in room["polygon"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=color, outline=(0, 0, 0))
    return _NORMALIZE(img)


def _load_rendered(sample: Dict, image_size: int) -> torch.Tensor:
    png = Path(sample["svg_path"]).parent / "F1_scaled.png"
    img = Image.open(png).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    return _NORMALIZE(img)


@router.get("/preview", response_model=PreviewResponse)
def preview_from_dataset(
    checkpoint:  str = Query(default=_DEFAULT_CKPT),
    data_dir:    str = Query(default=_DEFAULT_DATA),
    num_samples: int = Query(default=4, ge=1, le=12),
    image_size:  int = Query(default=256),
    ngf:         int = Query(default=32),
):
    generator, epoch, best_val_loss = _load_generator(ngf, checkpoint)

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = _BACKEND.parent / data_dir
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {data_path}")

    base_ds = CubiCasaDataset(
        data_dir=str(data_path), split="val",
        include_images=False, filter_non_residential=True,
        min_rooms=2, max_rooms=10,
    )
    valid_samples = [
        s for s in base_ds.samples
        if (Path(s["svg_path"]).parent / "F1_scaled.png").exists()
    ][:num_samples]

    if not valid_samples:
        raise HTTPException(status_code=500, detail="No validation samples with F1_scaled.png found")

    results: List[PreviewSample] = []
    with torch.no_grad():
        for s in valid_samples:
            try:
                sem  = _make_semantic_from_sample(s, image_size)
                tgt  = _load_rendered(s, image_size)
                fake = generator(sem.unsqueeze(0))[0]
                results.append(PreviewSample(
                    semantic_map=_tensor_to_b64(sem),
                    generated=_tensor_to_b64(fake),
                    target=_tensor_to_b64(tgt),
                ))
            except Exception:
                continue

    if not results:
        raise HTTPException(status_code=500, detail="Inference failed for all samples")

    return PreviewResponse(
        epoch=epoch,
        best_val_loss=round(best_val_loss, 4),
        checkpoint=str(_BACKEND.parent / checkpoint),
        num_samples=len(results),
        samples=results,
    )


# ── POST /api/preview/from-request — user room requirements ──────────────────

_graph_builder = RoomGraphBuilder()
_solver        = VastuConstraintSolver(grid_size=1)
_openings      = OpeningsPlacer()
_renderer      = SemanticRenderer(image_size=256)


@router.post("/preview/from-request", response_model=GeneratePreviewResponse)
def preview_from_request(
    request:    FloorPlanRequest,
    checkpoint: str = Query(default=_DEFAULT_CKPT),
    ngf:        int = Query(default=32),
):
    # ── 1. Build room graph & solve layout ───────────────────────────────
    _graph_builder.build_from_request(request)
    room_nodes = _graph_builder.get_nodes_by_priority()

    is_valid, msg = _graph_builder.validate_against_plot(request.plot_area)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    seed = random.randint(0, 100_000)
    layout = _solver.solve(request, room_nodes, seed=seed, variation_mode=0)
    if layout is None:
        raise HTTPException(
            status_code=400,
            detail="Constraint solver could not fit all rooms. Try a larger plot or fewer rooms."
        )
    layout = _openings.place_openings(layout)

    # ── 2. Build room list for SemanticRenderer (feet coordinates) ────────
    rooms_for_renderer = [
        {"type": r.room_type, "bbox": (r.x, r.y, r.width, r.height)}
        for r in layout.rooms
    ]

    sem_arr = _renderer.render_from_feet(
        rooms_for_renderer,
        plot_width_ft=layout.plot_width,
        plot_height_ft=layout.plot_length,
    )  # H×W×3 uint8

    # ── 3. Convert to tensor & run Pix2Pix ───────────────────────────────
    sem_pil    = Image.fromarray(sem_arr)
    sem_tensor = _NORMALIZE(sem_pil)  # (3, 256, 256) in [-1, 1]

    generator, epoch, best_val_loss = _load_generator(ngf, checkpoint)
    with torch.no_grad():
        fake = generator(sem_tensor.unsqueeze(0))[0]

    # ── 4. Build room metadata for legend ─────────────────────────────────
    rooms_meta = [
        {
            "room_type": r.room_type,
            "x": r.x, "y": r.y,
            "width": r.width, "height": r.height,
            "area": r.area,
        }
        for r in layout.rooms
    ]

    return GeneratePreviewResponse(
        epoch=epoch,
        best_val_loss=round(best_val_loss, 4),
        semantic_map=_pil_to_b64(sem_pil),
        generated=_tensor_to_b64(fake),
        rooms=rooms_meta,
    )
