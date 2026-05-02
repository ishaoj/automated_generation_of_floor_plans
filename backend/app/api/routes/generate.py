"""
Floor plan generation endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import base64
from typing import Optional

from app.models.schemas import (
    FloorPlanRequest,
    FloorPlanResponse,
    GeneratedPlan,
    VastuScore,
)
from app.services.generator import FloorPlanGenerator
from app.core.vastu_rules import get_vastu_rules

router = APIRouter()

# Lazy-loaded singleton — instantiated on first request, not at import time.
# This prevents a torch/GNN startup error from crashing the entire app process.
_generator: FloorPlanGenerator | None = None


def _get_generator() -> FloorPlanGenerator:
    global _generator
    if _generator is None:
        _generator = FloorPlanGenerator()
    return _generator


@router.post("/generate", response_model=FloorPlanResponse)
async def generate_floor_plans(request: FloorPlanRequest):
    """
    Generate Vastu-compliant floor plans based on user requirements.

    Returns multiple variations with Vastu compliance scores.
    """
    try:
        plans = _get_generator().generate(request)
        return FloorPlanResponse(
            success=True,
            message=f"Generated {len(plans)} floor plan variations",
            plans=plans,
            request_summary={
                "plot_size": f"{request.plot_length}x{request.plot_width} ft",
                "facing": request.facing_direction,
                "bedrooms": request.num_bedrooms,
                "bathrooms": request.num_bathrooms,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vastu-rules")
async def get_rules():
    """
    Get all Vastu rules used by the system.

    Useful for displaying rule explanations to users.
    """
    return get_vastu_rules()


@router.get("/room-types")
async def get_room_types():
    """Get available room types and their properties"""
    from app.models.room_types import ROOM_TYPES
    return ROOM_TYPES
