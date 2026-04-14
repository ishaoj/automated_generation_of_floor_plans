"""
Room type definitions with size constraints and Vastu preferences
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class RoomType(str, Enum):
    """Available room types"""
    BEDROOM = "bedroom"
    MASTER_BEDROOM = "master_bedroom"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    DINING = "dining"
    PUJA_ROOM = "puja_room"
    PARKING = "parking"
    BALCONY = "balcony"
    STAIRCASE = "staircase"
    STORE = "store"
    ENTRANCE = "entrance"
    OTS = "ots"  # Open to Sky - courtyard/light well


@dataclass
class RoomConfig:
    """Configuration for a room type"""
    name: str
    min_area: float  # sq ft
    max_area: float  # sq ft
    min_width: float  # ft
    min_height: float  # ft
    ideal_ratio: tuple[float, float]  # width:height ratio range
    priority: int  # Higher = placed first
    can_be_external: bool  # Can be on plot boundary


# Room configurations with realistic Indian residential sizes
# NOTE: max_area increased to allow rooms to expand and fill plot
# The constraint solver will try to maximize room sizes to cover the plot
ROOM_TYPES: dict[str, RoomConfig] = {
    RoomType.MASTER_BEDROOM: RoomConfig(
        name="Master Bedroom",
        min_area=150,
        max_area=400,  # Increased from 250 to allow expansion
        min_width=12,
        min_height=12,
        ideal_ratio=(0.8, 1.2),
        priority=10,
        can_be_external=True,
    ),
    RoomType.BEDROOM: RoomConfig(
        name="Bedroom",
        min_area=100,
        max_area=300,  # Increased from 180 to allow expansion
        min_width=10,
        min_height=10,
        ideal_ratio=(0.75, 1.33),
        priority=9,
        can_be_external=True,
    ),
    RoomType.LIVING_ROOM: RoomConfig(
        name="Living Room",
        min_area=150,
        max_area=500,  # Increased from 300 - living rooms can be large
        min_width=12,
        min_height=12,
        ideal_ratio=(0.6, 1.5),
        priority=8,
        can_be_external=True,
    ),
    RoomType.KITCHEN: RoomConfig(
        name="Kitchen",
        min_area=60,
        max_area=200,  # Increased from 120 for spacious kitchens
        min_width=8,
        min_height=8,
        ideal_ratio=(0.7, 1.4),
        priority=7,
        can_be_external=True,
    ),
    RoomType.DINING: RoomConfig(
        name="Dining Room",
        min_area=80,
        max_area=250,  # Increased from 150
        min_width=8,
        min_height=8,
        ideal_ratio=(0.7, 1.4),
        priority=6,
        can_be_external=False,
    ),
    RoomType.BATHROOM: RoomConfig(
        name="Bathroom",
        min_area=35,
        max_area=100,  # Increased from 60 for larger bathrooms
        min_width=5,
        min_height=6,
        ideal_ratio=(0.6, 1.2),
        priority=5,
        can_be_external=True,
    ),
    RoomType.PUJA_ROOM: RoomConfig(
        name="Puja Room",
        min_area=25,
        max_area=80,  # Increased from 50
        min_width=5,
        min_height=5,
        ideal_ratio=(0.8, 1.2),
        priority=4,
        can_be_external=False,
    ),
    RoomType.PARKING: RoomConfig(
        name="Parking",
        min_area=80,  # Reduced from 100 - compact single car parking
        max_area=200,
        min_width=8,   # Reduced from 9 - more flexible
        min_height=10, # FIXED: Reduced from 18 to 10 - was too long!
        ideal_ratio=(0.6, 1.0),  # Allow squarer shapes
        priority=3,
        can_be_external=True,
    ),
    RoomType.BALCONY: RoomConfig(
        name="Balcony",
        min_area=25,  # Reduced from 30 - small balconies are common
        max_area=100, # Increased for larger balconies
        min_width=3,   # Reduced from 4 - narrow balconies ok
        min_height=5,  # Reduced from 6
        ideal_ratio=(0.2, 0.6),  # Balconies are typically long and narrow
        priority=2,
        can_be_external=True,
    ),
    RoomType.STAIRCASE: RoomConfig(
        name="Staircase",
        min_area=35,  # Reduced from 40 - compact stairs
        max_area=100, # Increased for U-shaped stairs
        min_width=4,
        min_height=6,  # Reduced from 8
        ideal_ratio=(0.5, 0.8),
        priority=1,
        can_be_external=False,
    ),
    RoomType.STORE: RoomConfig(
        name="Store Room",
        min_area=20,
        max_area=50,
        min_width=4,
        min_height=5,
        ideal_ratio=(0.7, 1.3),
        priority=0,
        can_be_external=False,
    ),
    RoomType.OTS: RoomConfig(
        name="Open to Sky",
        min_area=36,  # Minimum 6x6 ft courtyard
        max_area=200, # Can be a large courtyard
        min_width=6,
        min_height=6,
        ideal_ratio=(0.8, 1.2),  # Should be roughly square
        priority=11,  # Highest priority - place first in center
        can_be_external=False,  # OTS should be internal
    ),
}


def get_room_config(room_type: str) -> RoomConfig:
    """Get configuration for a room type"""
    if room_type in ROOM_TYPES:
        return ROOM_TYPES[room_type]
    # Default config for unknown room types
    return RoomConfig(
        name=room_type.replace("_", " ").title(),
        min_area=50,
        max_area=150,
        min_width=6,
        min_height=6,
        ideal_ratio=(0.7, 1.3),
        priority=0,
        can_be_external=True,
    )
