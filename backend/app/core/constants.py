"""
Constants used throughout the application
"""

from enum import Enum


class Direction(str, Enum):
    """Cardinal and inter-cardinal directions"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"
    CENTER = "center"


# Direction mappings
DIRECTIONS = {
    "north": Direction.NORTH,
    "south": Direction.SOUTH,
    "east": Direction.EAST,
    "west": Direction.WEST,
    "northeast": Direction.NORTHEAST,
    "northwest": Direction.NORTHWEST,
    "southeast": Direction.SOUTHEAST,
    "southwest": Direction.SOUTHWEST,
    "center": Direction.CENTER,
}

# Zone coordinates as percentage of plot (x_start, y_start, x_end, y_end)
# Assuming North is at top of the plot
ZONES = {
    Direction.NORTH: (0.33, 0.0, 0.67, 0.33),
    Direction.SOUTH: (0.33, 0.67, 0.67, 1.0),
    Direction.EAST: (0.67, 0.33, 1.0, 0.67),
    Direction.WEST: (0.0, 0.33, 0.33, 0.67),
    Direction.NORTHEAST: (0.67, 0.0, 1.0, 0.33),
    Direction.NORTHWEST: (0.0, 0.0, 0.33, 0.33),
    Direction.SOUTHEAST: (0.67, 0.67, 1.0, 1.0),
    Direction.SOUTHWEST: (0.0, 0.67, 0.33, 1.0),
    Direction.CENTER: (0.33, 0.33, 0.67, 0.67),
}

# Colors for rendering
ROOM_COLORS = {
    "master_bedroom": "#E8D5B7",  # Warm beige
    "bedroom": "#F5E6D3",  # Light beige
    "living_room": "#D4E6F1",  # Light blue
    "kitchen": "#FADBD8",  # Light coral
    "dining": "#D5F5E3",  # Light green
    "bathroom": "#E8DAEF",  # Light purple
    "puja_room": "#FCF3CF",  # Light yellow
    "parking": "#D5DBDB",  # Light gray
    "balcony": "#ABEBC6",  # Mint green
    "staircase": "#CCD1D1",  # Gray
    "store": "#F2D7D5",  # Light pink
    "entrance": "#85C1E9",  # Sky blue
    "ots": "#87CEEB",  # Sky blue - Open to Sky (Brahmasthan)
}

# Wall thickness in pixels (for rendering)
WALL_THICKNESS = 3

# Minimum passage width in feet
MIN_PASSAGE_WIDTH = 3.0

# Standard door width in feet
STANDARD_DOOR_WIDTH = 3.0
