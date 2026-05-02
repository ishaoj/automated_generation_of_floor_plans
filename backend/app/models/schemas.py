"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
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


class PlotType(str, Enum):
    """
    Plot types based on road access / open sides

    LEARNING NOTE:
    - In real estate, plots can have different configurations
    - This affects where we can place windows (need exterior wall)
    - Also affects ventilation and natural light
    """
    OPEN_ALL = "open_all"          # Standalone plot - all 4 sides open (rare, premium)
    OPEN_ONE = "open_one"          # Row house - only facing side open (3 sides shared)
    OPEN_TWO_CORNER = "open_two_corner"  # Corner plot - facing + one adjacent side open
    OPEN_TWO_OPPOSITE = "open_two_opposite"  # Through plot - facing + opposite side open (rare)


class FloorPlanRequest(BaseModel):
    """Input parameters for floor plan generation"""

    # Plot dimensions (in feet)
    plot_length: float = Field(..., gt=15, le=200, description="Plot length in feet")
    plot_width: float = Field(..., gt=15, le=200, description="Plot width in feet")

    # Facing direction (main entrance direction)
    facing_direction: Literal["north", "south", "east", "west"] = Field(
        ..., description="Direction the main entrance faces"
    )

    # Room requirements
    num_bedrooms: int = Field(..., ge=1, le=6, description="Number of bedrooms")
    num_bathrooms: int = Field(..., ge=1, le=5, description="Number of bathrooms")

    # Optional rooms
    include_kitchen: bool = Field(default=True, description="Include kitchen")
    include_living_room: bool = Field(default=True, description="Include living room")
    include_dining: bool = Field(default=True, description="Include dining area")
    include_puja_room: bool = Field(default=False, description="Include puja/prayer room")
    include_parking: bool = Field(default=False, description="Include parking space")
    include_balcony: bool = Field(default=False, description="Include balcony")
    include_staircase: bool = Field(default=False, description="Include staircase (for multi-floor)")
    include_ots: bool = Field(default=False, description="Include OTS (Open to Sky) courtyard - Vastu Brahmasthan")

    # Generation settings
    num_variations: int = Field(default=3, ge=1, le=5, description="Number of variations to generate")

    # Plot configuration
    plot_type: Literal["open_all", "open_one", "open_two_corner", "open_two_opposite"] = Field(
        default="open_all",
        description="Plot type: open_all (standalone), open_one (row house), open_two_corner (corner plot)"
    )

    # For corner plots: which adjacent side is also open (left or right when facing the plot)
    corner_open_side: Optional[Literal["left", "right"]] = Field(
        default=None,
        description="For corner plots: which side is also open (left/right relative to facing)"
    )

    @field_validator("plot_length", "plot_width")
    @classmethod
    def validate_dimensions(cls, v):
        if v < 15:
            raise ValueError("Minimum dimension is 15 feet")
        return v

    @property
    def plot_area(self) -> float:
        """Calculate total plot area in square feet"""
        return self.plot_length * self.plot_width

    def get_open_walls(self) -> list[str]:
        """
        Get list of walls that are open (have road access / exterior exposure)

        LEARNING NOTE:
        This is important because:
        1. Windows can only be placed on OPEN walls (exterior walls)
        2. Closed walls are shared with neighbors - no windows there
        3. This affects natural light and ventilation in the floor plan

        Returns:
            List of open wall directions: ["north", "south", "east", "west"]
        """
        facing = self.facing_direction

        if self.plot_type == "open_all":
            # All 4 sides are open (standalone plot)
            return ["north", "south", "east", "west"]

        elif self.plot_type == "open_one":
            # Only the facing side is open (row house)
            return [facing]

        elif self.plot_type == "open_two_corner":
            # Facing + one adjacent side (corner plot)
            # Need to determine which adjacent side based on corner_open_side
            adjacent_walls = {
                "north": {"left": "west", "right": "east"},
                "south": {"left": "east", "right": "west"},
                "east": {"left": "north", "right": "south"},
                "west": {"left": "south", "right": "north"},
            }
            side = self.corner_open_side or "right"  # Default to right
            adjacent = adjacent_walls[facing][side]
            return [facing, adjacent]

        elif self.plot_type == "open_two_opposite":
            # Facing + opposite side (through plot)
            opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}
            return [facing, opposite[facing]]

        return [facing]  # Fallback


class Door(BaseModel):
    """Represents a door in the floor plan"""

    id: str = Field(..., description="Unique door identifier")
    x: float = Field(..., description="X position of door center")
    y: float = Field(..., description="Y position of door center")
    width: float = Field(default=3.0, description="Door width in feet")
    orientation: str = Field(..., description="horizontal or vertical")
    connects: list[str] = Field(default_factory=list, description="Room IDs this door connects")
    is_main_entrance: bool = Field(default=False, description="Is this the main entrance")


class Window(BaseModel):
    """Represents a window in the floor plan"""

    id: str = Field(..., description="Unique window identifier")
    x: float = Field(..., description="X position of window center")
    y: float = Field(..., description="Y position of window center")
    width: float = Field(default=4.0, description="Window width in feet")
    orientation: str = Field(..., description="horizontal or vertical")
    room_id: str = Field(..., description="Room this window belongs to")
    wall: str = Field(..., description="Which wall: north, south, east, west")


class Room(BaseModel):
    """Represents a room in the floor plan"""

    id: str = Field(..., description="Unique room identifier")
    room_type: str = Field(..., description="Type of room (bedroom, kitchen, etc.)")
    x: float = Field(..., description="X position (from left)")
    y: float = Field(..., description="Y position (from top)")
    width: float = Field(..., description="Room width")
    height: float = Field(..., description="Room height")
    area: float = Field(..., description="Room area in sq ft")
    zone: str = Field(..., description="Vastu zone (N, S, E, W, NE, etc.)")


class Layout(BaseModel):
    """Complete floor plan layout"""

    rooms: list[Room] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    plot_length: float
    plot_width: float
    facing_direction: str
    entrance_position: tuple[float, float] = Field(default=(0, 0))

    # NEW: Track which walls are open (for rendering and window placement)
    open_walls: list[str] = Field(
        default_factory=lambda: ["north", "south", "east", "west"],
        description="Walls that are open/exterior (can have windows)"
    )
    plot_type: str = Field(default="open_all", description="Type of plot configuration")


class VastuScore(BaseModel):
    """Vastu compliance score breakdown"""

    total: int = Field(..., ge=0, le=100, description="Overall compliance score")
    room_placement: int = Field(..., ge=0, le=100, description="Room placement score")
    entrance: int = Field(..., ge=0, le=100, description="Entrance placement score")
    center_clear: int = Field(..., ge=0, le=100, description="Brahmasthan clarity score")
    proportions: int = Field(..., ge=0, le=100, description="Room proportions score")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")


class GeneratedPlan(BaseModel):
    """A single generated floor plan"""

    plan_id: str = Field(..., description="Unique plan identifier")
    layout: Layout = Field(..., description="Room layout data")
    vastu_score: VastuScore = Field(..., description="Vastu compliance scores")
    image_base64: str = Field(..., description="Base64 encoded architectural floor plan image")
    image_format: str = Field(default="png", description="Image format")
    design_suggestions: list[str] = Field(
        default_factory=list,
        description="Architectural design improvement suggestions"
    )


class FloorPlanResponse(BaseModel):
    """API response for floor plan generation"""

    success: bool = Field(..., description="Whether generation was successful")
    message: str = Field(..., description="Status message")
    plans: list[GeneratedPlan] = Field(default_factory=list, description="Generated plans")
    request_summary: dict = Field(default_factory=dict, description="Summary of request params")
