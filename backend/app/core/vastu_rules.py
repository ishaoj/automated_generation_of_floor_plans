"""
Vastu Shastra rules database

This module contains all Vastu principles encoded as rules that can be used
for constraint solving and compliance scoring.
"""

from typing import Literal
from app.models.room_types import RoomType


# Direction type
DirectionType = Literal["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "center"]


# Room placement preferences based on Vastu Shastra
# Format: room_type -> {facing_direction: [preferred_zones]}
ROOM_PLACEMENT_RULES: dict[str, dict[str, list[str]]] = {
    RoomType.MASTER_BEDROOM: {
        "north": ["southwest", "south"],
        "south": ["southwest", "south"],
        "east": ["southwest", "south"],
        "west": ["southwest", "south"],
    },
    RoomType.BEDROOM: {
        "north": ["south", "west", "northwest"],
        "south": ["south", "west", "northwest"],
        "east": ["south", "west", "northwest"],
        "west": ["south", "west", "northwest"],
    },
    RoomType.LIVING_ROOM: {
        "north": ["north", "northeast", "east"],
        "south": ["north", "east", "northeast"],
        "east": ["north", "northeast", "east"],
        "west": ["north", "northeast", "east"],
    },
    RoomType.KITCHEN: {
        "north": ["southeast"],
        "south": ["southeast"],
        "east": ["southeast"],
        "west": ["southeast"],
    },
    RoomType.DINING: {
        "north": ["west", "east"],
        "south": ["west", "east"],
        "east": ["west", "east"],
        "west": ["west", "east"],
    },
    RoomType.BATHROOM: {
        "north": ["northwest", "west"],
        "south": ["northwest", "west"],
        "east": ["northwest", "west"],
        "west": ["northwest", "west"],
    },
    RoomType.PUJA_ROOM: {
        "north": ["northeast"],
        "south": ["northeast"],
        "east": ["northeast"],
        "west": ["northeast"],
    },
    RoomType.PARKING: {
        "north": ["northwest", "southeast"],
        "south": ["northwest", "southeast"],
        "east": ["northwest", "southeast"],
        "west": ["northwest", "southeast"],
    },
    RoomType.STAIRCASE: {
        "north": ["south", "southwest", "west"],
        "south": ["south", "southwest", "west"],
        "east": ["south", "southwest", "west"],
        "west": ["south", "southwest", "west"],
    },
    RoomType.STORE: {
        "north": ["south", "southwest"],
        "south": ["south", "southwest"],
        "east": ["south", "southwest"],
        "west": ["south", "southwest"],
    },
    # OTS (Open to Sky) - Courtyard placement based on Vastu
    # Center (Brahmasthan) is ideal, Northeast is also auspicious
    RoomType.OTS: {
        "north": ["center", "northeast"],  # North facing: center or NE
        "south": ["center", "northeast"],  # South facing: center or NE
        "east": ["center", "northeast"],   # East facing: center or NE
        "west": ["center", "northeast"],   # West facing: center or NE
    },
}

# Zones to avoid for each room type (forbidden placements)
ROOM_AVOIDANCE_RULES: dict[str, list[str]] = {
    RoomType.MASTER_BEDROOM: ["northeast"],
    RoomType.BEDROOM: ["southeast"],  # Kitchen zone
    RoomType.LIVING_ROOM: ["southwest"],  # Too heavy for living
    RoomType.KITCHEN: ["northeast", "north", "southwest"],  # Water/heavy zones
    RoomType.DINING: ["southeast"],  # Kitchen zone
    RoomType.BATHROOM: ["northeast", "center", "southwest"],  # Sacred/heavy zones
    RoomType.PUJA_ROOM: ["south", "southwest", "southeast"],  # Negative zones
    RoomType.PARKING: ["northeast"],  # Sacred zone
    RoomType.STAIRCASE: ["northeast", "center"],  # Sacred zones
    RoomType.STORE: ["northeast"],  # Sacred zone
    RoomType.OTS: ["southwest", "south"],  # OTS should not be in heavy/negative zones
}

# Entrance placement preferences based on facing direction
ENTRANCE_RULES = {
    "north": {
        "ideal_zones": ["north", "northeast"],
        "acceptable_zones": ["east"],
        "avoid_zones": ["south", "southwest", "west"],
        "best_position": "slightly east of center on north wall",
    },
    "south": {
        "ideal_zones": ["south"],
        "acceptable_zones": ["southeast"],
        "avoid_zones": ["southwest", "west"],
        "best_position": "slightly east of center on south wall",
    },
    "east": {
        "ideal_zones": ["east", "northeast"],
        "acceptable_zones": ["north"],
        "avoid_zones": ["south", "southwest", "west"],
        "best_position": "slightly north of center on east wall",
    },
    "west": {
        "ideal_zones": ["west", "northwest"],
        "acceptable_zones": ["north"],
        "avoid_zones": ["south", "southwest"],
        "best_position": "slightly north of center on west wall",
    },
}

# Center (Brahmasthan) rules
BRAHMASTHAN_RULES = {
    "must_be_open": True,  # Center should be open or have minimal construction
    "max_coverage_percent": 10,  # Maximum % of center that can be covered
    "allowed_uses": ["courtyard", "open_space", "light_well"],
    "forbidden_uses": ["bathroom", "kitchen", "staircase", "pillars", "walls"],
}

# Adjacency rules (which rooms should/shouldn't be adjacent)
ADJACENCY_RULES = {
    "preferred": [
        (RoomType.KITCHEN, RoomType.DINING),  # Kitchen should be near dining
        (RoomType.BEDROOM, RoomType.BATHROOM),  # Attached bathroom convenience
        (RoomType.LIVING_ROOM, RoomType.DINING),  # Common area connectivity
    ],
    "avoid": [
        (RoomType.KITCHEN, RoomType.BATHROOM),  # Hygiene concerns
        (RoomType.PUJA_ROOM, RoomType.BATHROOM),  # Sacred vs impure
        (RoomType.BEDROOM, RoomType.KITCHEN),  # Smoke/heat concerns
    ],
}

# Proportion and dimension rules
PROPORTION_RULES = {
    "plot_ratio_ideal": (1.0, 1.5),  # Length:Width ratio (square to slightly rectangular)
    "room_ratio_max": 2.0,  # Maximum L:W ratio for any room
    "bedroom_min_side": 10,  # Minimum side length in feet
    "bathroom_min_side": 5,
    "kitchen_min_side": 8,
}

# Weight/importance of each rule category for scoring
RULE_WEIGHTS = {
    "room_placement": 0.40,  # Room zone placement
    "entrance": 0.25,  # Entrance location
    "center_clear": 0.20,  # Brahmasthan clarity
    "proportions": 0.10,  # Room proportions
    "adjacency": 0.05,  # Room adjacency
}


def get_preferred_zones(room_type: str, facing: str) -> list[str]:
    """Get preferred Vastu zones for a room type given plot facing"""
    if room_type in ROOM_PLACEMENT_RULES:
        return ROOM_PLACEMENT_RULES[room_type].get(facing, [])
    return []


def get_avoided_zones(room_type: str) -> list[str]:
    """Get zones to avoid for a room type"""
    return ROOM_AVOIDANCE_RULES.get(room_type, [])


def get_vastu_rules() -> dict:
    """
    Get all Vastu rules in a structured format for API response
    """
    return {
        "room_placement": {
            room_type: {
                "preferred_zones": zones,
                "avoid_zones": ROOM_AVOIDANCE_RULES.get(room_type, []),
            }
            for room_type, zones in ROOM_PLACEMENT_RULES.items()
        },
        "entrance": ENTRANCE_RULES,
        "center": BRAHMASTHAN_RULES,
        "adjacency": {
            "preferred": [
                {"rooms": list(pair), "reason": "Functional convenience"}
                for pair in ADJACENCY_RULES["preferred"]
            ],
            "avoid": [
                {"rooms": list(pair), "reason": "Vastu principle"}
                for pair in ADJACENCY_RULES["avoid"]
            ],
        },
        "weights": RULE_WEIGHTS,
        "general_guidelines": [
            "Main entrance should ideally face North or East",
            "Kitchen should be in Southeast (Agni corner)",
            "Master bedroom should be in Southwest",
            "Puja room should be in Northeast (Ishan corner)",
            "Toilets should never be in Northeast",
            "Center of house (Brahmasthan) should be open",
            "Staircase should be in South, West, or Southwest",
            "Living room should be in North or East",
            "Avoid sleeping with head towards North",
            "Cook should face East while cooking",
        ],
    }


# Zone scoring matrix (how well each zone matches ideal placement)
# Score from 0-100 based on zone match
ZONE_SCORE_MATRIX = {
    # If ideal zone is northeast
    "northeast": {"northeast": 100, "north": 70, "east": 70, "center": 40, "northwest": 30, "southeast": 30, "west": 20, "south": 10, "southwest": 0},
    "southeast": {"southeast": 100, "south": 70, "east": 70, "center": 40, "northeast": 30, "southwest": 30, "west": 20, "north": 10, "northwest": 0},
    "southwest": {"southwest": 100, "south": 70, "west": 70, "center": 40, "southeast": 30, "northwest": 30, "east": 20, "north": 10, "northeast": 0},
    "northwest": {"northwest": 100, "north": 70, "west": 70, "center": 40, "northeast": 30, "southwest": 30, "east": 20, "south": 10, "southeast": 0},
    "north": {"north": 100, "northeast": 80, "northwest": 80, "center": 50, "east": 40, "west": 40, "south": 20, "southeast": 20, "southwest": 20},
    "south": {"south": 100, "southeast": 80, "southwest": 80, "center": 50, "east": 40, "west": 40, "north": 20, "northeast": 20, "northwest": 20},
    "east": {"east": 100, "northeast": 80, "southeast": 80, "center": 50, "north": 40, "south": 40, "west": 20, "northwest": 20, "southwest": 20},
    "west": {"west": 100, "northwest": 80, "southwest": 80, "center": 50, "north": 40, "south": 40, "east": 20, "northeast": 20, "southeast": 20},
    "center": {"center": 100, "north": 60, "south": 60, "east": 60, "west": 60, "northeast": 40, "northwest": 40, "southeast": 40, "southwest": 40},
}


def calculate_zone_match_score(ideal_zone: str, actual_zone: str) -> int:
    """Calculate how well actual placement matches ideal zone"""
    if ideal_zone in ZONE_SCORE_MATRIX:
        return ZONE_SCORE_MATRIX[ideal_zone].get(actual_zone, 0)
    return 50  # Default middle score for unknown zones
