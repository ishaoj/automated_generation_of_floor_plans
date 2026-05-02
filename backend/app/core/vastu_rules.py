"""
Vastu Shastra rules database — comprehensive rule set.

Encodes the traditional 9-zone directional principles as machine-readable
constraints for the CP-SAT solver, layout validator, and Vastu scorer.

Zone elements:
  NE  — Water  : purity, clarity, must be lightest/most open
  SE  — Fire   : heat, energy → kitchen
  SW  — Earth  : stability, heaviness → master bedroom, maximum weight
  NW  — Air    : movement, transition → bathrooms, guest areas
  N   — Water  : prosperity, openness → living/entrance
  E   — Air    : health, morning sun → living, balcony
  S   — Fire   : protection (heavy walls S/W)
  W   — Earth  : stability, storage
  Center (Brahmasthan) — Space : must be completely empty
"""

from typing import Literal
from app.models.room_types import RoomType


DirectionType = Literal[
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest", "center"
]


# ── Room placement preferences ────────────────────────────────────────────────
# Format: room_type → {facing_direction: [preferred_zones, in priority order]}

ROOM_PLACEMENT_RULES: dict[str, dict[str, list[str]]] = {

    # SW = earth/stability.  Strictly avoid NE (water/purity) and SE (fire).
    RoomType.MASTER_BEDROOM: {
        "north": ["southwest", "south"],
        "south": ["southwest", "south"],
        "east":  ["southwest", "south"],
        "west":  ["southwest", "south"],
    },

    # Secondary bedrooms: NW (air/transition) or W/S; keep away from SE (fire).
    RoomType.BEDROOM: {
        "north": ["northwest", "south", "west"],
        "south": ["northwest", "south", "west"],
        "east":  ["northwest", "south", "west"],
        "west":  ["northwest", "south", "west"],
    },

    # N/NE/E — morning light, public access from entrance.
    RoomType.LIVING_ROOM: {
        "north": ["north", "northeast", "east"],
        "south": ["north", "northeast", "east"],
        "east":  ["north", "northeast", "east"],
        "west":  ["north", "northeast", "east"],
    },

    # SE (fire/agni) is primary.  NW is secondary (movement of air = cooking exhaust).
    # Strictly avoid NE (water/purity), SW (earth heaviness), and center.
    RoomType.KITCHEN: {
        "north": ["southeast", "northwest"],
        "south": ["southeast", "northwest"],
        "east":  ["southeast", "northwest"],
        "west":  ["southeast", "northwest"],
    },

    # Dining connects living and kitchen; W or E works well.
    RoomType.DINING: {
        "north": ["west", "east"],
        "south": ["west", "east"],
        "east":  ["west", "east"],
        "west":  ["west", "east"],
    },

    # NW (air/movement) and W are the correct bathroom zones.
    # Strictly avoid NE (purity), center (Brahmasthan), SW (earth/stability).
    RoomType.BATHROOM: {
        "north": ["northwest", "west"],
        "south": ["northwest", "west"],
        "east":  ["northwest", "west"],
        "west":  ["northwest", "west"],
    },

    # NE (Ishan/water) — the zone of highest purity. Only acceptable zone.
    RoomType.PUJA_ROOM: {
        "north": ["northeast"],
        "south": ["northeast"],
        "east":  ["northeast"],
        "west":  ["northeast"],
    },

    # Parking near entrance: NW or SE (vehicle entry, transition zone).
    RoomType.PARKING: {
        "north": ["northwest", "southeast"],
        "south": ["northwest", "southeast"],
        "east":  ["northwest", "southeast"],
        "west":  ["northwest", "southeast"],
    },

    # Staircase: S, SW, or W only.  Must never be NE or center.
    RoomType.STAIRCASE: {
        "north": ["south", "southwest", "west"],
        "south": ["south", "southwest", "west"],
        "east":  ["south", "southwest", "west"],
        "west":  ["south", "southwest", "west"],
    },

    # Balcony: N, E, or NE — open sides for morning light and air.
    RoomType.BALCONY: {
        "north": ["north", "northeast", "east"],
        "south": ["north", "northeast", "east"],
        "east":  ["north", "northeast", "east"],
        "west":  ["north", "northeast", "east"],
    },

    # Store: S or SW (heavy/stable zone).
    RoomType.STORE: {
        "north": ["south", "southwest"],
        "south": ["south", "southwest"],
        "east":  ["south", "southwest"],
        "west":  ["south", "southwest"],
    },

    # OTS (Open to Sky / Brahmasthan): center is mandatory, NE is secondary.
    RoomType.OTS: {
        "north": ["center", "northeast"],
        "south": ["center", "northeast"],
        "east":  ["center", "northeast"],
        "west":  ["center", "northeast"],
    },
}


# ── Zone avoidance rules ──────────────────────────────────────────────────────
# Rooms placed in these zones receive a heavy score penalty (near zero).

ROOM_AVOIDANCE_RULES: dict[str, list[str]] = {
    RoomType.MASTER_BEDROOM: ["northeast", "southeast"],   # Disrupts sleep; too fiery
    RoomType.BEDROOM:        ["southeast"],                 # Kitchen/fire zone
    RoomType.LIVING_ROOM:    ["southwest"],                 # Heavy/stable zone, not social
    RoomType.KITCHEN:        ["northeast", "southwest", "center"],  # Purity + earth zones
    RoomType.DINING:         ["southeast"],                 # Kitchen/fire zone
    RoomType.BATHROOM:       ["northeast", "center", "southwest"],  # Critical: pollutes sacred zones
    RoomType.PUJA_ROOM:      ["south", "southwest", "southeast"],   # Negative/heavy zones
    RoomType.PARKING:        ["northeast"],                 # Sacred zone
    RoomType.STAIRCASE:      ["northeast", "center"],       # Sacred zones; blocks Brahmasthan
    RoomType.BALCONY:        ["southwest", "south"],        # Exposed to harsh afternoon sun/wind
    RoomType.STORE:          ["northeast"],                 # Sacred zone
    RoomType.OTS:            ["southwest", "south"],        # Must not be in heavy zones
}


# ── Entrance placement rules ──────────────────────────────────────────────────
# Per facing direction: ideal > acceptable > avoid

ENTRANCE_RULES = {
    "north": {
        "ideal_zones":      ["north", "northeast"],
        "acceptable_zones": ["east"],
        "avoid_zones":      ["south", "southwest", "northwest", "west"],
        "best_position":    "slightly east of center on north wall",
        "reason":           "Aligns with earth's magnetic field; maximises morning UV exposure",
    },
    "south": {
        "ideal_zones":      ["south"],
        "acceptable_zones": ["southeast"],
        "avoid_zones":      ["southwest", "west", "northwest"],
        "best_position":    "slightly east of center on south wall",
        "reason":           "South entrance acceptable when slightly east to reduce negative Vastu",
    },
    "east": {
        "ideal_zones":      ["east", "northeast"],
        "acceptable_zones": ["north"],
        "avoid_zones":      ["south", "southwest", "west"],
        "best_position":    "slightly north of center on east wall",
        "reason":           "East entrance captures early morning sunlight (auspicious)",
    },
    "west": {
        "ideal_zones":      ["west", "northwest"],
        "acceptable_zones": ["north"],
        "avoid_zones":      ["south", "southwest", "southeast"],
        "best_position":    "slightly north of center on west wall",
        "reason":           "NW sector of west wall is the acceptable transition zone",
    },
}


# ── Brahmasthan (centre) rules ────────────────────────────────────────────────

BRAHMASTHAN_RULES = {
    "must_be_open":        True,
    "max_coverage_percent": 10,   # Max % of center zone that any room may occupy
    "allowed_uses":        ["courtyard", "open_space", "ots", "light_well"],
    "forbidden_uses":      ["bathroom", "kitchen", "staircase", "bedroom",
                            "store", "pillars", "load_bearing_walls"],
    "reason":              "Center is the Brahmasthan — spatial/energy core of the house. "
                           "Blocking it traps negative energy and cuts ventilation.",
}


# ── Adjacency rules ───────────────────────────────────────────────────────────

ADJACENCY_RULES = {
    "preferred": [
        (RoomType.KITCHEN,    RoomType.DINING),      # Unobstructed food service path
        (RoomType.BEDROOM,    RoomType.BATHROOM),     # Every bedroom needs bath access
        (RoomType.LIVING_ROOM, RoomType.DINING),      # Social zone connectivity
    ],
    "avoid": [
        (RoomType.KITCHEN,    RoomType.BATHROOM),     # Hygiene — wet zones don't mix
        (RoomType.PUJA_ROOM,  RoomType.BATHROOM),     # Sacred vs impure; critical Vastu rule
        (RoomType.BEDROOM,    RoomType.KITCHEN),      # Smoke, heat, smell contamination
        (RoomType.PUJA_ROOM,  RoomType.PARKING),      # Fumes and sacred space incompatible
    ],
}


# ── Privacy gradient zones ────────────────────────────────────────────────────
# Defines the public→semi-public→private layering of a well-designed house.
# Used by VastuScorer._score_privacy_gradient().

PRIVACY_ZONES = {
    "public": {
        "room_types": {"parking", "entrance", "lobby", "foyer", "porch"},
        "description": "First point of contact; should be closest to the main entrance wall",
        "depth_from_entrance": (0.0, 0.35),   # Within first 35% of plot depth
    },
    "semi_public": {
        "room_types": {"living_room", "dining", "kitchen"},
        "description": "Social/family zone; accessible to visitors but not through private areas",
        "depth_from_entrance": (0.20, 0.70),  # Middle band of plot
    },
    "private": {
        "room_types": {"master_bedroom", "bedroom", "bathroom", "puja_room"},
        "description": "Sleeping and personal spaces; must be deepest in the plan",
        "depth_from_entrance": (0.50, 1.00),  # Back half of plot
    },
}


# ── Window orientation rules ──────────────────────────────────────────────────

WINDOW_RULES = {
    "large_openings":  ["north", "east"],       # Diffused light; morning sun (N/E walls)
    "small_openings":  ["south", "west"],        # Block harsh afternoon sun/heat
    "reason":          "N/E walls receive indirect and morning light; S/W receive harsh "
                       "afternoon sun. Larger openings on S/W create heat gain and glare.",
}


# ── Opening count rule ────────────────────────────────────────────────────────

EVEN_OPENINGS_RULE = {
    "total_must_be_even":         True,
    "avoid_multiples_of":         10,
    "reason":                     "Total doors + windows should be an even number "
                                  "(2, 4, 6, 8, 12, …) but never a multiple of 10.",
}


# ── Structural weight rules ───────────────────────────────────────────────────

STRUCTURAL_RULES = {
    "thick_walls":  ["south", "west"],     # Heavier walls block afternoon heat
    "light_walls":  ["north", "east"],     # Lighter walls maximise openings and light
    "balcony_walls": ["north", "east", "northeast"],  # Balconies face open/airy sides
    "plot_ratio":   {
        "allowed":  (1.0, 2.0),            # Length:Width max 1:2
        "ideal":    (1.0, 1.5),
        "reason":   "Irregular or overly elongated plots cause unequal energy distribution",
    },
}


# ── Proportion rules ──────────────────────────────────────────────────────────

PROPORTION_RULES = {
    "plot_ratio_ideal":   (1.0, 1.5),
    "room_ratio_max":     2.0,
    "bedroom_min_side":   10,
    "bathroom_min_side":   5,
    "kitchen_min_side":    8,
    "corridor_min_width":  3.5,
    "min_ceiling_height":  9,
}


# ── Rule weights for scoring ──────────────────────────────────────────────────

RULE_WEIGHTS = {
    "room_placement":   0.35,
    "entrance":         0.20,
    "center_clear":     0.20,
    "privacy_gradient": 0.15,
    "proportions":      0.10,
}


# ── Zone scoring matrix ───────────────────────────────────────────────────────
# Score 0–100 for how well the actual zone matches the ideal zone.
# Adjacent zones receive partial credit.

ZONE_SCORE_MATRIX = {
    "northeast": {
        "northeast": 100, "north": 70, "east": 70, "center": 40,
        "northwest": 30,  "southeast": 30, "west": 20, "south": 10, "southwest": 0,
    },
    "southeast": {
        "southeast": 100, "south": 70, "east": 70, "center": 40,
        "northeast": 30,  "southwest": 30, "west": 20, "north": 10, "northwest": 0,
    },
    "southwest": {
        "southwest": 100, "south": 70, "west": 70, "center": 40,
        "southeast": 30,  "northwest": 30, "east": 20, "north": 10, "northeast": 0,
    },
    "northwest": {
        "northwest": 100, "north": 70, "west": 70, "center": 40,
        "northeast": 30,  "southwest": 30, "east": 20, "south": 10, "southeast": 0,
    },
    "north": {
        "north": 100, "northeast": 80, "northwest": 80, "center": 50,
        "east": 40,   "west": 40,      "south": 20,    "southeast": 20, "southwest": 20,
    },
    "south": {
        "south": 100, "southeast": 80, "southwest": 80, "center": 50,
        "east": 40,   "west": 40,      "north": 20,    "northeast": 20, "northwest": 20,
    },
    "east": {
        "east": 100, "northeast": 80, "southeast": 80, "center": 50,
        "north": 40, "south": 40,     "west": 20,     "northwest": 20, "southwest": 20,
    },
    "west": {
        "west": 100, "northwest": 80, "southwest": 80, "center": 50,
        "north": 40, "south": 40,     "east": 20,     "northeast": 20, "southeast": 20,
    },
    "center": {
        "center": 100, "north": 60, "south": 60, "east": 60, "west": 60,
        "northeast": 40, "northwest": 40, "southeast": 40, "southwest": 40,
    },
}


# ── Helper functions ──────────────────────────────────────────────────────────

def get_preferred_zones(room_type: str, facing: str) -> list[str]:
    if room_type in ROOM_PLACEMENT_RULES:
        return ROOM_PLACEMENT_RULES[room_type].get(facing, [])
    return []


def get_avoided_zones(room_type: str) -> list[str]:
    return ROOM_AVOIDANCE_RULES.get(room_type, [])


def calculate_zone_match_score(ideal_zone: str, actual_zone: str) -> int:
    if ideal_zone in ZONE_SCORE_MATRIX:
        return ZONE_SCORE_MATRIX[ideal_zone].get(actual_zone, 0)
    return 50


def get_vastu_rules() -> dict:
    """Return all Vastu rules in structured form for the API /vastu-rules endpoint."""
    return {
        "room_placement": {
            room_type: {
                "preferred_zones": zones,
                "avoid_zones":     ROOM_AVOIDANCE_RULES.get(room_type, []),
            }
            for room_type, zones in ROOM_PLACEMENT_RULES.items()
        },
        "entrance":   ENTRANCE_RULES,
        "center":     BRAHMASTHAN_RULES,
        "privacy":    PRIVACY_ZONES,
        "windows":    WINDOW_RULES,
        "openings":   EVEN_OPENINGS_RULE,
        "structural": STRUCTURAL_RULES,
        "adjacency": {
            "preferred": [
                {"rooms": list(pair), "reason": "Functional convenience"}
                for pair in ADJACENCY_RULES["preferred"]
            ],
            "avoid": [
                {"rooms": list(pair), "reason": "Vastu/hygiene principle"}
                for pair in ADJACENCY_RULES["avoid"]
            ],
        },
        "weights":  RULE_WEIGHTS,
        "general_guidelines": [
            "Main entrance: North, East, or North-East (NE is ideal)",
            "Kitchen: South-East (Agni corner); secondary option North-West",
            "Kitchen strictly forbidden in NE, SW, or Center",
            "Master bedroom: South-West; avoid NE (magnetic disruption) and SE (fire zone)",
            "Puja/Prayer room: North-East (Ishan corner) exclusively",
            "Bathrooms: North-West or West; never NE, Center, or SW",
            "Center (Brahmasthan): must remain completely empty — no walls, pillars, or heavy rooms",
            "Staircase: South, South-West, or West; never NE or Center",
            "Balconies: North, East, or North-East only",
            "Windows: larger openings on N/E walls; smaller openings on S/W walls",
            "Total doors + windows must be an even number, not a multiple of 10",
            "Layout should transition Public → Semi-public → Private from entrance inward",
            "Cook should face East while cooking (kitchen in SE achieves this naturally)",
            "Bathrooms and kitchen must never share a common wall (hygiene + Vastu)",
        ],
    }
