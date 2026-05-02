"""
Vastu compliance scorer - calculates how well a layout follows Vastu principles
"""

from statistics import mean
from app.models.schemas import Layout, VastuScore, Room
from app.core.vastu_rules import (
    get_preferred_zones,
    get_avoided_zones,
    ZONE_SCORE_MATRIX,
    ENTRANCE_RULES,
    BRAHMASTHAN_RULES,
    PRIVACY_ZONES,
    RULE_WEIGHTS,
)
from app.core.constants import ZONES, Direction


class VastuScorer:
    """
    Calculates Vastu compliance scores for floor plan layouts

    Evaluates:
    1. Room placement in correct zones
    2. Entrance location
    3. Center (Brahmasthan) clarity
    4. Room proportions
    """

    def score(self, layout: Layout) -> VastuScore:
        """
        Calculate comprehensive Vastu score for a layout

        Args:
            layout: Floor plan layout to evaluate

        Returns:
            VastuScore with total and breakdown
        """
        facing = layout.facing_direction

        room_placement_score = self._score_room_placement(layout.rooms, facing)
        entrance_score       = self._score_entrance(layout, facing)
        center_score         = self._score_center(layout)
        proportion_score     = self._score_proportions(layout.rooms)
        privacy_score        = self._score_privacy_gradient(layout)

        # Weighted total (room placement 35%, entrance 20%, center 20%,
        # privacy gradient 15%, proportions 10%)
        total = int(
            room_placement_score * 0.35
            + entrance_score       * 0.20
            + center_score         * 0.20
            + privacy_score        * 0.15
            + proportion_score     * 0.10
        )

        suggestions = self._generate_suggestions(
            layout, room_placement_score, entrance_score, center_score, privacy_score
        )

        return VastuScore(
            total=min(100, max(0, total)),
            room_placement=room_placement_score,
            entrance=entrance_score,
            center_clear=center_score,
            proportions=proportion_score,
            suggestions=suggestions,
        )

    def _score_room_placement(self, rooms: list[Room], facing: str) -> int:
        """Score how well rooms are placed in their ideal zones"""
        if not rooms:
            return 50

        scores = []

        for room in rooms:
            preferred = get_preferred_zones(room.room_type, facing)
            avoided = get_avoided_zones(room.room_type)
            actual_zone = room.zone

            if not preferred:
                scores.append(70)  # No specific preference
                continue

            # Check if in avoided zone (penalty)
            if actual_zone in avoided:
                scores.append(20)
                continue

            # Check match with preferred zones
            best_score = 0
            for pref_zone in preferred:
                if pref_zone in ZONE_SCORE_MATRIX:
                    zone_score = ZONE_SCORE_MATRIX[pref_zone].get(actual_zone, 30)
                    best_score = max(best_score, zone_score)

            scores.append(best_score if best_score > 0 else 50)

        return int(mean(scores)) if scores else 50

    def _score_entrance(self, layout: Layout, facing: str) -> int:
        """Score entrance placement according to Vastu"""
        rules = ENTRANCE_RULES.get(facing, {})

        if not rules:
            return 70

        # Determine entrance zone from position
        entrance_zone = self._get_zone_from_position(
            layout.entrance_position[0],
            layout.entrance_position[1],
            layout.plot_width,
            layout.plot_length
        )

        ideal_zones = rules.get("ideal_zones", [])
        acceptable_zones = rules.get("acceptable_zones", [])
        avoid_zones = rules.get("avoid_zones", [])

        if entrance_zone in ideal_zones:
            return 100
        elif entrance_zone in acceptable_zones:
            return 75
        elif entrance_zone in avoid_zones:
            return 25
        else:
            return 50

    def _score_center(self, layout: Layout) -> int:
        """Score how clear the center (Brahmasthan) is"""
        # Get center zone bounds
        center_bounds = ZONES[Direction.CENTER]
        cx_min = center_bounds[0] * layout.plot_width
        cy_min = center_bounds[1] * layout.plot_length
        cx_max = center_bounds[2] * layout.plot_width
        cy_max = center_bounds[3] * layout.plot_length

        center_area = (cx_max - cx_min) * (cy_max - cy_min)

        # Calculate how much of center is occupied by rooms
        occupied_area = 0

        for room in layout.rooms:
            # Calculate intersection with center
            ix_min = max(room.x, cx_min)
            iy_min = max(room.y, cy_min)
            ix_max = min(room.x + room.width, cx_max)
            iy_max = min(room.y + room.height, cy_max)

            if ix_max > ix_min and iy_max > iy_min:
                occupied_area += (ix_max - ix_min) * (iy_max - iy_min)

        coverage_percent = (occupied_area / center_area) * 100 if center_area > 0 else 0
        max_allowed = BRAHMASTHAN_RULES["max_coverage_percent"]

        if coverage_percent <= max_allowed:
            return 100
        elif coverage_percent <= max_allowed * 2:
            return 70
        elif coverage_percent <= max_allowed * 4:
            return 40
        else:
            return 20

    def _score_proportions(self, rooms: list[Room]) -> int:
        """Score room proportions (should be close to square)"""
        if not rooms:
            return 70

        scores = []

        for room in rooms:
            if room.width <= 0 or room.height <= 0:
                continue

            ratio = max(room.width, room.height) / min(room.width, room.height)

            # Ideal ratio is 1.0-1.5, max acceptable is 2.0
            if ratio <= 1.2:
                scores.append(100)
            elif ratio <= 1.5:
                scores.append(85)
            elif ratio <= 2.0:
                scores.append(60)
            else:
                scores.append(30)

        return int(mean(scores)) if scores else 70

    def _score_privacy_gradient(self, layout: Layout) -> int:
        """
        Score how well the layout follows the Public → Semi-public → Private gradient.

        A well-designed house layers rooms from the entrance inward:
          Public (0–35%): parking, entrance, foyer
          Semi-public (20–70%): living room, dining, kitchen
          Private (50–100%): bedrooms, bathrooms, puja room

        Scoring:
          Each room that lands in its correct depth band scores 100.
          Each room that is in the wrong band scores 0 (or 50 for adjacent bands).

        "Depth" is measured as distance from the entrance wall, normalised 0–1.
        Facing north → depth = y / plot_length (y=0 is the entrance side)
        Facing south → depth = (pl - y - h) / pl  (y=pl is entrance side)
        Facing east  → depth = (pw - x - w) / pw
        Facing west  → depth = x / pw
        """
        if not layout.rooms:
            return 50

        pl = layout.plot_length
        pw = layout.plot_width
        facing = layout.facing_direction

        def depth(room: Room) -> float:
            cx = room.x + room.width  / 2
            cy = room.y + room.height / 2
            if facing == "north":
                return cy / pl
            elif facing == "south":
                return (pl - cy) / pl
            elif facing == "east":
                return (pw - cx) / pw
            else:  # west
                return cx / pw

        scores: list[int] = []
        for room in layout.rooms:
            rt = room.room_type.lower()
            d  = depth(room)

            # Determine which privacy zone this room belongs to
            if any(t in rt for t in PRIVACY_ZONES["public"]["room_types"]):
                lo, hi = PRIVACY_ZONES["public"]["depth_from_entrance"]
                # Slightly generous tolerance
                if lo <= d <= hi + 0.15:
                    scores.append(100)
                else:
                    scores.append(max(0, int(100 - abs(d - hi) * 200)))

            elif any(t in rt for t in PRIVACY_ZONES["semi_public"]["room_types"]):
                lo, hi = PRIVACY_ZONES["semi_public"]["depth_from_entrance"]
                if lo <= d <= hi:
                    scores.append(100)
                else:
                    scores.append(max(0, int(100 - min(abs(d - lo), abs(d - hi)) * 150)))

            elif any(t in rt for t in PRIVACY_ZONES["private"]["room_types"]):
                lo, hi = PRIVACY_ZONES["private"]["depth_from_entrance"]
                if lo <= d <= hi:
                    scores.append(100)
                else:
                    scores.append(max(0, int(100 - abs(d - lo) * 200)))

            # Rooms with no privacy zone classification (balcony, staircase, ots, store)
            # are neutral — don't penalise them
        return int(mean(scores)) if scores else 70

    def _get_zone_from_position(self, x: float, y: float, plot_width: float, plot_length: float) -> str:
        """Determine Vastu zone from position"""
        nx = x / plot_width if plot_width > 0 else 0.5
        ny = y / plot_length if plot_length > 0 else 0.5

        # Clamp to valid range
        nx = max(0, min(1, nx))
        ny = max(0, min(1, ny))

        if ny < 0.33:
            if nx < 0.33:
                return "northwest"
            elif nx > 0.67:
                return "northeast"
            else:
                return "north"
        elif ny > 0.67:
            if nx < 0.33:
                return "southwest"
            elif nx > 0.67:
                return "southeast"
            else:
                return "south"
        else:
            if nx < 0.33:
                return "west"
            elif nx > 0.67:
                return "east"
            else:
                return "center"

    def _generate_suggestions(
        self,
        layout: Layout,
        room_score: int,
        entrance_score: int,
        center_score: int,
        privacy_score: int = 70,
    ) -> list[str]:
        """
        Generate specific, prioritised Vastu improvement suggestions.

        Checks are ordered by severity — the most critical Vastu violations
        (NE kitchen, NE bathroom, etc.) appear first.
        """
        suggestions: list[str] = []
        facing = layout.facing_direction

        # ── Critical zone violations ──────────────────────────────────────────
        for room in layout.rooms:
            rt   = room.room_type
            zone = room.zone
            name = rt.replace("_", " ").title()

            # Kitchen in NE — most severe violation
            if "kitchen" in rt and zone in {"northeast"}:
                suggestions.append(
                    f"Critical: Kitchen is in NE (Water/Purity zone) — fire element "
                    f"in the purity zone is the most severe Vastu violation. "
                    f"Move kitchen to SE (ideal) or NW."
                )

            # Kitchen in SW
            elif "kitchen" in rt and zone == "southwest":
                suggestions.append(
                    f"Kitchen is in SW (Earth zone) — reserved for master bedroom stability. "
                    f"Relocate to SE (Agni/Fire zone) for cooking alignment."
                )

            # Master bedroom in NE
            elif rt == "master_bedroom" and zone == "northeast":
                suggestions.append(
                    f"Master bedroom in NE disrupts sleep — the water element and magnetic "
                    f"field alignment cause restlessness. Move to SW for grounding stability."
                )

            # Master bedroom in SE
            elif rt == "master_bedroom" and zone == "southeast":
                suggestions.append(
                    f"Master bedroom in SE (Fire zone) creates heat and restlessness. "
                    f"Move to SW (Earth/Stability) for calm, restorative sleep."
                )

            # Bathroom in NE — contaminates purity zone
            elif "bathroom" in rt and zone == "northeast":
                suggestions.append(
                    f"Bathroom in NE (Purity zone) is a critical Vastu violation — it "
                    f"contaminates the highest-purity zone. Relocate to NW or W."
                )

            # Bathroom in SW
            elif "bathroom" in rt and zone in {"southwest", "center"}:
                suggestions.append(
                    f"Bathroom in {zone.upper()} is forbidden — this zone must remain "
                    f"pure (Brahmasthan) or stable (SW for master bedroom). Move to NW/W."
                )

            # Puja room not in NE
            elif rt == "puja_room" and zone != "northeast":
                preferred = get_preferred_zones(rt, facing)
                suggestions.append(
                    f"Puja room in {zone.upper()} — the prayer room must exclusively occupy "
                    f"the NE (Ishan corner), the highest-purity zone in Vastu."
                )

        # ── Entrance placement ────────────────────────────────────────────────
        if entrance_score < 70:
            rules = ENTRANCE_RULES.get(facing, {})
            ideal = rules.get("ideal_zones", [])
            reason = rules.get("reason", "")
            if ideal:
                suggestions.append(
                    f"Entrance: best positioned in {' or '.join(z.upper() for z in ideal)} zone. "
                    f"{reason}"
                )

        # ── Centre / Brahmasthan ──────────────────────────────────────────────
        if center_score < 70:
            suggestions.append(
                "Brahmasthan (center) is partially blocked — this is the Space element "
                "core of the house. Keep it completely open or place only an OTS "
                "(Open to Sky) courtyard here for energy flow and natural ventilation."
            )

        # ── General misplaced rooms (lower priority) ──────────────────────────
        if room_score < 60 and len(suggestions) < 4:
            for room in layout.rooms:
                preferred = get_preferred_zones(room.room_type, facing)
                avoided   = get_avoided_zones(room.room_type)
                zone      = room.zone
                name      = room.room_type.replace("_", " ").title()
                if preferred and zone not in preferred and zone not in avoided:
                    suggestions.append(
                        f"{name} is in {zone.upper()} zone — preferred: "
                        f"{' or '.join(z.upper() for z in preferred[:2])}."
                    )
                    if len(suggestions) >= 4:
                        break

        # ── Privacy gradient ─────────────────────────────────────────────────
        if privacy_score < 60:
            suggestions.append(
                "Layout does not follow the Public → Semi-public → Private gradient. "
                "Public rooms (parking, entrance) should be near the entrance wall; "
                "private rooms (bedrooms, bathrooms) should be deepest in the plan."
            )

        return suggestions[:5]
