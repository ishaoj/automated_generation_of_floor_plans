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

        # Calculate individual scores
        room_placement_score = self._score_room_placement(layout.rooms, facing)
        entrance_score = self._score_entrance(layout, facing)
        center_score = self._score_center(layout)
        proportion_score = self._score_proportions(layout.rooms)

        # Weighted total
        total = int(
            room_placement_score * RULE_WEIGHTS["room_placement"] +
            entrance_score * RULE_WEIGHTS["entrance"] +
            center_score * RULE_WEIGHTS["center_clear"] +
            proportion_score * RULE_WEIGHTS["proportions"]
        ) * 100 // (
            RULE_WEIGHTS["room_placement"] +
            RULE_WEIGHTS["entrance"] +
            RULE_WEIGHTS["center_clear"] +
            RULE_WEIGHTS["proportions"]
        )

        # Generate improvement suggestions
        suggestions = self._generate_suggestions(
            layout, room_placement_score, entrance_score, center_score
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
        center_score: int
    ) -> list[str]:
        """Generate improvement suggestions based on scores"""
        suggestions = []

        if room_score < 60:
            # Find poorly placed rooms
            for room in layout.rooms:
                preferred = get_preferred_zones(room.room_type, layout.facing_direction)
                if preferred and room.zone not in preferred:
                    room_name = room.room_type.replace("_", " ").title()
                    suggestions.append(
                        f"Consider moving {room_name} to {'/'.join(preferred[:2]).upper()} zone"
                    )

        if entrance_score < 70:
            rules = ENTRANCE_RULES.get(layout.facing_direction, {})
            ideal = rules.get("ideal_zones", [])
            if ideal:
                suggestions.append(
                    f"Main entrance would be better positioned in {ideal[0].upper()} zone"
                )

        if center_score < 70:
            suggestions.append(
                "Try to keep the center (Brahmasthan) more open for better energy flow"
            )

        # Limit suggestions
        return suggestions[:5]
