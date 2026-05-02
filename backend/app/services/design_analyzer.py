"""
Architectural design quality analyzer.

Checks generated layouts against practical design principles —
circulation, ventilation, privacy, dimension quality — and returns
human-readable improvement suggestions shown in the output UI.

These are distinct from Vastu compliance suggestions: they are
architectural best-practice observations, not rule violations.
"""

from app.models.schemas import Layout, Room


class DesignAnalyzer:
    """
    Analyzes a Layout for architectural quality issues and returns
    a list of improvement suggestions (up to 6).
    """

    # Minimum bedroom dimension (ft) — narrower rooms can't fit standard furniture
    _MIN_BEDROOM_DIM = 10.0

    # Minimum shared wall (ft) considered meaningful acoustic separation
    _MIN_ACOUSTIC_GAP = 3.0

    def analyze(self, layout: Layout) -> list[str]:
        suggestions: list[str] = []

        self._check_corridor_absence(layout, suggestions)
        self._check_kitchen_exterior(layout, suggestions)
        self._check_bedroom_clustering(layout, suggestions)
        self._check_living_room_exterior(layout, suggestions)
        self._check_bedroom_dimensions(layout, suggestions)
        self._check_acoustic_bedroom_gap(layout, suggestions)

        return suggestions[:6]

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_corridor_absence(self, layout: Layout, suggestions: list[str]) -> None:
        """With 3+ bedrooms and no corridor, all bedrooms open into the living room."""
        bedrooms = [r for r in layout.rooms if "bedroom" in r.room_type]
        has_corridor = any(
            r.room_type in {"corridor", "entrance", "lobby"}
            for r in layout.rooms
        )
        if len(bedrooms) >= 3 and not has_corridor:
            suggestions.append(
                "Add a corridor: With 3+ bedrooms all opening directly into the living "
                "room, add a dedicated corridor (min 3.5 ft wide) to create a private "
                "bedroom wing. This prevents direct sightlines from the living area into "
                "bedrooms and reduces through-traffic in the social space."
            )

    def _check_kitchen_exterior(self, layout: Layout, suggestions: list[str]) -> None:
        """Kitchen should touch at least one exterior wall for ventilation."""
        kitchen = next((r for r in layout.rooms if "kitchen" in r.room_type), None)
        if kitchen is None:
            return
        tol = 1.5
        on_exterior = (
            kitchen.y <= tol
            or kitchen.y + kitchen.height >= layout.plot_length - tol
            or kitchen.x <= tol
            or kitchen.x + kitchen.width >= layout.plot_width - tol
        )
        if not on_exterior:
            suggestions.append(
                "Kitchen ventilation: Kitchen is fully enclosed by other rooms with no "
                "exterior wall access. Relocate it to a corner or boundary position to "
                "allow a window and rear service entrance — essential for cooking exhaust, "
                "cross-ventilation, and utility access."
            )

    def _check_bedroom_clustering(self, layout: Layout, suggestions: list[str]) -> None:
        """All bedrooms on the same half of the plot creates an unbalanced layout."""
        bedrooms = [r for r in layout.rooms if "bedroom" in r.room_type]
        if len(bedrooms) < 2:
            return
        pw = layout.plot_width
        pl = layout.plot_length
        cx = [r.x + r.width / 2 for r in bedrooms]
        cy = [r.y + r.height / 2 for r in bedrooms]
        all_north = all(y < pl * 0.5 for y in cy)
        all_south = all(y > pl * 0.5 for y in cy)
        all_west  = all(x < pw * 0.5 for x in cx)
        all_east  = all(x > pw * 0.5 for x in cx)
        if all_north or all_south or all_west or all_east:
            suggestions.append(
                "Bedroom distribution: All bedrooms are clustered on one side of the "
                "plot. Distribute secondary bedrooms across different zones (e.g., NW "
                "and SE) to balance the plan, give each bedroom individual exterior wall "
                "access, and prevent an overly dense private wing."
            )

    def _check_living_room_exterior(self, layout: Layout, suggestions: list[str]) -> None:
        """Living room should touch an exterior wall for natural light."""
        living = next((r for r in layout.rooms if r.room_type == "living_room"), None)
        if living is None:
            return
        tol = 1.5
        on_exterior = (
            living.y <= tol
            or living.y + living.height >= layout.plot_length - tol
            or living.x <= tol
            or living.x + living.width >= layout.plot_width - tol
        )
        if not on_exterior:
            suggestions.append(
                "Living room light: Living room has no exterior wall and will receive no "
                "natural light or ventilation. Reposition it to touch at least one plot "
                "boundary — the primary social space must have windows for comfort and "
                "building code compliance."
            )

    def _check_bedroom_dimensions(self, layout: Layout, suggestions: list[str]) -> None:
        """Flag bedrooms narrower than the minimum usable dimension."""
        bedrooms = [r for r in layout.rooms if "bedroom" in r.room_type]
        narrow = [
            r for r in bedrooms
            if min(r.width, r.height) < self._MIN_BEDROOM_DIM
        ]
        if narrow:
            names = ", ".join(
                r.room_type.replace("_", " ").title()
                + f" ({min(r.width, r.height):.0f}' min dim)"
                for r in narrow[:2]
            )
            suggestions.append(
                f"Room dimensions: {names} — minimum dimension is below the 10 ft "
                "threshold needed to fit a standard double bed with circulation clearance "
                "on both sides. Increase the narrower dimension or redistribute floor "
                "area from adjacent rooms."
            )

    def _check_acoustic_bedroom_gap(self, layout: Layout, suggestions: list[str]) -> None:
        """Bedrooms with almost no separation share sound paths."""
        bedrooms = [r for r in layout.rooms if "bedroom" in r.room_type]
        flagged: set[str] = set()
        for i, b1 in enumerate(bedrooms):
            for b2 in bedrooms[i + 1:]:
                if b1.id in flagged or b2.id in flagged:
                    continue
                # Gap in x direction
                gap_x = max(0.0, max(b1.x, b2.x) - min(b1.x + b1.width,  b2.x + b2.width))
                # Gap in y direction
                gap_y = max(0.0, max(b1.y, b2.y) - min(b1.y + b1.height, b2.y + b2.height))
                min_gap = min(gap_x, gap_y)
                if min_gap < self._MIN_ACOUSTIC_GAP:
                    n1 = b1.room_type.replace("_", " ").title()
                    n2 = b2.room_type.replace("_", " ").title()
                    suggestions.append(
                        f"Acoustic privacy: {n1} and {n2} are separated by less than "
                        f"{self._MIN_ACOUSTIC_GAP:.0f} ft with no intermediate room. "
                        "Place a bathroom, storage, or corridor between them to reduce "
                        "sound transmission between private sleeping spaces."
                    )
                    flagged.add(b1.id)
                    flagged.add(b2.id)
