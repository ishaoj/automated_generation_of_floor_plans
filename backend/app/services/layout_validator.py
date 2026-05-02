"""
Post-generation layout validator.
Checks all circulation and placement rules; returns violations list.
"""

from app.models.schemas import Layout, Room


class LayoutValidator:
    """
    Validates a generated Layout against hard architectural and Vastu rules.

    Rules checked:
     1. Entrance opens into Living Room or Parking only (never a bathroom)
     2. Parking (if present) is adjacent to the entrance wall
     3. Bathrooms do not share walls with each other
     4. Bathrooms are not directly accessible from the main entrance
     5. Bedrooms must not be directly adjacent to each other
     6. Bathroom must be adjacent to at least one bedroom (not floating near kitchen)
     7. Bathroom must not be adjacent to kitchen
     8. Pooja room must not be directly adjacent to parking
     9. Staircase (if present) must be adjacent to living room or parking only
    10. For row-house / corner plots, balconies must not be on closed side walls
    11. Puja room must not share a wall with any bathroom
    12. Living room and dining must be adjacent (share a wall)
    13. For 2-bed 2-bath layouts: each bathroom attached to exactly one bedroom (ensuite)
    14. Dining and kitchen must be adjacent (Entrance→Living→Dining→Kitchen flow)
    15. Spatial hierarchy: every bedroom must be larger than every bathroom
    16. Every bedroom must be adjacent to a circulation space (living/dining/corridor)
        so that no bedroom requires passing through a bathroom or another bedroom
    17. Kitchen must not be in the NE or SW zone (Vastu: fire must not pollute purity/earth)
    18. Master bedroom must not be in the NE or SE zone (disrupts sleep; fire zone)
    19. Balcony must be in the N, E, or NE zone (open/airy sides; avoid harsh S/W sun)
    20. Total number of openings (doors + windows) must be even and not a multiple of 10
    """

    ADJACENCY_TOLERANCE = 0.6   # feet — rooms considered touching if gap < this
    ENTRANCE_DEPTH      = 0.35  # parking must be within 35% of plot from entrance wall

    VALID_ENTRANCE_ROOMS = {"living_room", "parking", "entrance", "corridor"}

    def validate(self, layout: Layout) -> tuple[bool, list[str]]:
        """
        Returns (is_valid, violations).
        violations is an empty list when the layout is fully compliant.
        """
        violations: list[str] = []

        self._check_entrance_room(layout, violations)
        self._check_bathroom_not_at_entrance(layout, violations)
        self._check_bathroom_separation(layout, violations)
        self._check_parking_near_entrance(layout, violations)
        self._check_bedroom_separation(layout, violations)
        self._check_bathroom_near_bedroom(layout, violations)
        self._check_bathroom_not_near_kitchen(layout, violations)
        self._check_pooja_not_adjacent_to_parking(layout, violations)
        self._check_staircase_location(layout, violations)
        self._check_balcony_not_on_closed_wall(layout, violations)
        self._check_puja_not_adjacent_to_bathroom(layout, violations)
        self._check_living_dining_adjacent(layout, violations)
        self._check_dining_kitchen_adjacent(layout, violations)
        self._check_ensuite_pairing(layout, violations)
        self._check_bedroom_larger_than_bathroom(layout, violations)
        self._check_bedroom_circulation_access(layout, violations)
        self._check_kitchen_not_in_ne_sw(layout, violations)
        self._check_master_bedroom_zone(layout, violations)
        self._check_balcony_in_north_east_zone(layout, violations)
        self._check_even_openings(layout, violations)

        return len(violations) == 0, violations

    # ── Rule 1: entrance room ─────────────────────────────────────────────────

    def _check_entrance_room(self, layout: Layout, violations: list[str]) -> None:
        entry_room = self._get_entrance_connected_room(layout)
        if entry_room is None:
            return
        if entry_room.room_type not in self.VALID_ENTRANCE_ROOMS:
            violations.append(
                f"Main entrance opens into '{entry_room.room_type}' — "
                "must open into Living Room or Parking."
            )

    # ── Rule 4: bathroom at entrance ─────────────────────────────────────────

    def _check_bathroom_not_at_entrance(self, layout: Layout, violations: list[str]) -> None:
        entry_room = self._get_entrance_connected_room(layout)
        if entry_room and "bathroom" in entry_room.room_type:
            violations.append(
                "Main entrance leads directly into a bathroom — strictly prohibited."
            )

    # ── Rule 3: bathrooms separated from each other ───────────────────────────

    def _check_bathroom_separation(self, layout: Layout, violations: list[str]) -> None:
        baths = [r for r in layout.rooms if "bathroom" in r.room_type]
        for i, b1 in enumerate(baths):
            for b2 in baths[i + 1:]:
                if self._rooms_are_adjacent(b1, b2):
                    violations.append(
                        f"Bathrooms '{b1.id}' and '{b2.id}' share a wall — "
                        "bathrooms must not be adjacent to each other."
                    )

    # ── Rule 2: parking near entrance wall ────────────────────────────────────

    def _check_parking_near_entrance(self, layout: Layout, violations: list[str]) -> None:
        parking = next((r for r in layout.rooms if r.room_type == "parking"), None)
        if parking is None:
            return
        facing = layout.facing_direction
        pw, pl = layout.plot_width, layout.plot_length
        depth  = self.ENTRANCE_DEPTH
        ok = {
            "north": parking.y                   < pl * depth,
            "south": parking.y + parking.height  > pl * (1 - depth),
            "east":  parking.x + parking.width   > pw * (1 - depth),
            "west":  parking.x                   < pw * depth,
        }.get(facing, True)
        if not ok:
            violations.append(
                f"Parking is too far from the {facing} entrance — "
                "must be within 35% of the plot from the entrance wall."
            )

    # ── Rule 5: bedrooms must not be adjacent to each other ───────────────────

    def _check_bedroom_separation(self, layout: Layout, violations: list[str]) -> None:
        bedrooms = [r for r in layout.rooms if "bedroom" in r.room_type]
        for i, b1 in enumerate(bedrooms):
            for b2 in bedrooms[i + 1:]:
                if self._rooms_are_adjacent(b1, b2):
                    violations.append(
                        f"Bedroom '{b1.id}' and bedroom '{b2.id}' share a wall — "
                        "bedrooms must be separated by a corridor or other room."
                    )

    # ── Rule 6: bathroom must be near a bedroom ───────────────────────────────

    def _check_bathroom_near_bedroom(self, layout: Layout, violations: list[str]) -> None:
        baths    = [r for r in layout.rooms if "bathroom" in r.room_type]
        bedrooms = [r for r in layout.rooms if "bedroom"  in r.room_type]
        if not baths or not bedrooms:
            return
        for bath in baths:
            near_bedroom = any(self._rooms_are_adjacent(bath, br) for br in bedrooms)
            if not near_bedroom:
                violations.append(
                    f"Bathroom '{bath.id}' is not adjacent to any bedroom — "
                    "bathrooms should be attached to or near bedrooms."
                )

    # ── Rule 7: bathroom must not be adjacent to kitchen ─────────────────────

    def _check_bathroom_not_near_kitchen(self, layout: Layout, violations: list[str]) -> None:
        baths    = [r for r in layout.rooms if "bathroom" in r.room_type]
        kitchens = [r for r in layout.rooms if "kitchen"  in r.room_type]
        for bath in baths:
            for kitchen in kitchens:
                if self._rooms_are_adjacent(bath, kitchen):
                    violations.append(
                        f"Bathroom '{bath.id}' is adjacent to kitchen '{kitchen.id}' — "
                        "bathrooms must not be placed next to the kitchen."
                    )

    # ── Rule 8: pooja room must not be adjacent to parking ────────────────────

    def _check_pooja_not_adjacent_to_parking(self, layout: Layout, violations: list[str]) -> None:
        pooja_rooms = [r for r in layout.rooms if "puja" in r.room_type or "pooja" in r.room_type]
        parking     = [r for r in layout.rooms if "parking" in r.room_type]
        for pooja in pooja_rooms:
            for park in parking:
                if self._rooms_are_adjacent(pooja, park):
                    violations.append(
                        f"Pooja room '{pooja.id}' is directly adjacent to parking — "
                        "pooja room should be accessible from the living area, not parking."
                    )

    # ── Rule 9: staircase must be in living room or parking area ──────────────

    def _check_staircase_location(self, layout: Layout, violations: list[str]) -> None:
        staircases    = [r for r in layout.rooms if "staircase" in r.room_type]
        allowed_rooms = [r for r in layout.rooms
                         if r.room_type in {"living_room", "parking", "entrance", "corridor"}]
        bedrooms      = [r for r in layout.rooms if "bedroom" in r.room_type]

        for stair in staircases:
            # Must be adjacent to at least one allowed room
            near_allowed = any(self._rooms_are_adjacent(stair, r) for r in allowed_rooms)
            if not near_allowed:
                violations.append(
                    f"Staircase '{stair.id}' is not adjacent to any living/parking area — "
                    "staircases must be accessible from the living room or parking."
                )
            # Must NOT be inside / adjacent to a bedroom
            near_bedroom = any(self._rooms_are_adjacent(stair, br) for br in bedrooms)
            if near_bedroom:
                violations.append(
                    f"Staircase '{stair.id}' is adjacent to a bedroom — "
                    "staircases must not be inside or directly accessible from bedrooms."
                )

    # ── Rule 10: balcony not on closed side walls (row house / corner plot) ───

    def _check_balcony_not_on_closed_wall(self, layout: Layout, violations: list[str]) -> None:
        balconies  = [r for r in layout.rooms if "balcony" in r.room_type]
        open_walls = getattr(layout, "open_walls", ["north", "south", "east", "west"])

        # Only applies when some walls are closed (not a standalone plot)
        if set(open_walls) == {"north", "south", "east", "west"}:
            return

        pw = layout.plot_width
        pl = layout.plot_length
        tol = 1.0

        for balcony in balconies:
            # Determine which boundary wall the balcony touches
            touches: list[str] = []
            if balcony.y <= tol:
                touches.append("north")
            if balcony.y + balcony.height >= pl - tol:
                touches.append("south")
            if balcony.x <= tol:
                touches.append("west")
            if balcony.x + balcony.width >= pw - tol:
                touches.append("east")

            for wall in touches:
                if wall not in open_walls:
                    violations.append(
                        f"Balcony '{balcony.id}' is on the '{wall}' wall which is a "
                        "closed/shared wall — balconies must only face open (road-facing) walls."
                    )

    # ── Rule 11: puja room not adjacent to bathroom ───────────────────────────

    def _check_puja_not_adjacent_to_bathroom(self, layout: Layout, violations: list[str]) -> None:
        pooja_rooms = [r for r in layout.rooms if "puja" in r.room_type or "pooja" in r.room_type]
        bathrooms   = [r for r in layout.rooms if "bathroom" in r.room_type]
        for pooja in pooja_rooms:
            for bath in bathrooms:
                if self._rooms_are_adjacent(pooja, bath):
                    violations.append(
                        f"Puja room '{pooja.id}' shares a wall with bathroom '{bath.id}' — "
                        "puja room must not be adjacent to any bathroom."
                    )

    # ── Rule 12: living room and dining must be adjacent ─────────────────────

    def _check_living_dining_adjacent(self, layout: Layout, violations: list[str]) -> None:
        living = [r for r in layout.rooms if r.room_type == "living_room"]
        dining = [r for r in layout.rooms if r.room_type == "dining"]
        if not living or not dining:
            return
        if not self._rooms_are_adjacent(living[0], dining[0]):
            violations.append(
                "Living room and dining are not adjacent — "
                "they should share a wall for practical functional flow."
            )

    # ── Rule 14: dining and kitchen must be adjacent ──────────────────────────

    def _check_dining_kitchen_adjacent(self, layout: Layout, violations: list[str]) -> None:
        dining  = [r for r in layout.rooms if r.room_type == "dining"]
        kitchen = [r for r in layout.rooms if r.room_type == "kitchen"]
        if not dining or not kitchen:
            return
        if not self._rooms_are_adjacent(dining[0], kitchen[0]):
            violations.append(
                "Dining and kitchen are not adjacent — "
                "they must share a wall for the Entrance→Living→Dining→Kitchen flow."
            )

    # ── Rule 13: 2-bed 2-bath ensuite pairing ────────────────────────────────

    def _check_ensuite_pairing(self, layout: Layout, violations: list[str]) -> None:
        """
        For exactly 2-bedroom + 2-bathroom layouts each bathroom must be
        an ensuite: adjacent to exactly one bedroom, and each bedroom must
        have exactly one bathroom attached to it.
        """
        bedrooms  = [r for r in layout.rooms if "bedroom" in r.room_type]
        bathrooms = [r for r in layout.rooms if "bathroom" in r.room_type]
        if len(bedrooms) != 2 or len(bathrooms) != 2:
            return

        for bath in bathrooms:
            adjacent_beds = [br for br in bedrooms if self._rooms_are_adjacent(bath, br)]
            if len(adjacent_beds) == 0:
                violations.append(
                    f"Bathroom '{bath.id}' has no adjacent bedroom — "
                    "in a 2-bed 2-bath layout every bathroom must be an ensuite."
                )
            elif len(adjacent_beds) > 1:
                violations.append(
                    f"Bathroom '{bath.id}' is adjacent to {len(adjacent_beds)} bedrooms — "
                    "each ensuite bathroom must attach to exactly one bedroom."
                )

        for bed in bedrooms:
            adjacent_baths = [ba for ba in bathrooms if self._rooms_are_adjacent(ba, bed)]
            if len(adjacent_baths) == 0:
                violations.append(
                    f"Bedroom '{bed.id}' has no attached bathroom — "
                    "in a 2-bed 2-bath layout each bedroom must have an ensuite."
                )
            elif len(adjacent_baths) > 1:
                violations.append(
                    f"Bedroom '{bed.id}' is adjacent to {len(adjacent_baths)} bathrooms — "
                    "each bedroom must have exactly one ensuite bathroom."
                )

    # ── Rule 15: spatial hierarchy — bedrooms larger than bathrooms ──────────

    def _check_bedroom_larger_than_bathroom(self, layout: Layout, violations: list[str]) -> None:
        bedrooms  = [r for r in layout.rooms if "bedroom"  in r.room_type]
        bathrooms = [r for r in layout.rooms if "bathroom" in r.room_type]
        if not bedrooms or not bathrooms:
            return
        for bed in bedrooms:
            for bath in bathrooms:
                if bath.area >= bed.area:
                    violations.append(
                        f"Spatial hierarchy violated: bathroom '{bath.id}' "
                        f"({bath.area:.0f} sq ft) is not smaller than bedroom "
                        f"'{bed.id}' ({bed.area:.0f} sq ft)."
                    )

    # ── Rule 16: every bedroom has independent access to a circulation space ──

    _CIRCULATION_TYPES = {"living_room", "dining", "corridor", "entrance", "lobby"}

    def _check_bedroom_circulation_access(self, layout: Layout, violations: list[str]) -> None:
        """
        Each bedroom must share a wall with at least one circulation space
        (living room, dining, corridor, entrance, or lobby).

        If a bedroom's only neighbours are bathrooms and/or other bedrooms,
        occupants would be forced to pass through a bathroom or another
        bedroom to reach the rest of the house — both are prohibited.
        """
        bedrooms     = [r for r in layout.rooms if "bedroom" in r.room_type]
        circulation  = [r for r in layout.rooms
                        if any(t in r.room_type for t in self._CIRCULATION_TYPES)]

        if not bedrooms or not circulation:
            return

        for bed in bedrooms:
            has_access = any(self._rooms_are_adjacent(bed, c) for c in circulation)
            if not has_access:
                adj_types = [
                    r.room_type for r in layout.rooms
                    if r.id != bed.id and self._rooms_are_adjacent(bed, r)
                ]
                violations.append(
                    f"Bedroom '{bed.id}' has no direct access to a circulation space "
                    f"(living/dining/corridor) — only adjacent to: "
                    f"{adj_types or ['nothing']}. "
                    "This forces passage through a bathroom or another bedroom."
                )

    # ── Rule 17: kitchen not in NE or SW zone ────────────────────────────────

    def _check_kitchen_not_in_ne_sw(self, layout: Layout, violations: list[str]) -> None:
        """
        Kitchen in the NE (Water/Purity) zone is the most serious Vastu violation —
        fire element in the water zone.  SW (Earth) is also forbidden: it is the
        stability zone reserved for the heaviest, most grounding room (master bedroom).
        """
        forbidden = {"northeast", "southwest"}
        for r in layout.rooms:
            if "kitchen" in r.room_type and r.zone in forbidden:
                violations.append(
                    f"Kitchen '{r.id}' is in the {r.zone.upper()} zone — "
                    f"strictly forbidden by Vastu (NE = Water/Purity, SW = Earth/Stability). "
                    "Kitchen must be in SE (primary) or NW (secondary)."
                )

    # ── Rule 18: master bedroom not in NE or SE zone ─────────────────────────

    def _check_master_bedroom_zone(self, layout: Layout, violations: list[str]) -> None:
        """
        NE disrupts sleep (water element + magnetic field alignment).
        SE is the fire zone — heat and energy are incompatible with rest.
        Master bedroom belongs in SW (earth/stability).
        """
        forbidden = {"northeast", "southeast"}
        for r in layout.rooms:
            if r.room_type == "master_bedroom" and r.zone in forbidden:
                violations.append(
                    f"Master bedroom '{r.id}' is in the {r.zone.upper()} zone — "
                    "forbidden by Vastu. NE disrupts sleep via magnetic fields; "
                    "SE (Fire zone) creates restlessness. Must be in SW."
                )

    # ── Rule 19: balcony in N, E, or NE zone only ────────────────────────────

    def _check_balcony_in_north_east_zone(self, layout: Layout, violations: list[str]) -> None:
        """
        Balconies on S/W walls expose occupants to harsh afternoon sun and heat.
        N/E/NE sides receive morning light, diffused illumination, and cool breezes —
        the correct orientation for an open, airy transitional space.
        """
        allowed = {"north", "northeast", "east"}
        for r in layout.rooms:
            if "balcony" in r.room_type and r.zone not in allowed:
                violations.append(
                    f"Balcony '{r.id}' is in the {r.zone.upper()} zone — "
                    "balconies must face North, East, or North-East for morning light "
                    "and protection from harsh S/W afternoon sun."
                )

    # ── Rule 20: even number of total openings ────────────────────────────────

    def _check_even_openings(self, layout: Layout, violations: list[str]) -> None:
        """
        Vastu prescribes that the total count of doors + windows must be even
        (2, 4, 6, 8, 12, …) and must not be a multiple of 10 (10, 20, 30, …).
        """
        total = len(layout.doors) + len(layout.windows)
        if total == 0:
            return
        if total % 2 != 0:
            violations.append(
                f"Total openings (doors + windows) is {total} — an odd number. "
                "Vastu requires an even total count (2, 4, 6, 8, 12, …)."
            )
        elif total % 10 == 0:
            violations.append(
                f"Total openings (doors + windows) is {total} — a multiple of 10. "
                "Vastu forbids multiples of 10; adjust by adding or removing one opening."
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_entrance_connected_room(self, layout: Layout):
        for door in layout.doors:
            if door.is_main_entrance and door.connects:
                return self._room_by_id(layout, door.connects[0])
        return None

    def _room_by_id(self, layout: Layout, room_id: str):
        for r in layout.rooms:
            if r.id == room_id:
                return r
        return None

    def _rooms_are_adjacent(self, r1: Room, r2: Room) -> bool:
        tol = self.ADJACENCY_TOLERANCE
        share_v = (
            abs(r1.x + r1.width  - r2.x) < tol or
            abs(r2.x + r2.width  - r1.x) < tol
        )
        h_overlap = (r1.y < r2.y + r2.height) and (r2.y < r1.y + r1.height)

        share_h = (
            abs(r1.y + r1.height - r2.y) < tol or
            abs(r2.y + r2.height - r1.y) < tol
        )
        v_overlap = (r1.x < r2.x + r2.width) and (r2.x < r1.x + r1.width)

        return (share_v and h_overlap) or (share_h and v_overlap)
