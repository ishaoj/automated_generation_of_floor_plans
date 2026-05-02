"""
Door and window placement logic for floor plans
"""

from typing import Optional
from app.models.schemas import Layout, Door, Window, Room


class OpeningsPlacer:
    """
    Places doors and windows in a floor plan layout.

    External door rules:
    - Main entrance on the facing boundary wall (living room > parking > other).
    - Parking gets its own external vehicle entrance if it touches any open wall
      and its main entrance has not already been served by the main entrance door.
    - No room may receive more than MAX_DOORS_PER_ROOM external doors.
    - Doors are only placed where functionally necessary; redundant or duplicate
      openings are skipped.

    Window placement rules (Vastu):
    - Prefer windows on North and East walls (morning sunlight).
    - Avoid windows on South and West (afternoon heat).
    - Each major room should have at least one window if on an open exterior wall.
    - Window width: 4 feet standard.
    """

    # Maximum number of external doors any single room may receive.
    MAX_DOORS_PER_ROOM  = 2

    # Standard door widths (feet)
    door_width           = 3.0    # pedestrian entrance
    _PARKING_DOOR_WIDTH  = 9.0    # vehicle entrance (single-car bay)
    window_width         = 4.0
    min_wall_clearance   = 1.0    # minimum clearance from wall corners

    def __init__(self):
        pass  # all widths are class-level constants

    def place_openings(self, layout: Layout) -> Layout:
        """
        Add external entrance openings and windows to the layout.

        Door placement order:
          1. Main pedestrian entrance (living room / entrance zone).
          2. Parking vehicle entrance (independent external opening, if applicable).
        Interior doors are intentionally omitted — only boundary openings are shown.
        """
        doors: list[Door] = []
        # Tracks how many external doors have been assigned to each room (by id).
        door_count_per_room: dict[str, int] = {}

        main_door = self._place_main_entrance(layout)
        if main_door:
            doors.append(main_door)
            for rid in main_door.connects:
                door_count_per_room[rid] = door_count_per_room.get(rid, 0) + 1

        parking_door = self._place_parking_entrance(layout, doors, door_count_per_room)
        if parking_door:
            doors.append(parking_door)
            for rid in parking_door.connects:
                door_count_per_room[rid] = door_count_per_room.get(rid, 0) + 1

        layout.doors   = doors
        layout.windows = self._place_windows(layout)
        return layout

    def _place_main_entrance(self, layout: Layout) -> Optional[Door]:
        """
        Place the main entrance door on the facing-direction wall of the
        preferred entrance room (living room > parking > other allowed rooms).

        The door is placed on the plot boundary wall that corresponds to
        the facing direction, at the x/y of the preferred room, so the
        entrance always opens into an appropriate space.
        """
        facing = layout.facing_direction
        pw     = layout.plot_width
        pl     = layout.plot_length

        entrance_room = self._find_entrance_room(layout)
        if entrance_room is None:
            return None

        # Door position: on the plot boundary in the facing direction,
        # horizontally/vertically centred on the entrance room.
        if facing == "north":
            x           = entrance_room.x + entrance_room.width / 2
            y           = 0.0
            orientation = "horizontal"
        elif facing == "south":
            x           = entrance_room.x + entrance_room.width / 2
            y           = pl
            orientation = "horizontal"
        elif facing == "east":
            x           = pw
            y           = entrance_room.y + entrance_room.height / 2
            orientation = "vertical"
        else:  # west
            x           = 0.0
            y           = entrance_room.y + entrance_room.height / 2
            orientation = "vertical"

        return Door(
            id="main_entrance",
            x=x,
            y=y,
            width=self.door_width,
            orientation=orientation,
            connects=[entrance_room.id],
            is_main_entrance=True,
        )

    # Room types that may receive the main entrance door (priority order)
    _ENTRANCE_PRIORITY = ["living_room", "parking", "entrance", "corridor", "dining"]
    # Room types that must never receive the main entrance door
    _ENTRANCE_FORBIDDEN = {"bathroom", "toilet", "master_bedroom", "bedroom", "puja_room"}

    def _find_entrance_room(self, layout: Layout) -> Optional[Room]:
        """
        Find the best room to receive the main entrance door.

        Priority:
          1. Living Room or Parking on the entrance side
          2. Any allowed room on the entrance side (closest to entrance_position)
          3. Living Room or Parking anywhere in the layout
          4. Any non-forbidden room (closest to entrance_position)
        """
        facing       = layout.facing_direction
        entrance_pos = layout.entrance_position

        # Depth limit = 30 % of the relevant plot dimension (min 8 ft)
        if facing in ("north", "south"):
            depth_limit = max(8.0, layout.plot_length * 0.30)
        else:
            depth_limit = max(8.0, layout.plot_width  * 0.30)

        def on_entrance_side(room: Room) -> bool:
            if facing == "north":
                return room.y <= depth_limit
            elif facing == "south":
                return room.y + room.height >= layout.plot_length - depth_limit
            elif facing == "east":
                return room.x + room.width >= layout.plot_width - depth_limit
            else:  # west
                return room.x <= depth_limit

        def dist_to_entrance(room: Room) -> float:
            cx = room.x + room.width  / 2
            cy = room.y + room.height / 2
            return ((cx - entrance_pos[0]) ** 2 + (cy - entrance_pos[1]) ** 2) ** 0.5

        def is_forbidden(room: Room) -> bool:
            return any(f in room.room_type for f in self._ENTRANCE_FORBIDDEN)

        # Pass 1: preferred type touching the entrance boundary
        preferred_on_side = [
            r for r in layout.rooms
            if on_entrance_side(r)
            and any(r.room_type == p for p in self._ENTRANCE_PRIORITY)
        ]
        if preferred_on_side:
            return min(preferred_on_side, key=dist_to_entrance)

        # Pass 2: any allowed room touching the entrance boundary
        # (always choose a room that physically touches the facing wall)
        allowed_on_side = [
            r for r in layout.rooms
            if on_entrance_side(r) and not is_forbidden(r)
        ]
        if allowed_on_side:
            return min(allowed_on_side, key=dist_to_entrance)

        # Pass 3: widen depth to 50% and retry preferred types
        depth_limit_wide = max(depth_limit * 1.5, layout.plot_length * 0.50
                               if facing in ("north", "south")
                               else layout.plot_width * 0.50)

        def on_wide_side(room: Room) -> bool:
            if facing == "north":
                return room.y <= depth_limit_wide
            elif facing == "south":
                return room.y + room.height >= layout.plot_length - depth_limit_wide
            elif facing == "east":
                return room.x + room.width >= layout.plot_width - depth_limit_wide
            else:
                return room.x <= depth_limit_wide

        preferred_wide = [
            r for r in layout.rooms
            if on_wide_side(r)
            and any(r.room_type == p for p in self._ENTRANCE_PRIORITY)
        ]
        if preferred_wide:
            return min(preferred_wide, key=dist_to_entrance)

        # Pass 4: any non-forbidden room (closest to entrance_position as fallback)
        allowed = [r for r in layout.rooms if not is_forbidden(r)]
        if allowed:
            return min(allowed, key=dist_to_entrance)

        return layout.rooms[0] if layout.rooms else None

    def _place_parking_entrance(
        self,
        layout: Layout,
        existing_doors: list[Door],
        door_count_per_room: dict[str, int],
    ) -> Optional[Door]:
        """
        Place an independent external vehicle entrance on the plot boundary
        for the parking area.

        Rules enforced:
          - Only if a parking room exists in the layout.
          - Skipped when the main entrance already opens into parking
            (no redundant duplicate opening).
          - Parking must physically touch at least one open (exterior) wall
            within _BOUNDARY_TOL feet; facing wall is tried first.
          - Respects MAX_DOORS_PER_ROOM: skipped if parking already has 2 doors.
          - Width: _PARKING_DOOR_WIDTH (9 ft, vehicle-sized).
        """
        _BOUNDARY_TOL = 3.0   # feet — how close to boundary = "touches it"

        parking = next(
            (r for r in layout.rooms if "parking" in r.room_type.lower()), None
        )
        if parking is None:
            return None

        # If the main entrance already connects to parking, don't duplicate.
        for door in existing_doors:
            connected_types = {
                r.room_type for r in layout.rooms if r.id in door.connects
            }
            if any("parking" in rt for rt in connected_types):
                return None

        # Respect per-room door cap.
        if door_count_per_room.get(parking.id, 0) >= self.MAX_DOORS_PER_ROOM:
            return None

        facing     = layout.facing_direction
        pw         = layout.plot_width
        pl         = layout.plot_length
        open_walls = getattr(layout, "open_walls", ["north", "south", "east", "west"])

        # Try facing wall first, then remaining open walls.
        wall_order = [facing] + [w for w in open_walls if w != facing]

        for wall in wall_order:
            if wall not in open_walls:
                continue
            if wall == "north" and parking.y <= _BOUNDARY_TOL:
                return Door(
                    id="parking_entrance",
                    x=parking.x + parking.width / 2,
                    y=0.0,
                    width=self._PARKING_DOOR_WIDTH,
                    orientation="horizontal",
                    connects=[parking.id],
                    is_main_entrance=False,
                )
            if wall == "south" and parking.y + parking.height >= pl - _BOUNDARY_TOL:
                return Door(
                    id="parking_entrance",
                    x=parking.x + parking.width / 2,
                    y=pl,
                    width=self._PARKING_DOOR_WIDTH,
                    orientation="horizontal",
                    connects=[parking.id],
                    is_main_entrance=False,
                )
            if wall == "east" and parking.x + parking.width >= pw - _BOUNDARY_TOL:
                return Door(
                    id="parking_entrance",
                    x=pw,
                    y=parking.y + parking.height / 2,
                    width=self._PARKING_DOOR_WIDTH,
                    orientation="vertical",
                    connects=[parking.id],
                    is_main_entrance=False,
                )
            if wall == "west" and parking.x <= _BOUNDARY_TOL:
                return Door(
                    id="parking_entrance",
                    x=0.0,
                    y=parking.y + parking.height / 2,
                    width=self._PARKING_DOOR_WIDTH,
                    orientation="vertical",
                    connects=[parking.id],
                    is_main_entrance=False,
                )

        # Parking does not touch any open wall — no external entrance possible.
        return None

    # Rooms that may serve as a staircase's only entry point
    _STAIR_ALLOWED_NEIGHBOURS = {"living_room", "parking", "entrance", "corridor"}
    # Rooms that must not share a door
    _FORBIDDEN_DOOR_PAIRS: set[frozenset] = {
        frozenset({"puja_room",  "parking"}),
        frozenset({"puja_room",  "bathroom"}),
        frozenset({"kitchen",    "bathroom"}),
    }

    def _place_interior_doors(self, layout: Layout) -> list[Door]:
        """
        Place doors between adjacent rooms with architectural rules:

        1. Bedrooms get at most one interior door (private rooms).
        2. Bedroom-to-bedroom doors are forbidden — bedrooms must not
           connect directly to each other.
        3. Staircase doors are only allowed from living room, parking,
           entrance hall, or corridor.
        4. Puja room must not connect directly to parking or bathroom.
        5. Kitchen must not connect directly to bathroom.
        """
        doors = []
        door_count = 0
        processed_pairs: set = set()
        bedroom_door_count: dict[str, int] = {}  # room_id → door count

        for i, room1 in enumerate(layout.rooms):
            for j, room2 in enumerate(layout.rooms):
                if i >= j:
                    continue

                pair_key = tuple(sorted([room1.id, room2.id]))
                if pair_key in processed_pairs:
                    continue

                t1 = room1.room_type
                t2 = room2.room_type

                # ── Rule 1: bedroom-to-bedroom forbidden ──────────────────────
                if "bedroom" in t1 and "bedroom" in t2:
                    processed_pairs.add(pair_key)
                    continue

                # ── Rule 2: staircase only from allowed neighbours ─────────────
                if "staircase" in t1 and t2 not in self._STAIR_ALLOWED_NEIGHBOURS:
                    processed_pairs.add(pair_key)
                    continue
                if "staircase" in t2 and t1 not in self._STAIR_ALLOWED_NEIGHBOURS:
                    processed_pairs.add(pair_key)
                    continue

                # ── Rule 3: forbidden room-type pairs ─────────────────────────
                if frozenset({t1, t2}) in self._FORBIDDEN_DOOR_PAIRS:
                    processed_pairs.add(pair_key)
                    continue

                # ── Rule 4: each bedroom may have at most one interior door ────
                if "bedroom" in t1 and bedroom_door_count.get(room1.id, 0) >= 1:
                    continue
                if "bedroom" in t2 and bedroom_door_count.get(room2.id, 0) >= 1:
                    continue

                door = self._create_door_between_rooms(room1, room2, door_count)
                if door:
                    doors.append(door)
                    processed_pairs.add(pair_key)
                    door_count += 1
                    if "bedroom" in t1:
                        bedroom_door_count[room1.id] = bedroom_door_count.get(room1.id, 0) + 1
                    if "bedroom" in t2:
                        bedroom_door_count[room2.id] = bedroom_door_count.get(room2.id, 0) + 1

        return doors

    def _create_door_between_rooms(self, room1: Room, room2: Room,
                                    door_index: int) -> Optional[Door]:
        """Create a door between two adjacent rooms if they share a wall"""
        tolerance = 0.5

        # Check for horizontal adjacency (room1 left of room2 or vice versa)
        # Room1 right edge touches Room2 left edge
        if abs(room1.x + room1.width - room2.x) < tolerance:
            overlap_start = max(room1.y, room2.y)
            overlap_end = min(room1.y + room1.height, room2.y + room2.height)
            if overlap_end - overlap_start >= self.door_width + 2 * self.min_wall_clearance:
                door_y = (overlap_start + overlap_end) / 2
                return Door(
                    id=f"door_{door_index}",
                    x=room2.x,
                    y=door_y,
                    width=self.door_width,
                    orientation="vertical",
                    connects=[room1.id, room2.id],
                    is_main_entrance=False
                )

        # Room2 right edge touches Room1 left edge
        if abs(room2.x + room2.width - room1.x) < tolerance:
            overlap_start = max(room1.y, room2.y)
            overlap_end = min(room1.y + room1.height, room2.y + room2.height)
            if overlap_end - overlap_start >= self.door_width + 2 * self.min_wall_clearance:
                door_y = (overlap_start + overlap_end) / 2
                return Door(
                    id=f"door_{door_index}",
                    x=room1.x,
                    y=door_y,
                    width=self.door_width,
                    orientation="vertical",
                    connects=[room1.id, room2.id],
                    is_main_entrance=False
                )

        # Check for vertical adjacency (room1 above room2 or vice versa)
        # Room1 bottom edge touches Room2 top edge
        if abs(room1.y + room1.height - room2.y) < tolerance:
            overlap_start = max(room1.x, room2.x)
            overlap_end = min(room1.x + room1.width, room2.x + room2.width)
            if overlap_end - overlap_start >= self.door_width + 2 * self.min_wall_clearance:
                door_x = (overlap_start + overlap_end) / 2
                return Door(
                    id=f"door_{door_index}",
                    x=door_x,
                    y=room2.y,
                    width=self.door_width,
                    orientation="horizontal",
                    connects=[room1.id, room2.id],
                    is_main_entrance=False
                )

        # Room2 bottom edge touches Room1 top edge
        if abs(room2.y + room2.height - room1.y) < tolerance:
            overlap_start = max(room1.x, room2.x)
            overlap_end = min(room1.x + room1.width, room2.x + room2.width)
            if overlap_end - overlap_start >= self.door_width + 2 * self.min_wall_clearance:
                door_x = (overlap_start + overlap_end) / 2
                return Door(
                    id=f"door_{door_index}",
                    x=door_x,
                    y=room1.y,
                    width=self.door_width,
                    orientation="horizontal",
                    connects=[room1.id, room2.id],
                    is_main_entrance=False
                )

        return None

    def _place_windows(self, layout: Layout) -> list[Window]:
        """
        Place windows on exterior walls of each room

        LEARNING NOTE (IMPORTANT!):
        Windows can ONLY be placed on OPEN walls (exterior walls with road access).
        - In a row house (open_one), only the facing wall can have windows
        - In a corner plot (open_two_corner), facing + one adjacent wall can have windows
        - Closed walls are shared with neighbors - no windows there!

        This significantly affects natural light and ventilation in the design.
        """
        windows = []
        window_count = 0

        pw = layout.plot_width
        pl = layout.plot_length

        # Get open walls from layout (walls where windows CAN be placed)
        open_walls = getattr(layout, 'open_walls', ["north", "south", "east", "west"])

        for room in layout.rooms:
            # Skip bathrooms (small or no windows)
            if "bathroom" in room.room_type.lower() or "toilet" in room.room_type.lower():
                continue

            room_windows = []

            # Check each wall for exterior exposure AND if it's an open wall
            # North wall (y = 0) - only if north is open
            if room.y <= 1 and "north" in open_walls:
                room_windows.append(self._create_window(
                    room, "north", window_count, pw, pl
                ))
                window_count += 1

            # East wall (x + width = plot_width) - only if east is open
            if room.x + room.width >= pw - 1 and "east" in open_walls:
                room_windows.append(self._create_window(
                    room, "east", window_count, pw, pl
                ))
                window_count += 1

            # South wall (y + height = plot_length) - only if south is open
            if room.y + room.height >= pl - 1 and "south" in open_walls:
                # Only add if no north/east windows (Vastu: prefer N/E)
                if not any(w.wall in ["north", "east"] for w in room_windows):
                    room_windows.append(self._create_window(
                        room, "south", window_count, pw, pl
                    ))
                    window_count += 1

            # West wall (x = 0) - only if west is open
            if room.x <= 1 and "west" in open_walls:
                # Only add if no north/east windows (Vastu: prefer N/E)
                if not any(w.wall in ["north", "east"] for w in room_windows):
                    room_windows.append(self._create_window(
                        room, "west", window_count, pw, pl
                    ))
                    window_count += 1

            # Ensure each major room has at least one window (if possible)
            if not room_windows and room.area >= 80:
                # Add window on the best available OPEN wall
                best_wall = self._find_best_window_wall(room, pw, pl, open_walls)
                if best_wall:
                    room_windows.append(self._create_window(
                        room, best_wall, window_count, pw, pl
                    ))
                    window_count += 1

            windows.extend(room_windows)

        return windows

    def _create_window(self, room: Room, wall: str, index: int,
                       plot_width: float, plot_length: float) -> Window:
        """Create a window on specified wall of room"""
        if wall == "north":
            x = room.x + room.width / 2
            y = room.y
            orientation = "horizontal"
        elif wall == "south":
            x = room.x + room.width / 2
            y = room.y + room.height
            orientation = "horizontal"
        elif wall == "east":
            x = room.x + room.width
            y = room.y + room.height / 2
            orientation = "vertical"
        else:  # west
            x = room.x
            y = room.y + room.height / 2
            orientation = "vertical"

        return Window(
            id=f"window_{index}",
            x=x,
            y=y,
            width=self.window_width,
            orientation=orientation,
            room_id=room.id,
            wall=wall
        )

    def _find_best_window_wall(self, room: Room, plot_width: float, plot_length: float,
                                open_walls: list[str] = None) -> Optional[str]:
        """
        Find the best wall for a window based on Vastu (prefer N/E)

        LEARNING NOTE:
        This function respects which walls are OPEN (exterior).
        A room in the corner of a row house might only have ONE wall
        where a window is possible - the wall facing the road.
        """
        if open_walls is None:
            open_walls = ["north", "south", "east", "west"]

        # Check proximity to each plot boundary
        walls = []

        # Distance to north boundary - only if north is open
        if room.y < plot_length / 3 and "north" in open_walls:
            walls.append(("north", room.y))

        # Distance to east boundary - only if east is open
        if room.x + room.width > plot_width * 2/3 and "east" in open_walls:
            walls.append(("east", plot_width - (room.x + room.width)))

        # Distance to south boundary - only if south is open
        if room.y + room.height > plot_length * 2/3 and "south" in open_walls:
            walls.append(("south", plot_length - (room.y + room.height)))

        # Distance to west boundary - only if west is open
        if room.x < plot_width / 3 and "west" in open_walls:
            walls.append(("west", room.x))

        if not walls:
            return None

        # Prefer north/east (Vastu), then south, then west
        priority = {"north": 0, "east": 1, "south": 2, "west": 3}
        walls.sort(key=lambda w: (priority.get(w[0], 4), w[1]))

        return walls[0][0]
