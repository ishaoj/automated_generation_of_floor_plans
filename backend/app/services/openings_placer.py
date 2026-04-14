"""
Door and window placement logic for floor plans
"""

from typing import Optional
from app.models.schemas import Layout, Door, Window, Room


class OpeningsPlacer:
    """
    Places doors and windows in a floor plan layout

    Door placement rules:
    - Main entrance on the facing wall
    - Interior doors between adjacent rooms
    - Door width: 3 feet standard

    Window placement rules (Vastu):
    - Prefer windows on North and East walls (morning sunlight)
    - Avoid windows on South and West (afternoon heat)
    - Each room should have at least one window if on exterior
    - Window width: 4 feet standard
    """

    def __init__(self):
        self.door_width = 3.0
        self.window_width = 4.0
        self.min_wall_clearance = 1.0  # Minimum distance from corners

    def place_openings(self, layout: Layout) -> Layout:
        """
        Add doors and windows to the layout

        Args:
            layout: Floor plan layout with rooms

        Returns:
            Updated layout with doors and windows
        """
        doors = []
        windows = []

        # Place main entrance
        main_door = self._place_main_entrance(layout)
        if main_door:
            doors.append(main_door)

        # Place interior doors between adjacent rooms
        interior_doors = self._place_interior_doors(layout)
        doors.extend(interior_doors)

        # Place windows on exterior walls
        windows = self._place_windows(layout)

        # Update layout with openings
        layout.doors = doors
        layout.windows = windows

        return layout

    def _place_main_entrance(self, layout: Layout) -> Optional[Door]:
        """Place main entrance door based on facing direction"""
        facing = layout.facing_direction
        pw = layout.plot_width
        pl = layout.plot_length

        # Find the room closest to the entrance side
        entrance_room = self._find_entrance_room(layout)

        if facing == "north":
            x = layout.entrance_position[0]
            y = 0
            orientation = "horizontal"
        elif facing == "south":
            x = layout.entrance_position[0]
            y = pl
            orientation = "horizontal"
        elif facing == "east":
            x = pw
            y = layout.entrance_position[1]
            orientation = "vertical"
        else:  # west
            x = 0
            y = layout.entrance_position[1]
            orientation = "vertical"

        connects = [entrance_room.id] if entrance_room else []

        return Door(
            id="main_entrance",
            x=x,
            y=y,
            width=self.door_width,
            orientation=orientation,
            connects=connects,
            is_main_entrance=True
        )

    def _find_entrance_room(self, layout: Layout) -> Optional[Room]:
        """Find the room closest to the entrance"""
        facing = layout.facing_direction
        entrance_pos = layout.entrance_position

        best_room = None
        best_distance = float('inf')

        for room in layout.rooms:
            room_center_x = room.x + room.width / 2
            room_center_y = room.y + room.height / 2

            # Check if room is on the entrance side
            if facing == "north":
                if room.y <= 5:  # Room touches or near north wall
                    dist = abs(room_center_x - entrance_pos[0])
                    if dist < best_distance:
                        best_distance = dist
                        best_room = room
            elif facing == "south":
                if room.y + room.height >= layout.plot_length - 5:
                    dist = abs(room_center_x - entrance_pos[0])
                    if dist < best_distance:
                        best_distance = dist
                        best_room = room
            elif facing == "east":
                if room.x + room.width >= layout.plot_width - 5:
                    dist = abs(room_center_y - entrance_pos[1])
                    if dist < best_distance:
                        best_distance = dist
                        best_room = room
            else:  # west
                if room.x <= 5:
                    dist = abs(room_center_y - entrance_pos[1])
                    if dist < best_distance:
                        best_distance = dist
                        best_room = room

        # If no room on entrance side, find closest room
        if best_room is None and layout.rooms:
            for room in layout.rooms:
                room_center_x = room.x + room.width / 2
                room_center_y = room.y + room.height / 2
                dist = ((room_center_x - entrance_pos[0])**2 +
                       (room_center_y - entrance_pos[1])**2)**0.5
                if dist < best_distance:
                    best_distance = dist
                    best_room = room

        return best_room

    def _place_interior_doors(self, layout: Layout) -> list[Door]:
        """Place doors between adjacent rooms"""
        doors = []
        door_count = 0
        processed_pairs = set()

        for i, room1 in enumerate(layout.rooms):
            for j, room2 in enumerate(layout.rooms):
                if i >= j:
                    continue

                pair_key = tuple(sorted([room1.id, room2.id]))
                if pair_key in processed_pairs:
                    continue

                # Check if rooms are adjacent
                door = self._create_door_between_rooms(room1, room2, door_count)
                if door:
                    doors.append(door)
                    processed_pairs.add(pair_key)
                    door_count += 1

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
