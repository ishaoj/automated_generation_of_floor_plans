"""
Constraint solver using OR-Tools for floor plan layout optimization
"""

from dataclasses import dataclass
from typing import Optional
import random

from ortools.sat.python import cp_model

from app.models.schemas import Room, Layout, FloorPlanRequest, Door, Window
from app.services.graph_builder import RoomGraphBuilder, RoomNode
from app.core.vastu_rules import get_preferred_zones, ZONE_SCORE_MATRIX
from app.core.constants import ZONES, Direction


@dataclass
class PlacedRoom:
    """Room with assigned position"""
    id: str
    room_type: str
    x: int  # Grid units from left
    y: int  # Grid units from top
    width: int  # Grid units
    height: int  # Grid units


class VastuConstraintSolver:
    """
    Constraint-based floor plan layout solver using OR-Tools CP-SAT

    Uses constraint programming to find valid room placements that:
    1. Don't overlap
    2. Stay within plot bounds
    3. Meet minimum size requirements
    4. Maximize Vastu compliance
    """

    def __init__(self, grid_size: int = 1):
        """
        Initialize solver

        Args:
            grid_size: Size of each grid unit in feet (smaller = more precise, slower)
        """
        self.grid_size = grid_size
        self.model: Optional[cp_model.CpModel] = None
        self.solver: Optional[cp_model.CpSolver] = None

    def solve(
        self,
        request: FloorPlanRequest,
        room_nodes: list[RoomNode],
        seed: Optional[int] = None,
        variation_mode: int = 0
    ) -> Optional[Layout]:
        """
        Solve for optimal room layout

        Args:
            request: Floor plan requirements
            room_nodes: List of rooms to place
            seed: Random seed for variation
            variation_mode: 0-4, different layout strategies

        Returns:
            Layout with room positions, or None if unsolvable
        """
        if seed is not None:
            random.seed(seed)

        # Convert to grid units
        plot_width = int(request.plot_width / self.grid_size)
        plot_height = int(request.plot_length / self.grid_size)

        self.model = cp_model.CpModel()

        # Determine dimension limits based on room count
        num_rooms = len(room_nodes)
        if num_rooms > 8:
            # Tight fit - allow almost full plot dimensions
            global_max_w = plot_width - 2
            global_max_h = plot_height - 2
        else:
            # Normal fit - limit to 70% of plot
            global_max_w = min(plot_width, int(plot_width * 0.7))
            global_max_h = min(plot_height, int(plot_height * 0.7))

        # Create variables for each room
        room_vars = {}
        for node in room_nodes:
            min_w = max(1, int(node.min_width / self.grid_size))
            min_h = max(1, int(node.min_height / self.grid_size))

            # Use global max dimensions
            max_w = global_max_w
            max_h = global_max_h

            room_vars[node.id] = {
                "x": self.model.NewIntVar(0, plot_width - min_w, f"{node.id}_x"),
                "y": self.model.NewIntVar(0, plot_height - min_h, f"{node.id}_y"),
                "w": self.model.NewIntVar(min_w, max_w, f"{node.id}_w"),
                "h": self.model.NewIntVar(min_h, max_h, f"{node.id}_h"),
                "node": node,
            }

        # Add constraints
        self._add_boundary_constraints(room_vars, plot_width, plot_height)
        self._add_no_overlap_constraints(room_vars)
        self._add_min_area_constraints(room_vars)
        self._add_aspect_ratio_constraints(room_vars)
        self._add_ots_center_constraint(room_vars, plot_width, plot_height)

        # Calculate room density to decide complexity level
        plot_area = plot_width * plot_height
        room_density = num_rooms / (plot_area / 100)  # rooms per 100 sq ft
        is_tight_layout = room_density > 0.5 or num_rooms > 8

        # Add Vastu objectives (soft constraints)
        # FOR TIGHT LAYOUTS: Use simplified Vastu to help solver find any solution
        if is_tight_layout:
            # Simplified Vastu - only basic direction hints, no complex zone checking
            vastu_score = self._add_simplified_vastu_objectives(
                room_vars, plot_width, plot_height, request.facing_direction
            )
        else:
            # Full Vastu for normal layouts
            vastu_score = self._add_vastu_objectives(
                room_vars, plot_width, plot_height, request.facing_direction
            )

        # Add variation-specific objectives to create diverse layouts
        variation_score = self._add_variation_objectives(
            room_vars, plot_width, plot_height, variation_mode
        )

        # NEW: Add coverage objective - maximize total room area
        # This ensures rooms expand to fill available space
        coverage_score = self._add_coverage_objective(room_vars, plot_width, plot_height)

        # Adaptive weights based on room density
        # With simplified Vastu for tight layouts, we can still optimize coverage
        if is_tight_layout:
            # Tight layout with simplified Vastu - can still maximize coverage
            # Lower Vastu weight since it's simplified, focus on coverage
            total_objective = 20 * vastu_score + 40 * coverage_score + 5 * variation_score
        elif room_density > 0.4:
            # Medium fit - balanced coverage and Vastu
            total_objective = 100 * vastu_score + 30 * coverage_score + 10 * variation_score
        else:
            # Loose fit - maximize coverage
            total_objective = 100 * vastu_score + 50 * coverage_score + 10 * variation_score

        self.model.Maximize(total_objective)

        # Solve with increased timeout for complex layouts
        self.solver = cp_model.CpSolver()
        # More time and workers for complex problems
        if num_rooms > 8:
            self.solver.parameters.max_time_in_seconds = 20.0
            self.solver.parameters.num_search_workers = 4  # Parallel search
        else:
            self.solver.parameters.max_time_in_seconds = 10.0
        self.solver.parameters.random_seed = seed if seed else random.randint(0, 10000)

        status = self.solver.Solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_layout(room_vars, request)
        else:
            return None

    def _add_variation_objectives(
        self,
        room_vars: dict,
        plot_width: int,
        plot_height: int,
        variation_mode: int
    ) -> cp_model.IntVar:
        """
        Add variation-specific objectives to create diverse layouts

        Different modes push rooms towards different areas of the plot
        """
        scores = []

        for room_id, vars in room_vars.items():
            room_score = self.model.NewIntVar(-200, 200, f"{room_id}_var_score")

            if variation_mode == 0:
                # Mode 0: Prefer compact layout (rooms towards center)
                center_dist_x = self.model.NewIntVar(0, plot_width, f"{room_id}_cdx")
                center_dist_y = self.model.NewIntVar(0, plot_height, f"{room_id}_cdy")
                self.model.AddAbsEquality(center_dist_x, vars["x"] - plot_width // 2)
                self.model.AddAbsEquality(center_dist_y, vars["y"] - plot_height // 2)
                # Negative score for distance from center (prefer closer)
                neg_dist = self.model.NewIntVar(-200, 0, f"{room_id}_neg_dist")
                self.model.Add(neg_dist == -center_dist_x - center_dist_y)
                self.model.Add(room_score == neg_dist)

            elif variation_mode == 1:
                # Mode 1: Spread out layout (rooms towards edges)
                center_dist_x = self.model.NewIntVar(0, plot_width, f"{room_id}_cdx")
                center_dist_y = self.model.NewIntVar(0, plot_height, f"{room_id}_cdy")
                self.model.AddAbsEquality(center_dist_x, vars["x"] - plot_width // 2)
                self.model.AddAbsEquality(center_dist_y, vars["y"] - plot_height // 2)
                # Positive score for distance from center (prefer farther)
                self.model.Add(room_score == center_dist_x + center_dist_y)

            elif variation_mode == 2:
                # Mode 2: Prefer left side of plot
                self.model.Add(room_score == plot_width - vars["x"])

            elif variation_mode == 3:
                # Mode 3: Prefer right side of plot
                self.model.Add(room_score == vars["x"])

            elif variation_mode == 4:
                # Mode 4: Prefer bottom of plot
                self.model.Add(room_score == vars["y"])

            else:
                # Default: no variation preference
                self.model.Add(room_score == 0)

            scores.append(room_score)

        if scores:
            total = self.model.NewIntVar(-10000, 10000, "variation_total")
            self.model.Add(total == sum(scores))
            return total
        return self.model.NewConstant(0)

    def _add_boundary_constraints(self, room_vars: dict, plot_width: int, plot_height: int):
        """Ensure rooms stay within plot boundaries"""
        for room_id, vars in room_vars.items():
            # Right edge within plot
            self.model.Add(vars["x"] + vars["w"] <= plot_width)
            # Bottom edge within plot
            self.model.Add(vars["y"] + vars["h"] <= plot_height)

    def _add_no_overlap_constraints(self, room_vars: dict):
        """Ensure no two rooms overlap"""
        room_ids = list(room_vars.keys())

        for i in range(len(room_ids)):
            for j in range(i + 1, len(room_ids)):
                r1 = room_vars[room_ids[i]]
                r2 = room_vars[room_ids[j]]

                # At least one of these must be true (disjunction)
                # r1 is left of r2, or r1 is right of r2, or r1 is above r2, or r1 is below r2
                b1 = self.model.NewBoolVar(f"left_{room_ids[i]}_{room_ids[j]}")
                b2 = self.model.NewBoolVar(f"right_{room_ids[i]}_{room_ids[j]}")
                b3 = self.model.NewBoolVar(f"above_{room_ids[i]}_{room_ids[j]}")
                b4 = self.model.NewBoolVar(f"below_{room_ids[i]}_{room_ids[j]}")

                self.model.Add(r1["x"] + r1["w"] <= r2["x"]).OnlyEnforceIf(b1)
                self.model.Add(r2["x"] + r2["w"] <= r1["x"]).OnlyEnforceIf(b2)
                self.model.Add(r1["y"] + r1["h"] <= r2["y"]).OnlyEnforceIf(b3)
                self.model.Add(r2["y"] + r2["h"] <= r1["y"]).OnlyEnforceIf(b4)

                # At least one must be true
                self.model.AddBoolOr([b1, b2, b3, b4])

    def _add_min_area_constraints(self, room_vars: dict):
        """Ensure rooms meet minimum area requirements"""
        for room_id, vars in room_vars.items():
            min_area_grid = int(vars["node"].min_area / (self.grid_size ** 2))
            # w * h >= min_area (linearized using auxiliary variable)
            area = self.model.NewIntVar(min_area_grid, 10000, f"{room_id}_area")
            self.model.AddMultiplicationEquality(area, [vars["w"], vars["h"]])
            self.model.Add(area >= min_area_grid)

    def _add_aspect_ratio_constraints(self, room_vars: dict):
        """
        Ensure rooms have realistic aspect ratios (not too elongated)

        WHY THIS IS NEEDED (for learning):
        - Without this, the solver might create a 5x33 bathroom to maximize area
        - Real rooms should have reasonable proportions
        - We enforce: width >= height/3 AND height >= width/3
        - This means aspect ratio stays between 1:3 and 3:1

        For special rooms like balconies, we allow more elongation (1:4).
        """
        for room_id, vars in room_vars.items():
            room_type = vars["node"].room_type.lower()

            # Balconies and staircases can be more elongated (1:4 ratio)
            if "balcony" in room_type or "staircase" in room_type:
                # w >= h/4 and h >= w/4 → 4w >= h and 4h >= w
                self.model.Add(4 * vars["w"] >= vars["h"])
                self.model.Add(4 * vars["h"] >= vars["w"])
            else:
                # Normal rooms: 1:3 ratio max
                # w >= h/3 and h >= w/3 → 3w >= h and 3h >= w
                self.model.Add(3 * vars["w"] >= vars["h"])
                self.model.Add(3 * vars["h"] >= vars["w"])

    def _add_ots_center_constraint(
        self,
        room_vars: dict,
        plot_width: int,
        plot_height: int
    ):
        """
        HARD CONSTRAINT: OTS (Open to Sky) must be placed in the center of the plot.

        This is a Vastu requirement - the Brahmasthan (center) should be open to sky.
        The OTS room must contain or overlap with the center point of the plot.

        For the OTS room to contain the center point (center_x, center_y):
        - x <= center_x  AND  x + w > center_x
        - y <= center_y  AND  y + h > center_y
        """
        center_x = plot_width // 2
        center_y = plot_height // 2

        for room_id, vars in room_vars.items():
            room_type = vars["node"].room_type.lower()

            if "ots" in room_type:
                # OTS must contain the center point
                # x <= center_x
                self.model.Add(vars["x"] <= center_x)
                # x + w > center_x  →  x + w >= center_x + 1
                self.model.Add(vars["x"] + vars["w"] >= center_x + 1)
                # y <= center_y
                self.model.Add(vars["y"] <= center_y)
                # y + h > center_y  →  y + h >= center_y + 1
                self.model.Add(vars["y"] + vars["h"] >= center_y + 1)

                # Also encourage OTS to be roughly square (good for courtyards)
                # Allow some flexibility: w and h within 50% of each other
                # 2*w >= h and 2*h >= w
                self.model.Add(2 * vars["w"] >= vars["h"])
                self.model.Add(2 * vars["h"] >= vars["w"])

    def _add_coverage_objective(
        self,
        room_vars: dict,
        plot_width: int,
        plot_height: int
    ) -> cp_model.IntVar:
        """
        Add objective to maximize total room coverage (fill the plot!)

        WHY THIS IS NEEDED (for learning):
        - Without this, rooms only need to meet MINIMUM size
        - The solver has no incentive to make rooms bigger
        - This objective says: "Bigger total area = Better score"
        - Result: Rooms expand to fill available space

        HOW IT WORKS:
        - For each room, we want to maximize (width + height)
        - This is simpler than area multiplication and works better with the solver
        - Bigger dimensions = higher score = larger rooms
        """
        dimension_scores = []

        for room_id, vars in room_vars.items():
            # Simple: maximize sum of dimensions (width + height)
            # This encourages rooms to grow in both directions
            dim_sum = self.model.NewIntVar(0, plot_width + plot_height, f"{room_id}_dim_sum")
            self.model.Add(dim_sum == vars["w"] + vars["h"])
            dimension_scores.append(dim_sum)

        # Total dimension score
        total_dim_score = self.model.NewIntVar(0, (plot_width + plot_height) * len(room_vars), "total_dim_score")
        self.model.Add(total_dim_score == sum(dimension_scores))

        return total_dim_score

    def _add_simplified_vastu_objectives(
        self,
        room_vars: dict,
        plot_width: int,
        plot_height: int,
        facing: str
    ) -> cp_model.IntVar:
        """
        Simplified Vastu objectives for tight layouts (many rooms)

        WHY THIS IS NEEDED (for learning):
        - The full Vastu objectives create 50+ boolean variables and complex constraints
        - For 10+ rooms, this makes the problem too hard for the solver
        - This simplified version uses LINEAR position preferences instead
        - Much faster to solve, still gives reasonable Vastu alignment

        HOW IT WORKS:
        - Kitchen: prefers southeast → reward high (x + y)
        - Living room: prefers northeast → reward high x, low y
        - Bedrooms: prefer south/southwest → reward high y, low x
        - Bathrooms: prefer northwest → reward low x, low y

        NOTE: We use simple addition/subtraction (no division) because
        OR-Tools IntVar doesn't support Python's // operator directly.
        """
        scores = []
        max_score = plot_width + plot_height

        for room_id, vars in room_vars.items():
            node = vars["node"]
            room_type = node.room_type.lower()

            # Create a simple position-based score
            room_score = self.model.NewIntVar(-max_score, max_score, f"{room_id}_simple_vscore")

            if "kitchen" in room_type:
                # Kitchen prefers southeast: high x + high y
                self.model.Add(room_score == vars["x"] + vars["y"])

            elif "living" in room_type:
                # Living room prefers north/northeast: low y, prefer high x
                self.model.Add(room_score == vars["x"] - vars["y"])

            elif "master" in room_type or "bedroom" in room_type:
                # Bedrooms prefer south/southwest: high y
                self.model.Add(room_score == vars["y"])

            elif "bathroom" in room_type or "toilet" in room_type:
                # Bathrooms prefer northwest: low x + low y (inverted = high score for low values)
                self.model.Add(room_score == plot_width - vars["x"])

            elif "puja" in room_type or "prayer" in room_type:
                # Puja room prefers northeast: high x, low y
                self.model.Add(room_score == vars["x"] - vars["y"])

            elif "parking" in room_type or "garage" in room_type:
                # Parking prefers northwest: low x
                self.model.Add(room_score == plot_width - vars["x"])

            elif "dining" in room_type:
                # Dining prefers west: low x
                self.model.Add(room_score == plot_width - vars["x"])

            elif "balcony" in room_type:
                # Balcony prefers east or north: high x or low y
                self.model.Add(room_score == vars["x"])

            elif "staircase" in room_type:
                # Staircase prefers south/west: high y, low x
                self.model.Add(room_score == vars["y"] - vars["x"])

            elif "ots" in room_type:
                # OTS (Open to Sky) MUST be in center (Brahmasthan)
                # Score highest when closest to center of plot
                center_x = plot_width // 2
                center_y = plot_height // 2
                # Use room corner position (simpler than center) for center preference
                # Prefer rooms placed near center of plot
                dist_x = self.model.NewIntVar(0, plot_width, f"{room_id}_dist_x")
                dist_y = self.model.NewIntVar(0, plot_height, f"{room_id}_dist_y")
                self.model.AddAbsEquality(dist_x, vars["x"] - center_x)
                self.model.AddAbsEquality(dist_y, vars["y"] - center_y)
                # Invert: closer to center = higher score
                self.model.Add(room_score == max_score - dist_x - dist_y)

            else:
                # Default: no preference
                self.model.Add(room_score == 0)

            scores.append(room_score)

        # Total simplified Vastu score
        if scores:
            total_score = self.model.NewIntVar(-max_score * len(scores), max_score * len(scores), "total_simple_vastu")
            self.model.Add(total_score == sum(scores))
            return total_score
        else:
            return self.model.NewConstant(0)

    def _add_vastu_objectives(
        self,
        room_vars: dict,
        plot_width: int,
        plot_height: int,
        facing: str
    ) -> cp_model.IntVar:
        """
        Add Vastu compliance as optimization objective

        Returns a score variable to maximize
        """
        scores = []

        for room_id, vars in room_vars.items():
            node = vars["node"]
            preferred_zones = node.preferred_zones

            if not preferred_zones:
                continue

            # Create zone membership indicators
            zone_scores = []

            for zone in preferred_zones:
                # Get zone bounds (as fractions of plot)
                zone_bounds = ZONES.get(Direction(zone))
                if not zone_bounds:
                    continue

                x_min = int(zone_bounds[0] * plot_width)
                y_min = int(zone_bounds[1] * plot_height)
                x_max = int(zone_bounds[2] * plot_width)
                y_max = int(zone_bounds[3] * plot_height)

                # Check if room center is in this zone
                center_x = self.model.NewIntVar(0, plot_width, f"{room_id}_cx")
                center_y = self.model.NewIntVar(0, plot_height, f"{room_id}_cy")

                # center = position + dimension/2 (approximated)
                self.model.Add(2 * center_x == 2 * vars["x"] + vars["w"])
                self.model.Add(2 * center_y == 2 * vars["y"] + vars["h"])

                # Is center in zone?
                in_zone = self.model.NewBoolVar(f"{room_id}_in_{zone}")

                in_x = self.model.NewBoolVar(f"{room_id}_inx_{zone}")
                in_y = self.model.NewBoolVar(f"{room_id}_iny_{zone}")

                self.model.Add(center_x >= x_min).OnlyEnforceIf(in_x)
                self.model.Add(center_x <= x_max).OnlyEnforceIf(in_x)
                self.model.Add(center_x < x_min).OnlyEnforceIf(in_x.Not())

                self.model.Add(center_y >= y_min).OnlyEnforceIf(in_y)
                self.model.Add(center_y <= y_max).OnlyEnforceIf(in_y)
                self.model.Add(center_y < y_min).OnlyEnforceIf(in_y.Not())

                self.model.AddBoolAnd([in_x, in_y]).OnlyEnforceIf(in_zone)

                # Score contribution if in preferred zone
                zone_score = self.model.NewIntVar(0, 100, f"{room_id}_{zone}_score")
                self.model.Add(zone_score == 100).OnlyEnforceIf(in_zone)
                self.model.Add(zone_score == 0).OnlyEnforceIf(in_zone.Not())

                zone_scores.append(zone_score)

            if zone_scores:
                # Take max of zone scores for this room
                room_score = self.model.NewIntVar(0, 100, f"{room_id}_vscore")
                self.model.AddMaxEquality(room_score, zone_scores)
                scores.append(room_score)

        # Total Vastu score
        if scores:
            total_score = self.model.NewIntVar(0, 100 * len(scores), "total_vastu")
            self.model.Add(total_score == sum(scores))
            return total_score
        else:
            return self.model.NewConstant(0)

    def _extract_layout(self, room_vars: dict, request: FloorPlanRequest) -> Layout:
        """Extract layout from solved model"""
        rooms = []

        for room_id, vars in room_vars.items():
            x = self.solver.Value(vars["x"]) * self.grid_size
            y = self.solver.Value(vars["y"]) * self.grid_size
            w = self.solver.Value(vars["w"]) * self.grid_size
            h = self.solver.Value(vars["h"]) * self.grid_size

            # Determine zone based on position
            zone = self._get_zone(
                x + w/2, y + h/2,
                request.plot_width, request.plot_length
            )

            rooms.append(Room(
                id=room_id,
                room_type=vars["node"].room_type,
                x=x,
                y=y,
                width=w,
                height=h,
                area=w * h,
                zone=zone,
            ))

        # Determine entrance position based on facing
        entrance_pos = self._get_entrance_position(request)

        # Get open walls from request (for window placement and rendering)
        open_walls = request.get_open_walls()

        # Generate doors between adjacent rooms
        doors = self._generate_doors(rooms, request)

        # Generate windows on exterior walls
        windows = self._generate_windows(rooms, open_walls, request)

        return Layout(
            rooms=rooms,
            doors=doors,
            windows=windows,
            plot_length=request.plot_length,
            plot_width=request.plot_width,
            facing_direction=request.facing_direction,
            entrance_position=entrance_pos,
            open_walls=open_walls,
            plot_type=request.plot_type,
        )

    def _get_zone(self, x: float, y: float, plot_width: float, plot_height: float) -> str:
        """Determine which Vastu zone a point falls in"""
        # Normalize to 0-1
        nx = x / plot_width
        ny = y / plot_height

        # Determine zone
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

    def _get_entrance_position(self, request: FloorPlanRequest) -> tuple[float, float]:
        """Calculate entrance position based on facing direction"""
        facing = request.facing_direction
        pw = request.plot_width
        pl = request.plot_length

        # Position entrance slightly off-center towards auspicious direction
        if facing == "north":
            return (pw * 0.55, 0)  # Slightly east of center on north wall
        elif facing == "south":
            return (pw * 0.55, pl)
        elif facing == "east":
            return (pw, pl * 0.45)  # Slightly north of center on east wall
        else:  # west
            return (0, pl * 0.45)

    def _generate_doors(self, rooms: list[Room], request: FloorPlanRequest) -> list[Door]:
        """
        Generate doors between adjacent rooms and at the main entrance.

        Door placement rules:
        1. Main entrance door on the facing wall
        2. Internal doors between adjacent rooms
        3. Bathroom doors from connected bedroom/corridor
        4. Kitchen door from dining/living area
        """
        doors = []
        door_id = 0

        # Main entrance door
        entrance_pos = self._get_entrance_position(request)
        facing = request.facing_direction
        orientation = "horizontal" if facing in ["north", "south"] else "vertical"

        doors.append(Door(
            id=f"door_{door_id}",
            x=entrance_pos[0],
            y=entrance_pos[1],
            width=3.5,
            orientation=orientation,
            connects=["exterior", "living_room"],
            is_main_entrance=True
        ))
        door_id += 1

        # Find adjacent rooms and add doors between them
        for i, room1 in enumerate(rooms):
            for room2 in rooms[i + 1:]:
                # Skip OTS - it doesn't need doors
                if room1.room_type == "ots" or room2.room_type == "ots":
                    continue

                # Check if rooms share a wall
                door_pos = self._find_shared_wall(room1, room2)
                if door_pos:
                    x, y, orient = door_pos
                    doors.append(Door(
                        id=f"door_{door_id}",
                        x=x,
                        y=y,
                        width=3.0,
                        orientation=orient,
                        connects=[room1.id, room2.id],
                        is_main_entrance=False
                    ))
                    door_id += 1

        return doors

    def _find_shared_wall(self, room1: Room, room2: Room) -> tuple[float, float, str] | None:
        """
        Find if two rooms share a wall and return door position.

        Returns (x, y, orientation) or None if no shared wall.
        """
        tolerance = 1.0  # Allow 1 ft tolerance for alignment

        # Room 1 boundaries
        r1_left = room1.x
        r1_right = room1.x + room1.width
        r1_top = room1.y
        r1_bottom = room1.y + room1.height

        # Room 2 boundaries
        r2_left = room2.x
        r2_right = room2.x + room2.width
        r2_top = room2.y
        r2_bottom = room2.y + room2.height

        # Check vertical shared wall (room1 right = room2 left or vice versa)
        if abs(r1_right - r2_left) <= tolerance:
            # Room1 is to the left of Room2
            overlap_top = max(r1_top, r2_top)
            overlap_bottom = min(r1_bottom, r2_bottom)
            if overlap_bottom - overlap_top >= 3:  # Min 3ft overlap for door
                door_y = (overlap_top + overlap_bottom) / 2
                return (r1_right, door_y, "vertical")

        if abs(r2_right - r1_left) <= tolerance:
            # Room2 is to the left of Room1
            overlap_top = max(r1_top, r2_top)
            overlap_bottom = min(r1_bottom, r2_bottom)
            if overlap_bottom - overlap_top >= 3:
                door_y = (overlap_top + overlap_bottom) / 2
                return (r1_left, door_y, "vertical")

        # Check horizontal shared wall (room1 bottom = room2 top or vice versa)
        if abs(r1_bottom - r2_top) <= tolerance:
            # Room1 is above Room2
            overlap_left = max(r1_left, r2_left)
            overlap_right = min(r1_right, r2_right)
            if overlap_right - overlap_left >= 3:  # Min 3ft overlap for door
                door_x = (overlap_left + overlap_right) / 2
                return (door_x, r1_bottom, "horizontal")

        if abs(r2_bottom - r1_top) <= tolerance:
            # Room2 is above Room1
            overlap_left = max(r1_left, r2_left)
            overlap_right = min(r1_right, r2_right)
            if overlap_right - overlap_left >= 3:
                door_x = (overlap_left + overlap_right) / 2
                return (door_x, r1_top, "horizontal")

        return None

    def _generate_windows(
        self,
        rooms: list[Room],
        open_walls: list[str],
        request: FloorPlanRequest
    ) -> list[Window]:
        """
        Generate windows on exterior walls of rooms.

        Window placement rules:
        1. Only on open/exterior walls (not shared with neighbors)
        2. Bedrooms: 1-2 windows
        3. Living room: 2 windows preferred
        4. Kitchen: 1 window (for ventilation)
        5. Bathroom: 1 small window (high placement)
        6. No windows for parking, staircase, store
        """
        windows = []
        window_id = 0

        # Rooms that should have windows
        window_rooms = {
            "master_bedroom": 2,
            "bedroom": 1,
            "living_room": 2,
            "kitchen": 1,
            "dining": 1,
            "puja_room": 1,
            "bathroom": 1,
        }

        for room in rooms:
            room_type = room.room_type
            if room_type not in window_rooms:
                continue

            num_windows = window_rooms[room_type]
            room_windows = self._place_windows_for_room(
                room, open_walls, request, num_windows, window_id
            )
            windows.extend(room_windows)
            window_id += len(room_windows)

        return windows

    def _place_windows_for_room(
        self,
        room: Room,
        open_walls: list[str],
        request: FloorPlanRequest,
        num_windows: int,
        start_id: int
    ) -> list[Window]:
        """Place windows for a single room on available exterior walls."""
        windows = []
        plot_w = request.plot_width
        plot_h = request.plot_length

        # Determine which room walls touch the exterior
        exterior_walls = []

        # Check north wall (room top at y=0)
        if room.y <= 1 and "north" in open_walls:
            exterior_walls.append(("north", room.x + room.width / 2, room.y, "horizontal"))

        # Check south wall (room bottom at y=plot_length)
        if room.y + room.height >= plot_h - 1 and "south" in open_walls:
            exterior_walls.append(("south", room.x + room.width / 2, room.y + room.height, "horizontal"))

        # Check west wall (room left at x=0)
        if room.x <= 1 and "west" in open_walls:
            exterior_walls.append(("west", room.x, room.y + room.height / 2, "vertical"))

        # Check east wall (room right at x=plot_width)
        if room.x + room.width >= plot_w - 1 and "east" in open_walls:
            exterior_walls.append(("east", room.x + room.width, room.y + room.height / 2, "vertical"))

        # Place windows on available exterior walls
        for i, (wall, x, y, orient) in enumerate(exterior_walls[:num_windows]):
            window_width = 3.0 if room.room_type == "bathroom" else 4.0
            windows.append(Window(
                id=f"window_{start_id + i}",
                x=x,
                y=y,
                width=window_width,
                orientation=orient,
                room_id=room.id,
                wall=wall
            ))

        return windows
