"""
Main floor plan generator - orchestrates the generation pipeline
"""

import uuid
import random
from typing import Optional

from app.models.schemas import (
    FloorPlanRequest,
    GeneratedPlan,
    Layout,
)
from app.services.graph_builder import RoomGraphBuilder
from app.services.constraint_solver import VastuConstraintSolver
from app.services.renderer import FloorPlanRenderer
from app.services.vastu_scorer import VastuScorer
from app.services.openings_placer import OpeningsPlacer
from app.config import settings

# ML imports (optional)
try:
    from ml.inference import GNNRefinerInference
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML module not available. Using constraint-only mode.")


class FloorPlanGenerator:
    """
    Main floor plan generation service

    Pipeline:
    1. Build room graph from requirements
    2. Use constraint solver to find valid layouts
    3. Generate multiple variations
    4. Score each layout for Vastu compliance
    5. Render to images
    """

    def __init__(self):
        self.graph_builder = RoomGraphBuilder()
        self.constraint_solver = VastuConstraintSolver(grid_size=1)
        self.renderer = FloorPlanRenderer(
            width=settings.image_width,
            height=settings.image_height
        )
        self.scorer = VastuScorer()
        self.openings_placer = OpeningsPlacer()

        # Initialize ML refiner if enabled
        self.gnn_refiner = None
        if settings.use_gnn_refiner and ML_AVAILABLE:
            try:
                self.gnn_refiner = GNNRefinerInference(
                    model_path=settings.gnn_model_path,
                    device=settings.ml_device,
                    enable=True,
                )
                print("✓ GNN refiner initialized")
            except Exception as e:
                print(f"Warning: Could not initialize GNN refiner: {e}")
                print("Continuing with constraint-only mode")
                self.gnn_refiner = None

    def generate(self, request: FloorPlanRequest) -> list[GeneratedPlan]:
        """
        Generate floor plan variations based on request

        Args:
            request: User's floor plan requirements

        Returns:
            List of generated plans with images and scores
        """
        # Build room graph
        graph = self.graph_builder.build_from_request(request)
        room_nodes = self.graph_builder.get_nodes_by_priority()

        # Validate plot size
        is_valid, message = self.graph_builder.validate_against_plot(request.plot_area)
        if not is_valid:
            raise ValueError(message)

        # Generate variations using different layout strategies
        plans = []
        variation_modes = [0, 1, 2, 3, 4]  # Different layout strategies

        for variation_idx in range(request.num_variations):
            if len(plans) >= request.num_variations:
                break

            # Use different variation mode for each plan
            variation_mode = variation_modes[variation_idx % len(variation_modes)]

            # Generate unique seed
            seed = random.randint(0, 100000) + variation_idx * 1000

            # Solve for layout with variation mode
            layout = self.constraint_solver.solve(
                request, room_nodes, seed=seed, variation_mode=variation_mode
            )

            if layout is None:
                continue

            # Refine layout with GNN (if enabled)
            if self.gnn_refiner is not None:
                try:
                    layout = self.gnn_refiner.refine(
                        layout,
                        request=request,
                        max_adjustment=settings.gnn_max_adjustment,
                        min_room_size=settings.gnn_min_room_size,
                    )
                    # Fix any overlaps introduced by GNN
                    layout = self._fix_overlaps(layout)
                except Exception as e:
                    print(f"Warning: GNN refinement failed: {e}")
                    # Continue with un-refined layout

            # Add doors and windows to layout
            layout = self.openings_placer.place_openings(layout)

            # Check for duplicate layouts (with relaxed threshold)
            if self._is_duplicate(layout, [p.layout for p in plans], threshold=0.15):
                # Try again with different seed
                layout = self.constraint_solver.solve(
                    request, room_nodes, seed=seed + 500, variation_mode=variation_mode
                )
                if layout is None or self._is_duplicate(layout, [p.layout for p in plans], threshold=0.15):
                    continue
                # Add doors and windows to the retried layout
                layout = self.openings_placer.place_openings(layout)

            # Score the layout
            vastu_score = self.scorer.score(layout)

            # Render to image
            image_base64 = self.renderer.render_to_base64(layout, vastu_score.total)

            # Create plan
            plan = GeneratedPlan(
                plan_id=str(uuid.uuid4())[:8],
                layout=layout,
                vastu_score=vastu_score,
                image_base64=image_base64,
                image_format="png",
            )

            plans.append(plan)

        # Sort by Vastu score (highest first)
        plans.sort(key=lambda p: p.vastu_score.total, reverse=True)

        if not plans:
            raise ValueError(
                "Could not generate valid floor plans for the given requirements. "
                "Try increasing plot size or reducing number of rooms."
            )

        return plans

    def _is_duplicate(self, layout: Layout, existing_layouts: list[Layout], threshold: float = 0.1) -> bool:
        """Check if layout is too similar to existing ones"""
        for existing in existing_layouts:
            if self._layouts_similar(layout, existing, threshold=threshold):
                return True
        return False

    def _layouts_similar(self, l1: Layout, l2: Layout, threshold: float = 0.1) -> bool:
        """
        Check if two layouts are similar (room positions within threshold)

        Args:
            l1, l2: Layouts to compare
            threshold: Fraction of plot size considered "same position"
        """
        if len(l1.rooms) != len(l2.rooms):
            return False

        # Create room position maps
        rooms1 = {r.room_type: (r.x, r.y) for r in l1.rooms}
        rooms2 = {r.room_type: (r.x, r.y) for r in l2.rooms}

        # Check if all rooms are in similar positions
        similar_count = 0
        for room_type, (x1, y1) in rooms1.items():
            if room_type in rooms2:
                x2, y2 = rooms2[room_type]
                dx = abs(x1 - x2) / l1.plot_width
                dy = abs(y1 - y2) / l1.plot_length

                if dx < threshold and dy < threshold:
                    similar_count += 1

        # If more than 70% of rooms are in similar positions, consider duplicate
        return similar_count / len(rooms1) > 0.7

    def _fix_overlaps(self, layout: Layout) -> Layout:
        """
        Fix any room overlaps in layout by slight adjustments

        This is needed after GNN refinement which might introduce small overlaps.
        """
        # Simple overlap resolution: shift overlapping rooms slightly
        rooms = layout.rooms
        max_iterations = 10

        for iteration in range(max_iterations):
            has_overlap = False

            for i, room1 in enumerate(rooms):
                for j, room2 in enumerate(rooms):
                    if i >= j:
                        continue

                    # Check if rooms overlap
                    if self._rooms_overlap(room1, room2):
                        has_overlap = True

                        # Shift room2 slightly away from room1
                        cx1 = room1.x + room1.width / 2
                        cy1 = room1.y + room1.height / 2
                        cx2 = room2.x + room2.width / 2
                        cy2 = room2.y + room2.height / 2

                        # Direction to move room2
                        dx = cx2 - cx1
                        dy = cy2 - cy1
                        dist = (dx**2 + dy**2)**0.5

                        if dist > 0:
                            # Normalize and scale
                            shift = 2.0  # feet
                            dx = (dx / dist) * shift
                            dy = (dy / dist) * shift

                            # Apply shift
                            room2.x += dx
                            room2.y += dy

                            # Clamp to plot bounds
                            room2.x = max(0, min(room2.x, layout.plot_width - room2.width))
                            room2.y = max(0, min(room2.y, layout.plot_length - room2.height))

            # If no overlaps, we're done
            if not has_overlap:
                break

        return layout

    def _rooms_overlap(self, room1, room2) -> bool:
        """Check if two rooms overlap"""
        return not (
            room1.x + room1.width <= room2.x or
            room2.x + room2.width <= room1.x or
            room1.y + room1.height <= room2.y or
            room2.y + room2.height <= room1.y
        )


class MockFloorPlanGenerator:
    """
    Mock generator for testing without full constraint solver

    Generates simple grid-based layouts
    """

    def __init__(self):
        self.renderer = FloorPlanRenderer()
        self.scorer = VastuScorer()

    def generate(self, request: FloorPlanRequest) -> list[GeneratedPlan]:
        """Generate mock floor plans for testing"""
        plans = []

        for i in range(request.num_variations):
            layout = self._create_mock_layout(request, seed=i)
            vastu_score = self.scorer.score(layout)
            image_base64 = self.renderer.render_to_base64(layout, vastu_score.total)

            plan = GeneratedPlan(
                plan_id=str(uuid.uuid4())[:8],
                layout=layout,
                vastu_score=vastu_score,
                image_base64=image_base64,
                image_format="png",
            )
            plans.append(plan)

        return plans

    def _create_mock_layout(self, request: FloorPlanRequest, seed: int = 0) -> Layout:
        """Create a simple grid-based mock layout"""
        from app.models.schemas import Room

        random.seed(seed)
        rooms = []

        # Simple grid placement
        pw = request.plot_width
        pl = request.plot_length

        # Place rooms in a grid pattern
        room_list = []

        # Living room - north/east area
        if request.include_living_room:
            room_list.append(("living_room", pw * 0.5, pl * 0.4, pw * 0.5, 0, "northeast"))

        # Master bedroom - southwest
        room_list.append(("master_bedroom", pw * 0.4, pl * 0.35, 0, pl * 0.55, "southwest"))

        # Kitchen - southeast
        if request.include_kitchen:
            room_list.append(("kitchen", pw * 0.3, pl * 0.25, pw * 0.7, pl * 0.7, "southeast"))

        # Bathrooms
        for i in range(request.num_bathrooms):
            room_id = "bathroom" if i == 0 else f"bathroom_{i+1}"
            room_list.append((room_id, pw * 0.15, pl * 0.15, pw * 0.1, pl * 0.4 + i * pl * 0.2, "northwest"))

        # Additional bedrooms
        for i in range(1, request.num_bedrooms):
            room_id = f"bedroom_{i+1}"
            room_list.append((room_id, pw * 0.3, pl * 0.3, pw * 0.35, pl * 0.05 + i * pl * 0.35, "west"))

        for room_type, width, height, x, y, zone in room_list:
            rooms.append(Room(
                id=room_type,
                room_type=room_type.split("_")[0] if "_" in room_type and room_type[-1].isdigit() else room_type,
                x=x,
                y=y,
                width=width,
                height=height,
                area=width * height,
                zone=zone,
            ))

        entrance_pos = (pw * 0.5, 0) if request.facing_direction == "north" else (pw * 0.5, pl)

        return Layout(
            rooms=rooms,
            plot_length=pl,
            plot_width=pw,
            facing_direction=request.facing_direction,
            entrance_position=entrance_pos,
        )
