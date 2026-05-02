"""
Main floor plan generator.

Pipeline:
  1. RoomGraphBuilder        → room dependency graph from user request
  2. VastuConstraintSolver   → initial layout (room positions via OR-Tools)
  3. OpeningsPlacer          → doors & windows (rule-based)
  4. MLInferencePipeline     → GNN refines room positions
  5. FloorPlanRenderer       → deterministic 2048×2048 architectural image
  6. VastuScorer             → compliance score
"""

import sys
import uuid
import random
import logging
from pathlib import Path
from typing import Optional

# Ensure backend/ and project root are on sys.path
_HERE = Path(__file__).resolve()
_BACK = _HERE.parent.parent.parent        # backend/
_ROOT = _BACK.parent                      # project root
for _p in [str(_ROOT), str(_BACK)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.models.schemas import (
    FloorPlanRequest,
    GeneratedPlan,
    Layout,
    Room,
)
from app.services.graph_builder       import RoomGraphBuilder
from app.services.constraint_solver   import VastuConstraintSolver
from app.services.vastu_scorer        import VastuScorer
from app.services.openings_placer     import OpeningsPlacer
from app.services.floor_plan_renderer import FloorPlanRenderer
from app.services.layout_validator    import LayoutValidator
from app.services.design_analyzer     import DesignAnalyzer
from app.config                       import settings

from inference.pipeline import MLInferencePipeline

logger = logging.getLogger(__name__)


class FloorPlanGenerator:
    """
    Main floor plan generation service.

    Raises RuntimeError on construction if ML models are not trained yet.
    Call `train_models()` instructions will be shown.
    """

    def __init__(self):
        self.graph_builder     = RoomGraphBuilder()
        self.constraint_solver = VastuConstraintSolver(grid_size=1)
        self.scorer            = VastuScorer()
        self.openings_placer   = OpeningsPlacer()
        self.renderer          = FloorPlanRenderer()
        self.validator         = LayoutValidator()
        self.design_analyzer   = DesignAnalyzer()

        # ── Load GNN pipeline ─────────────────────────────────────────────
        config_path = Path(settings.pipeline_config)
        if config_path.exists():
            self.ml_pipeline = MLInferencePipeline.from_config(str(config_path))
        else:
            self.ml_pipeline = MLInferencePipeline(
                gnn_path = settings.gnn_model_path,
                device   = settings.ml_device,
            )
        logger.info("FloorPlanGenerator ready (deterministic renderer + GNN).")

    # ------------------------------------------------------------------

    def generate(self, request: FloorPlanRequest) -> list[GeneratedPlan]:
        """
        Generate floor plan variations.

        Steps per variation:
          1. Constraint solver → initial room layout
          2. GNN → refined positions
          3. FloorPlanRenderer → 2048×2048 architectural image
          4. Vastu scorer → compliance score

        Returns list of GeneratedPlan sorted by Vastu score (best first).
        """
        # Build room graph
        graph      = self.graph_builder.build_from_request(request)
        room_nodes = self.graph_builder.get_nodes_by_priority()

        is_valid, message = self.graph_builder.validate_against_plot(request.plot_area)
        if not is_valid:
            raise ValueError(message)

        plans: list[GeneratedPlan] = []
        variation_modes = [0, 1, 2, 3, 4]
        MAX_RETRIES = 3   # validation retries per variation

        for variation_idx in range(request.num_variations):
            if len(plans) >= request.num_variations:
                break

            mode = variation_modes[variation_idx % len(variation_modes)]
            base_seed = random.randint(0, 100_000) + variation_idx * 1000

            layout     = None
            violations: list[str] = []

            # ── Retry loop: regenerate until constraints pass ────────────
            for attempt in range(MAX_RETRIES):
                seed = base_seed + attempt * 317

                candidate = self.constraint_solver.solve(
                    request, room_nodes, seed=seed, variation_mode=mode
                )
                if candidate is None:
                    continue

                candidate = self.openings_placer.place_openings(candidate)
                ok, cand_violations = self.validator.validate(candidate)

                if ok:
                    layout     = candidate
                    violations = []
                    break

                # Track best (fewest violations) fallback
                if layout is None or len(cand_violations) < len(violations):
                    layout     = candidate
                    violations = cand_violations

            if layout is None:
                continue

            # ── Dedup check ──────────────────────────────────────────────
            if self._is_duplicate(layout, [p.layout for p in plans]):
                continue

            # ── Vastu score ──────────────────────────────────────────────
            vastu_score = self.scorer.score(layout)

            # Append any unresolved rule violations as suggestions
            if violations:
                vastu_score.suggestions = (
                    [f"[Rule violation] {v}" for v in violations[:3]]
                    + vastu_score.suggestions
                )[:5]

            # ── ML pipeline — GNN refines positions + Pix2Pix ───────────
            layout_dict = self._layout_to_dict(layout)
            images      = self.ml_pipeline.generate(layout_dict)

            # ── Apply GNN-refined positions to the layout ─────────────────
            # The GNN was trained on real CubiCasa5k floor plans and nudges
            # room positions toward more architecturally natural arrangements.
            # We clamp its output to the plot bounds and fall back to the
            # original constraint-solver layout if refinement is too disruptive.
            refined_layout = self._build_refined_layout(layout, images["refined_rooms"])

            # Re-place doors/windows on the GNN-refined room positions
            refined_layout = self.openings_placer.place_openings(refined_layout)

            # ── Render the GNN-refined layout (primary image) ─────────────
            arch_image = self.renderer.render(refined_layout)

            # ── Architectural design quality suggestions ───────────────────
            design_suggestions = self.design_analyzer.analyze(refined_layout)

            plans.append(GeneratedPlan(
                plan_id            = str(uuid.uuid4())[:8],
                layout             = refined_layout,
                vastu_score        = vastu_score,
                image_base64       = arch_image,
                image_format       = "png",
                design_suggestions = design_suggestions,
            ))

        plans.sort(key=lambda p: p.vastu_score.total, reverse=True)

        if not plans:
            raise ValueError(
                "Could not generate valid floor plans for the given requirements. "
                "Try increasing plot size or reducing number of rooms."
            )

        return plans

    # ------------------------------------------------------------------
    # GNN refinement helpers
    # ------------------------------------------------------------------

    def _build_refined_layout(
        self,
        original: Layout,
        refined_rooms: list[dict],
    ) -> Layout:
        """
        Merge GNN-refined room positions into a new Layout.

        Safety rules applied before accepting refinement:
          1. Clamp every room to stay inside the plot boundary.
          2. Ensure minimum 1 ft dimension so rooms don't collapse.
          3. If total overlap area across all pairs exceeds 15% of plot
             area → fall back to the original layout (GNN diverged).

        Preserves: room_type, id, zone, plot metadata, open_walls.
        """
        pw = original.plot_width
        pl = original.plot_length

        # ── Build clamped Room objects ────────────────────────────────────
        clamped: list[Room] = []
        for orig, ref in zip(original.rooms, refined_rooms):
            x = float(ref.get("x", orig.x))
            y = float(ref.get("y", orig.y))
            w = float(ref.get("width",  orig.width))
            h = float(ref.get("height", orig.height))

            # Clamp position so room stays fully inside the plot
            w  = max(1.0, min(w, pw))
            h  = max(1.0, min(h, pl))
            x  = max(0.0, min(x, pw - w))
            y  = max(0.0, min(y, pl - h))

            clamped.append(Room(
                id        = orig.id,
                room_type = orig.room_type,
                x=x, y=y, width=w, height=h,
                area      = w * h,
                zone      = orig.zone,   # Vastu zone kept from constraint solver
            ))

        # ── Overlap quality check ─────────────────────────────────────────
        if self._refinement_too_disruptive(clamped, pw * pl):
            logger.warning(
                "GNN refinement produced excessive overlaps — "
                "falling back to constraint-solver layout."
            )
            return original

        return Layout(
            rooms            = clamped,
            doors            = [],   # openings_placer will repopulate
            windows          = [],
            plot_length      = original.plot_length,
            plot_width       = original.plot_width,
            facing_direction = original.facing_direction,
            entrance_position= original.entrance_position,
            open_walls       = original.open_walls,
            plot_type        = original.plot_type,
        )

    @staticmethod
    def _refinement_too_disruptive(
        rooms: list[Room], plot_area: float, max_overlap_ratio: float = 0.15
    ) -> bool:
        """
        Return True if pairwise room overlap exceeds max_overlap_ratio of
        the total plot area — indicates the GNN produced invalid positions.
        """
        total_overlap = 0.0
        n = len(rooms)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = rooms[i], rooms[j]
                ox = max(0.0, min(a.x + a.width,  b.x + b.width)  - max(a.x, b.x))
                oy = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
                total_overlap += ox * oy

        return (total_overlap / max(plot_area, 1.0)) > max_overlap_ratio

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layout_to_dict(layout: Layout) -> dict:
        """Convert a Pydantic Layout to a plain dict for the ML pipeline."""
        return {
            "plot_width":       layout.plot_width,
            "plot_length":      layout.plot_length,
            "facing_direction": layout.facing_direction,
            "rooms": [
                {
                    "id":        r.id,
                    "room_type": r.room_type,
                    "x":         r.x,
                    "y":         r.y,
                    "width":     r.width,
                    "height":    r.height,
                    "area":      r.area,
                    "zone":      r.zone,
                }
                for r in layout.rooms
            ],
        }

    def _is_duplicate(
        self, layout: Layout, existing: list[Layout], threshold: float = 0.25
    ) -> bool:
        for ex in existing:
            if self._layouts_similar(layout, ex, threshold):
                return True
        return False

    @staticmethod
    def _layouts_similar(l1: Layout, l2: Layout, threshold: float = 0.25) -> bool:
        if len(l1.rooms) != len(l2.rooms):
            return False
        rooms1 = {r.room_type: (r.x, r.y) for r in l1.rooms}
        rooms2 = {r.room_type: (r.x, r.y) for r in l2.rooms}
        similar = sum(
            1 for rt, (x1, y1) in rooms1.items()
            if rt in rooms2
            and abs(x1 - rooms2[rt][0]) / max(l1.plot_width,  1) < threshold
            and abs(y1 - rooms2[rt][1]) / max(l1.plot_length, 1) < threshold
        )
        return similar / max(len(rooms1), 1) > 0.7
