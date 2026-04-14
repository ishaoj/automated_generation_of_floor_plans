"""
Floor plan renderer — architectural style inspired by CubiCasa5k dataset.

Design principles:
- Solid filled walls (plot background = wall color, rooms cut out as light fills)
- Exterior wall hatching (diagonal lines like architectural drawings)
- Muted, professional room colors
- Proper door swing arcs
- CubiCasa-style window representation (triple parallel lines)
- Clean sans-serif labels with dimensions
"""

import io
import math
import base64
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

from app.models.schemas import Layout, Room, Door, Window
from app.config import settings


# Architectural color palette (very light, muted — like CubiCasa renders)
ROOM_COLORS = {
    "master_bedroom":  "#F5EBE0",   # warm cream
    "bedroom":         "#EBF0F5",   # cool white-blue
    "living_room":     "#EBF5EB",   # very light green
    "kitchen":         "#FFF8EC",   # warm white
    "bathroom":        "#EBF5FF",   # pale blue
    "dining":          "#F5F5EB",   # light khaki
    "puja_room":       "#FFF9E6",   # soft gold
    "storage":         "#F0F0F0",   # light gray
    "utility":         "#F0F0F0",
    "balcony":         "#E8F4EA",   # soft green
    "parking":         "#EBEBEB",   # silver-gray
    "staircase":       "#F0EFEA",
    "corridor":        "#F7F7F5",
    "ots":             "#D6EAF8",   # sky blue
}

WALL_COLOR        = "#1C1C1E"   # near-black (CubiCasa wall color)
WALL_BODY_COLOR   = "#D4D0C8"   # light tan for wall fill body
EXTERIOR_W        = 7           # exterior wall thickness (px)
INTERIOR_W        = 3           # interior wall thickness (px)
HATCH_COLOR       = "#9E9E9E"   # exterior wall hatch color
HATCH_SPACING     = 6           # px between hatch lines
GRID_COLOR        = "#EBEBEB"
TEXT_COLOR        = "#1C1C1E"
DIM_COLOR         = "#6B6B6B"
ENTRANCE_COLOR    = "#2E7D32"
WINDOW_FILL       = "#B3D9F7"
DOOR_COLOR        = "#5D4037"


class FloorPlanRenderer:
    """Architectural-style floor plan renderer."""

    def __init__(self, width: int = 900, height: int = 900, padding: int = 70, style: str = "architectural"):
        self.width = width
        self.height = height
        self.padding = padding
        self.style = style

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def render(self, layout: Layout, vastu_score: Optional[int] = None) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), "#FAFAF8")
        draw = ImageDraw.Draw(img)

        scale, ox, oy = self._compute_scale(layout)

        px1 = ox
        py1 = oy
        px2 = ox + layout.plot_width * scale
        py2 = oy + layout.plot_length * scale

        # 1. Subtle grid
        self._draw_grid(draw, layout, scale, ox, oy)

        # 2. Fill entire plot with wall-body color (walls fill remaining space)
        draw.rectangle([px1, py1, px2, py2], fill=WALL_BODY_COLOR, outline=None)

        # 3. Cut rooms out with their fills (draws room interior, no outline yet)
        for room in layout.rooms:
            self._fill_room(draw, room, scale, ox, oy)

        # 4. Draw wall lines over everything
        self._draw_all_walls(draw, layout, scale, ox, oy)

        # 5. Exterior wall hatching
        self._draw_exterior_hatching(draw, px1, py1, px2, py2)

        # 6. Plot boundary (thick exterior outline)
        draw.rectangle([px1, py1, px2, py2], fill=None, outline=WALL_COLOR, width=EXTERIOR_W)

        # 7. Openings
        for door in layout.doors:
            self._draw_door(draw, door, scale, ox, oy)
        for window in layout.windows:
            self._draw_window(draw, window, scale, ox, oy)

        # 8. Entrance
        self._draw_entrance(draw, layout, scale, ox, oy, px1, py1, px2, py2)

        # 9. Room labels
        for room in layout.rooms:
            self._draw_room_label(draw, room, scale, ox, oy)

        # 10. Dimension labels on plot edges
        self._draw_plot_dims(draw, layout, scale, ox, oy, px1, py1, px2, py2)

        # 11. Compass rose
        self._draw_compass(draw, layout.facing_direction)

        # 12. Vastu score badge
        if vastu_score is not None:
            self._draw_score_badge(draw, vastu_score)

        # 13. Title
        self._draw_title(draw, layout)

        return img

    def render_to_base64(self, layout: Layout, vastu_score: Optional[int] = None) -> str:
        img = self.render(layout, vastu_score)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------ #
    # Scale / layout helpers
    # ------------------------------------------------------------------ #

    def _compute_scale(self, layout: Layout):
        avail_w = self.width - 2 * self.padding
        avail_h = self.height - 2 * self.padding
        scale = min(avail_w / layout.plot_width, avail_h / layout.plot_length)
        ox = self.padding + (avail_w - layout.plot_width * scale) / 2
        oy = self.padding + (avail_h - layout.plot_length * scale) / 2
        return scale, ox, oy

    def _room_px(self, room: Room, scale: float, ox: float, oy: float):
        """Return pixel coords (x1,y1,x2,y2) of a room."""
        x1 = ox + room.x * scale
        y1 = oy + room.y * scale
        x2 = x1 + room.width * scale
        y2 = y1 + room.height * scale
        return x1, y1, x2, y2

    # ------------------------------------------------------------------ #
    # Drawing steps
    # ------------------------------------------------------------------ #

    def _draw_grid(self, draw: ImageDraw, layout: Layout, scale: float, ox: float, oy: float):
        step = 5  # feet
        x = 0
        while x <= layout.plot_width:
            px = ox + x * scale
            draw.line([(px, oy), (px, oy + layout.plot_length * scale)], fill=GRID_COLOR, width=1)
            x += step
        y = 0
        while y <= layout.plot_length:
            py = oy + y * scale
            draw.line([(ox, py), (ox + layout.plot_width * scale, py)], fill=GRID_COLOR, width=1)
            y += step

    def _fill_room(self, draw: ImageDraw, room: Room, scale: float, ox: float, oy: float):
        """Fill room interior with its color (no outline — walls are drawn separately)."""
        x1, y1, x2, y2 = self._room_px(room, scale, ox, oy)
        color = ROOM_COLORS.get(room.room_type, "#FFFFFF")
        draw.rectangle([x1, y1, x2, y2], fill=color)

        # Special sky-pattern for OTS
        if room.room_type == "ots":
            self._draw_ots_pattern(draw, x1, y1, x2, y2)

    def _draw_all_walls(self, draw: ImageDraw, layout: Layout, scale: float, ox: float, oy: float):
        """
        Draw interior wall lines between every pair of rooms that share an edge.
        We draw lines at every room boundary with INTERIOR_W thickness.
        """
        rooms = layout.rooms
        n = len(rooms)
        drawn = set()

        for i in range(n):
            x1a, y1a, x2a, y2a = self._room_px(rooms[i], scale, ox, oy)

            for j in range(i + 1, n):
                if (i, j) in drawn:
                    continue
                x1b, y1b, x2b, y2b = self._room_px(rooms[j], scale, ox, oy)

                # Shared vertical wall (right edge of i == left edge of j, or vice versa)
                if abs(x2a - x1b) < 2:
                    # Draw vertical line at x2a
                    y_top = max(y1a, y1b)
                    y_bot = min(y2a, y2b)
                    if y_bot > y_top:
                        draw.line([(x2a, y_top), (x2a, y_bot)], fill=WALL_COLOR, width=INTERIOR_W)
                    drawn.add((i, j))

                elif abs(x2b - x1a) < 2:
                    y_top = max(y1a, y1b)
                    y_bot = min(y2a, y2b)
                    if y_bot > y_top:
                        draw.line([(x1a, y_top), (x1a, y_bot)], fill=WALL_COLOR, width=INTERIOR_W)
                    drawn.add((i, j))

                # Shared horizontal wall (bottom edge of i == top edge of j, or vice versa)
                elif abs(y2a - y1b) < 2:
                    x_left = max(x1a, x1b)
                    x_right = min(x2a, x2b)
                    if x_right > x_left:
                        draw.line([(x_left, y2a), (x_right, y2a)], fill=WALL_COLOR, width=INTERIOR_W)
                    drawn.add((i, j))

                elif abs(y2b - y1a) < 2:
                    x_left = max(x1a, x1b)
                    x_right = min(x2a, x2b)
                    if x_right > x_left:
                        draw.line([(x_left, y1a), (x_right, y1a)], fill=WALL_COLOR, width=INTERIOR_W)
                    drawn.add((i, j))

    def _draw_exterior_hatching(self, draw: ImageDraw, px1, py1, px2, py2):
        """Draw diagonal hatch lines just outside the plot boundary to indicate solid exterior walls."""
        hatch_band = 5  # px width of hatch zone outside the boundary
        clip_expand = hatch_band + 2

        # Top wall
        for x in range(int(px1), int(px2), HATCH_SPACING):
            draw.line([(x, py1 - clip_expand), (x + hatch_band, py1)], fill=HATCH_COLOR, width=1)

        # Bottom wall
        for x in range(int(px1), int(px2), HATCH_SPACING):
            draw.line([(x, py2), (x + hatch_band, py2 + clip_expand)], fill=HATCH_COLOR, width=1)

        # Left wall
        for y in range(int(py1), int(py2), HATCH_SPACING):
            draw.line([(px1 - clip_expand, y), (px1, y + hatch_band)], fill=HATCH_COLOR, width=1)

        # Right wall
        for y in range(int(py1), int(py2), HATCH_SPACING):
            draw.line([(px2, y), (px2 + clip_expand, y + hatch_band)], fill=HATCH_COLOR, width=1)

    def _draw_ots_pattern(self, draw: ImageDraw, x1, y1, x2, y2):
        step = 10
        x = x1
        while x < x2 + (y2 - y1):
            sx = max(x1, x)
            sy = max(y1, y1 + (x1 - x))
            ex = min(x2, x + (y2 - y1))
            ey = min(y2, y1 + (x - x1) + (y2 - y1))
            if sx < x2 and ey > y1:
                draw.line([(sx, sy), (ex, ey)], fill="#93C6E0", width=1)
            x += step

    def _draw_door(self, draw: ImageDraw, door: Door, scale: float, ox: float, oy: float):
        x = ox + door.x * scale
        y = oy + door.y * scale
        dw = door.width * scale
        color = ENTRANCE_COLOR if door.is_main_entrance else DOOR_COLOR
        thick = 6 if door.is_main_entrance else 4

        # Erase door opening in wall (draw room-colored rectangle)
        if door.orientation == "horizontal":
            # White gap in wall
            draw.rectangle([x - dw/2, y - thick/2 - 1, x + dw/2, y + thick/2 + 1], fill="#F5F5F5")
            # Door leaf line + swing arc
            r = dw * 0.85
            draw.arc([x - dw/2, y, x - dw/2 + r*2, y + r*2], start=270, end=360, fill=DOOR_COLOR, width=1)
            draw.line([(x - dw/2, y), (x - dw/2 + r, y + r)], fill=DOOR_COLOR, width=2)
            # Door threshold
            draw.line([(x - dw/2, y), (x + dw/2, y)], fill=color, width=thick)
        else:
            draw.rectangle([x - thick/2 - 1, y - dw/2, x + thick/2 + 1, y + dw/2], fill="#F5F5F5")
            r = dw * 0.85
            draw.arc([x, y - dw/2, x + r*2, y - dw/2 + r*2], start=180, end=270, fill=DOOR_COLOR, width=1)
            draw.line([(x, y - dw/2), (x + r, y - dw/2 + r)], fill=DOOR_COLOR, width=2)
            draw.line([(x, y - dw/2), (x, y + dw/2)], fill=color, width=thick)

    def _draw_window(self, draw: ImageDraw, window: Window, scale: float, ox: float, oy: float):
        """CubiCasa-style window: three parallel lines in a light-blue gap."""
        x = ox + window.x * scale
        y = oy + window.y * scale
        ww = window.width * scale
        wt = 8  # window thickness in px

        if window.orientation == "horizontal":
            draw.rectangle([x - ww/2, y - wt/2, x + ww/2, y + wt/2], fill=WINDOW_FILL)
            for offset in (-wt/4, 0, wt/4):
                draw.line([(x - ww/2, y + offset), (x + ww/2, y + offset)], fill="#2980B9", width=1)
            draw.rectangle([x - ww/2, y - wt/2, x + ww/2, y + wt/2], outline=WALL_COLOR, width=1)
        else:
            draw.rectangle([x - wt/2, y - ww/2, x + wt/2, y + ww/2], fill=WINDOW_FILL)
            for offset in (-wt/4, 0, wt/4):
                draw.line([(x + offset, y - ww/2), (x + offset, y + ww/2)], fill="#2980B9", width=1)
            draw.rectangle([x - wt/2, y - ww/2, x + wt/2, y + ww/2], outline=WALL_COLOR, width=1)

    def _draw_entrance(self, draw: ImageDraw, layout: Layout, scale, ox, oy, px1, py1, px2, py2):
        ex = ox + layout.entrance_position[0] * scale
        ey = oy + layout.entrance_position[1] * scale
        facing = layout.facing_direction
        aw = 14  # arrow half-width

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
        except Exception:
            font = ImageFont.load_default()

        label = "ENTRANCE"

        if facing == "north":
            # Arrow pointing downward from top
            draw.polygon([(ex, ey - 22), (ex - aw, ey - 8), (ex + aw, ey - 8)], fill=ENTRANCE_COLOR)
            draw.line([(ex, ey - 22), (ex, ey - 40)], fill=ENTRANCE_COLOR, width=2)
            bb = draw.textbbox((0, 0), label, font=font)
            draw.text((ex - (bb[2]-bb[0])//2, ey - 52), label, fill=ENTRANCE_COLOR, font=font)
        elif facing == "south":
            draw.polygon([(ex, ey + 22), (ex - aw, ey + 8), (ex + aw, ey + 8)], fill=ENTRANCE_COLOR)
            draw.line([(ex, ey + 22), (ex, ey + 40)], fill=ENTRANCE_COLOR, width=2)
            bb = draw.textbbox((0, 0), label, font=font)
            draw.text((ex - (bb[2]-bb[0])//2, ey + 42), label, fill=ENTRANCE_COLOR, font=font)
        elif facing == "east":
            draw.polygon([(ex + 22, ey), (ex + 8, ey - aw), (ex + 8, ey + aw)], fill=ENTRANCE_COLOR)
            draw.line([(ex + 22, ey), (ex + 40, ey)], fill=ENTRANCE_COLOR, width=2)
            bb = draw.textbbox((0, 0), label, font=font)
            draw.text((ex + 42, ey - (bb[3]-bb[1])//2), label, fill=ENTRANCE_COLOR, font=font)
        else:  # west
            draw.polygon([(ex - 22, ey), (ex - 8, ey - aw), (ex - 8, ey + aw)], fill=ENTRANCE_COLOR)
            draw.line([(ex - 22, ey), (ex - 40, ey)], fill=ENTRANCE_COLOR, width=2)
            bb = draw.textbbox((0, 0), label, font=font)
            draw.text((ex - 42 - (bb[2]-bb[0]), ey - (bb[3]-bb[1])//2), label, fill=ENTRANCE_COLOR, font=font)

    def _draw_room_label(self, draw: ImageDraw, room: Room, scale: float, ox: float, oy: float):
        x1, y1, x2, y2 = self._room_px(room, scale, ox, oy)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        room_px_w = x2 - x1
        room_px_h = y2 - y1

        try:
            name_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
            dim_font  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
        except Exception:
            name_font = ImageFont.load_default()
            dim_font  = name_font

        name = "Open to Sky" if room.room_type == "ots" else room.room_type.replace("_", " ").title()
        dim  = f"{room.width:.0f}' × {room.height:.0f}'"

        nb = draw.textbbox((0, 0), name, font=name_font)
        db = draw.textbbox((0, 0), dim, font=dim_font)
        nw, nh = nb[2] - nb[0], nb[3] - nb[1]
        dw      = db[2] - db[0]

        # Only draw text if room is large enough
        if nw < room_px_w - 4 and (nh + 12) < room_px_h:
            draw.text((cx - nw / 2, cy - nh - 2), name, fill=TEXT_COLOR, font=name_font)
        if dw < room_px_w - 4 and 20 < room_px_h:
            draw.text((cx - dw / 2, cy + 3), dim, fill=DIM_COLOR, font=dim_font)

    def _draw_plot_dims(self, draw: ImageDraw, layout: Layout, scale, ox, oy, px1, py1, px2, py2):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except Exception:
            font = ImageFont.load_default()

        # Width at bottom
        wt = f"{layout.plot_width:.0f} ft"
        bb = draw.textbbox((0, 0), wt, font=font)
        draw.text(((px1 + px2) / 2 - (bb[2]-bb[0]) / 2, py2 + 12), wt, fill=TEXT_COLOR, font=font)

        # Length on right
        lt = f"{layout.plot_length:.0f} ft"
        bb = draw.textbbox((0, 0), lt, font=font)
        draw.text((px2 + 12, (py1 + py2) / 2 - (bb[3]-bb[1]) / 2), lt, fill=TEXT_COLOR, font=font)

    def _draw_compass(self, draw: ImageDraw, facing: str):
        cx, cy, r = 48, 48, 22
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=TEXT_COLOR, width=2, fill="#FAFAF8")

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except Exception:
            font = ImageFont.load_default()

        dirs = [("N", cx, cy - r - 14, "#C0392B"), ("S", cx, cy + r + 4, TEXT_COLOR),
                ("E", cx + r + 5, cy - 6, TEXT_COLOR), ("W", cx - r - 14, cy - 6, TEXT_COLOR)]
        for d, dx, dy, col in dirs:
            bb = draw.textbbox((0, 0), d, font=font)
            draw.text((dx - (bb[2]-bb[0])//2, dy), d, fill=col, font=font)

        # Cardinal tick marks
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            ix = cx + (r - 5) * math.sin(rad)
            iy = cy - (r - 5) * math.cos(rad)
            ox2 = cx + r * math.sin(rad)
            oy2 = cy - r * math.cos(rad)
            draw.line([(ix, iy), (ox2, oy2)], fill=TEXT_COLOR, width=1)

    def _draw_score_badge(self, draw: ImageDraw, score: int):
        bx, by, br = self.width - 50, 50, 30
        color = "#27AE60" if score >= 80 else ("#F39C12" if score >= 60 else "#E74C3C")
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=color, outline="#FFFFFF", width=3)

        try:
            sf = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            lf = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
        except Exception:
            sf = lf = ImageFont.load_default()

        st = str(score)
        bb = draw.textbbox((0, 0), st, font=sf)
        draw.text((bx - (bb[2]-bb[0])//2, by - 14), st, fill="#FFFFFF", font=sf)
        draw.text((bx - 13, by + 8), "Vastu", fill="#FFFFFF", font=lf)

    def _draw_title(self, draw: ImageDraw, layout: Layout):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        except Exception:
            font = ImageFont.load_default()

        title = f"Floor Plan  ·  {layout.facing_direction.title()} Facing  ·  {layout.plot_width:.0f}' × {layout.plot_length:.0f}'"
        bb = draw.textbbox((0, 0), title, font=font)
        draw.text(((self.width - (bb[2]-bb[0])) / 2, self.height - 28), title, fill=TEXT_COLOR, font=font)
