"""
Deterministic, rule-based architectural floor plan renderer.
No GAN/ML — uses only PIL for crisp 2048×2048 output.
"""

import base64
import io
import math
from PIL import Image, ImageDraw, ImageFont

from app.models.schemas import Layout

# ── Palette (matches FloorPlanViewer.tsx) ─────────────────────────────────────
ROOM_COLORS: dict[str, tuple] = {
    "master_bedroom": (255, 200, 100),
    "bedroom":        (255, 200, 100),
    "living_room":    ( 80, 220, 120),
    "kitchen":        (255, 255,  80),
    "bathroom":       (100, 180, 255),
    "dining":         (160, 220, 160),
    "puja_room":      (220, 100, 220),
    "parking":        (180, 180, 180),
    "balcony":        (160, 220, 240),
    "staircase":      (255, 140, 140),
    "store":          (180, 150, 110),
    "ots":            (200, 255, 200),
    "corridor":       (240, 220, 192),
    "utility":        (200, 190, 160),
    "entrance":       (255, 240, 200),
}

ROOM_LABELS: dict[str, str] = {
    "master_bedroom": "Master Bedroom",
    "bedroom":        "Bedroom",
    "living_room":    "Living Room",
    "kitchen":        "Kitchen",
    "bathroom":       "Bathroom",
    "dining":         "Dining",
    "puja_room":      "Puja Room",
    "parking":        "Parking",
    "balcony":        "Balcony",
    "staircase":      "Staircase",
    "store":          "Store",
    "ots":            "Open to Sky",
    "corridor":       "Corridor",
    "utility":        "Utility",
    "entrance":       "Entrance",
}

# ── Drawing palette ────────────────────────────────────────────────────────────
_BG          = (248, 245, 240)   # warm off-white canvas
_WALL        = ( 28,  28,  28)   # near-black walls
_DOOR_GAP    = (248, 245, 240)   # gap matches background
_DOOR_ARC    = ( 28,  28,  28)   # interior door arc/leaf
_ENT_ARC     = (200, 120,   0)   # amber — main entrance arc/leaf
_PKG_ARC     = ( 70,  70,  70)   # charcoal — parking vehicle entrance
_WIN_FILL    = (160, 215, 255)   # window fill
_WIN_LINE    = ( 80, 160, 220)   # window frame line
_LABEL       = ( 20,  20,  20)   # room name text
_DIM_TEXT    = ( 70,  70,  70)   # dimension annotation
_COMPASS_FG  = ( 35,  35,  35)   # compass letters / ring
_NORTH_FILL  = ( 35,  35,  35)   # north arrow (filled dark)
_SOUTH_FILL  = (200, 200, 200)   # south arrow (light)

# Font search paths (macOS → Linux fallbacks)
_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


class FloorPlanRenderer:
    """
    Renders a Layout object → crisp 2048×2048 architectural floor plan PNG.

    Usage:
        b64 = FloorPlanRenderer().render(layout)
    """

    TARGET_SIZE = 2048
    MARGIN      = 200   # pixels around the floor plan (dimension labels, legend, compass)
    WALL_THICK  = 6     # exterior boundary wall thickness (px)
    INT_WALL    = 4     # interior wall thickness (px)

    def render(self, layout: Layout) -> str:
        """Return base64-encoded PNG string (no data-URI prefix)."""
        img = self._draw(layout)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Main draw method ───────────────────────────────────────────────────────

    def _draw(self, layout: Layout) -> Image.Image:
        S  = self.TARGET_SIZE
        M  = self.MARGIN
        pw = float(layout.plot_width)
        pl = float(layout.plot_length)

        # Pixels-per-foot scale — uniform, fits within the drawable area
        avail = S - 2 * M
        scale = min(avail / pw, avail / pl)

        # Centre the floor plan in the drawable area
        draw_w = pw * scale
        draw_h = pl * scale
        ox = M + (avail - draw_w) / 2   # left edge of plot (pixels)
        oy = M + (avail - draw_h) / 2   # top edge of plot (pixels)

        def to_px(xf: float, yf: float) -> tuple[float, float]:
            return (ox + xf * scale, oy + yf * scale)

        img  = Image.new("RGB", (S, S), _BG)
        draw = ImageDraw.Draw(img)

        # Draw in painter's order (back to front)
        self._fill_rooms(draw, layout, scale, ox, oy)
        self._draw_windows(draw, layout, scale, ox, oy)
        self._draw_interior_walls(draw, layout, scale, ox, oy)
        self._draw_plot_boundary(draw, pw, pl, scale, ox, oy)
        self._draw_doors(draw, layout, scale, ox, oy)
        self._draw_labels(draw, layout, scale, ox, oy)
        self._draw_dimensions(draw, pw, pl, scale, ox, oy, S)
        self._draw_compass(draw, layout.facing_direction, S, M)
        self._draw_legend(draw, layout, S, M)

        return img

    # ── 1. Room fills ──────────────────────────────────────────────────────────

    def _fill_rooms(self, draw: ImageDraw.ImageDraw, layout: Layout,
                    scale: float, ox: float, oy: float) -> None:
        for room in layout.rooms:
            color = ROOM_COLORS.get(room.room_type, (210, 210, 210))
            x0 = ox + room.x * scale
            y0 = oy + room.y * scale
            x1 = x0 + room.width  * scale
            y1 = y0 + room.height * scale
            draw.rectangle([x0, y0, x1, y1], fill=color)

    # ── 2. Windows ────────────────────────────────────────────────────────────

    def _draw_windows(self, draw: ImageDraw.ImageDraw, layout: Layout,
                      scale: float, ox: float, oy: float) -> None:
        thick = max(5, int(scale * 0.6))   # ~0.6 ft thick symbol
        frame = max(2, thick // 3)

        for win in layout.windows:
            wx = ox + win.x * scale
            wy = oy + win.y * scale
            hw = win.width * scale / 2

            if win.orientation == "horizontal":
                # Window runs left-right on N or S wall
                draw.line([(wx - hw, wy), (wx + hw, wy)], fill=_WIN_FILL, width=thick)
                draw.line([(wx - hw, wy), (wx + hw, wy)], fill=_WIN_LINE, width=frame)
                draw.line([(wx - hw, wy - frame), (wx + hw, wy - frame)],
                          fill=_WIN_LINE, width=1)
                draw.line([(wx - hw, wy + frame), (wx + hw, wy + frame)],
                          fill=_WIN_LINE, width=1)
            else:
                # Window runs top-bottom on E or W wall
                draw.line([(wx, wy - hw), (wx, wy + hw)], fill=_WIN_FILL, width=thick)
                draw.line([(wx, wy - hw), (wx, wy + hw)], fill=_WIN_LINE, width=frame)
                draw.line([(wx - frame, wy - hw), (wx - frame, wy + hw)],
                          fill=_WIN_LINE, width=1)
                draw.line([(wx + frame, wy - hw), (wx + frame, wy + hw)],
                          fill=_WIN_LINE, width=1)

    # ── 3. Interior walls (room outlines) ─────────────────────────────────────

    def _draw_interior_walls(self, draw: ImageDraw.ImageDraw, layout: Layout,
                              scale: float, ox: float, oy: float) -> None:
        for room in layout.rooms:
            x0 = ox + room.x * scale
            y0 = oy + room.y * scale
            x1 = x0 + room.width  * scale
            y1 = y0 + room.height * scale
            draw.rectangle([x0, y0, x1, y1],
                           outline=_WALL, width=self.INT_WALL)

    # ── 4. Plot boundary ──────────────────────────────────────────────────────

    def _draw_plot_boundary(self, draw: ImageDraw.ImageDraw,
                             pw: float, pl: float,
                             scale: float, ox: float, oy: float) -> None:
        x0, y0 = ox, oy
        x1 = ox + pw * scale
        y1 = oy + pl * scale
        draw.rectangle([x0, y0, x1, y1],
                       outline=_WALL, width=self.WALL_THICK)

    # ── 5. Entrance opening ────────────────────────────────────────────────────

    def _draw_doors(self, draw: ImageDraw.ImageDraw, layout: Layout,
                    scale: float, ox: float, oy: float) -> None:
        for door in layout.doors:
            if door.is_main_entrance:
                self._draw_entrance_opening(draw, door, scale, ox, oy)
            elif door.id == "parking_entrance":
                self._draw_parking_entrance(draw, door, scale, ox, oy)

    def _draw_entrance_opening(self, draw, door, scale: float,
                                ox: float, oy: float) -> None:
        """
        Draw the main entrance as a clean architectural opening:
        - Gap (erased segment) in the boundary wall
        - Two short jamb lines at each edge of the gap
        - Bold amber fill in the opening
        - Directional arrow pointing into the plot
        - 'ENTRANCE' label
        """
        gap_px   = door.width * scale
        half     = gap_px / 2
        jamb     = max(6, int(scale * 0.5))    # jamb stub length in px
        thick    = self.WALL_THICK
        font     = _load_font(max(12, int(scale * 0.7)))

        px = ox + door.x * scale
        py = oy + door.y * scale

        if door.orientation == "horizontal":
            # ── Erase the gap in the boundary wall ───────────────────────
            draw.line([(px - half, py), (px + half, py)],
                      fill=_DOOR_GAP, width=thick + 6)

            # ── Amber fill strip across the opening ──────────────────────
            draw.line([(px - half, py), (px + half, py)],
                      fill=_ENT_ARC, width=max(3, thick - 2))

            # ── Jambs: short perpendicular stubs at each edge ─────────────
            draw.line([(px - half, py - jamb), (px - half, py + jamb)],
                      fill=_WALL, width=thick)
            draw.line([(px + half, py - jamb), (px + half, py + jamb)],
                      fill=_WALL, width=thick)

            # ── Arrow: pointing inward (downward for north-facing) ─────────
            aw    = max(6, int(gap_px * 0.18))
            ay    = py + max(8, int(scale * 0.8))   # inward side
            tip_y = ay + aw * 2
            draw.polygon([(px, tip_y), (px - aw, ay), (px + aw, ay)],
                         fill=_ENT_ARC)

            # ── Label ─────────────────────────────────────────────────────
            lbl = "ENTRANCE"
            bb  = draw.textbbox((0, 0), lbl, font=font)
            lw  = bb[2] - bb[0]
            draw.text((px - lw / 2, tip_y + 4), lbl, fill=_ENT_ARC, font=font)

        else:  # vertical (east / west facing)
            draw.line([(px, py - half), (px, py + half)],
                      fill=_DOOR_GAP, width=thick + 6)
            draw.line([(px, py - half), (px, py + half)],
                      fill=_ENT_ARC, width=max(3, thick - 2))

            draw.line([(px - jamb, py - half), (px + jamb, py - half)],
                      fill=_WALL, width=thick)
            draw.line([(px - jamb, py + half), (px + jamb, py + half)],
                      fill=_WALL, width=thick)

            # Arrow pointing inward (rightward for west-facing, leftward for east)
            aw    = max(6, int(gap_px * 0.18))
            inward = 1 if door.x <= 1 else -1   # west→right, east→left
            ax     = px + inward * max(8, int(scale * 0.8))
            tip_x  = ax + inward * aw * 2
            draw.polygon([(tip_x, py), (ax, py - aw), (ax, py + aw)],
                         fill=_ENT_ARC)

            lbl = "ENTRANCE"
            bb  = draw.textbbox((0, 0), lbl, font=font)
            lw  = bb[2] - bb[0]
            lh  = bb[3] - bb[1]
            draw.text((tip_x + inward * 4, py - lh / 2), lbl,
                      fill=_ENT_ARC, font=font)

    def _draw_parking_entrance(self, draw, door, scale: float,
                               ox: float, oy: float) -> None:
        """
        Draw the parking vehicle entrance on the plot boundary:
        - Wider gap than the pedestrian entrance (9 ft vs 3 ft).
        - Charcoal fill strip and jamb stubs.
        - Inward arrow and 'PARKING' label.
        Mirrors _draw_entrance_opening() but uses _PKG_ARC color.
        """
        gap_px = door.width * scale
        half   = gap_px / 2
        jamb   = max(6, int(scale * 0.5))
        thick  = self.WALL_THICK
        font   = _load_font(max(12, int(scale * 0.7)))

        px = ox + door.x * scale
        py = oy + door.y * scale

        if door.orientation == "horizontal":
            # Erase gap in boundary wall
            draw.line([(px - half, py), (px + half, py)],
                      fill=_DOOR_GAP, width=thick + 6)
            # Charcoal fill strip
            draw.line([(px - half, py), (px + half, py)],
                      fill=_PKG_ARC, width=max(3, thick - 2))
            # Jamb stubs
            draw.line([(px - half, py - jamb), (px - half, py + jamb)],
                      fill=_WALL, width=thick)
            draw.line([(px + half, py - jamb), (px + half, py + jamb)],
                      fill=_WALL, width=thick)
            # Inward arrow
            aw    = max(6, int(gap_px * 0.12))
            ay    = py + max(8, int(scale * 0.8))
            tip_y = ay + aw * 2
            draw.polygon([(px, tip_y), (px - aw, ay), (px + aw, ay)],
                         fill=_PKG_ARC)
            # Label
            lbl = "PARKING"
            bb  = draw.textbbox((0, 0), lbl, font=font)
            lw  = bb[2] - bb[0]
            draw.text((px - lw / 2, tip_y + 4), lbl, fill=_PKG_ARC, font=font)

        else:  # vertical (east / west wall)
            draw.line([(px, py - half), (px, py + half)],
                      fill=_DOOR_GAP, width=thick + 6)
            draw.line([(px, py - half), (px, py + half)],
                      fill=_PKG_ARC, width=max(3, thick - 2))
            draw.line([(px - jamb, py - half), (px + jamb, py - half)],
                      fill=_WALL, width=thick)
            draw.line([(px - jamb, py + half), (px + jamb, py + half)],
                      fill=_WALL, width=thick)
            aw     = max(6, int(gap_px * 0.12))
            inward = 1 if door.x <= 1 else -1
            ax     = px + inward * max(8, int(scale * 0.8))
            tip_x  = ax + inward * aw * 2
            draw.polygon([(tip_x, py), (ax, py - aw), (ax, py + aw)],
                         fill=_PKG_ARC)
            lbl = "PARKING"
            bb  = draw.textbbox((0, 0), lbl, font=font)
            lw  = bb[2] - bb[0]
            lh  = bb[3] - bb[1]
            draw.text((tip_x + inward * 4, py - lh / 2), lbl,
                      fill=_PKG_ARC, font=font)

    # ── 6. Room labels ────────────────────────────────────────────────────────

    def _draw_labels(self, draw: ImageDraw.ImageDraw, layout: Layout,
                     scale: float, ox: float, oy: float) -> None:
        for room in layout.rooms:
            room_px_w = room.width  * scale
            room_px_h = room.height * scale
            cx = ox + (room.x + room.width  / 2) * scale
            cy = oy + (room.y + room.height / 2) * scale

            # Choose font size proportional to the smaller room dimension
            min_dim_px = min(room_px_w, room_px_h)
            fs_name = max(10, min(24, int(min_dim_px * 0.13)))
            fs_dim  = max(9,  min(18, int(min_dim_px * 0.10)))
            font_name = _load_font(fs_name)
            font_dim  = _load_font(fs_dim)

            name  = ROOM_LABELS.get(room.room_type,
                                     room.room_type.replace("_", " ").title())
            dim_s = f"{room.width:.0f}' × {room.height:.0f}'"

            nb = draw.textbbox((0, 0), name, font=font_name)
            nw = nb[2] - nb[0]
            nh = nb[3] - nb[1]

            db = draw.textbbox((0, 0), dim_s, font=font_dim)
            dw = db[2] - db[0]
            dh = db[3] - db[1]

            total_h = nh + 3 + dh

            if nw < room_px_w * 0.88 and total_h < room_px_h * 0.60:
                name_y = cy - total_h / 2
                draw.text((cx - nw / 2, name_y), name,
                          fill=_LABEL, font=font_name)
                draw.text((cx - dw / 2, name_y + nh + 3), dim_s,
                          fill=_DIM_TEXT, font=font_dim)
            elif nw < room_px_w * 0.88 and nh < room_px_h * 0.55:
                # Only room name, not dimensions
                draw.text((cx - nw / 2, cy - nh / 2), name,
                          fill=_LABEL, font=font_name)

    # ── 7. Dimension annotations ──────────────────────────────────────────────

    def _draw_dimensions(self, draw: ImageDraw.ImageDraw,
                          pw: float, pl: float,
                          scale: float, ox: float, oy: float, S: int) -> None:
        fs   = max(14, int(scale * 1.0))
        font = _load_font(fs)
        tick = 14    # tick mark length
        gap  = 10    # gap between plot edge and dimension line

        # ── Bottom: plot width ──
        y_dim = oy + pl * scale + gap
        mid_x = ox + pw * scale / 2
        label = f"{pw:.0f}'"
        bb    = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]

        # Horizontal dimension line
        draw.line([(ox, y_dim + tick // 2), (ox + pw * scale, y_dim + tick // 2)],
                  fill=_DIM_TEXT, width=1)
        draw.line([(ox, y_dim), (ox, y_dim + tick)], fill=_DIM_TEXT, width=2)
        draw.line([(ox + pw * scale, y_dim), (ox + pw * scale, y_dim + tick)],
                  fill=_DIM_TEXT, width=2)
        draw.text((mid_x - tw / 2, y_dim + tick + 4), label,
                  fill=_DIM_TEXT, font=font)

        # ── Right: plot length ──
        x_dim = ox + pw * scale + gap
        mid_y = oy + pl * scale / 2
        label2 = f"{pl:.0f}'"
        bb2    = draw.textbbox((0, 0), label2, font=font)
        tw2, th2 = bb2[2] - bb2[0], bb2[3] - bb2[1]

        draw.line([(x_dim + tick // 2, oy), (x_dim + tick // 2, oy + pl * scale)],
                  fill=_DIM_TEXT, width=1)
        draw.line([(x_dim, oy), (x_dim + tick, oy)], fill=_DIM_TEXT, width=2)
        draw.line([(x_dim, oy + pl * scale), (x_dim + tick, oy + pl * scale)],
                  fill=_DIM_TEXT, width=2)
        draw.text((x_dim + tick + 4, mid_y - th2 / 2), label2,
                  fill=_DIM_TEXT, font=font)

    # ── 8. Compass rose ───────────────────────────────────────────────────────

    def _draw_compass(self, draw: ImageDraw.ImageDraw,
                       facing: str, S: int, M: int) -> None:
        # Fit compass entirely within the top-right M×M corner
        r  = M // 5          # compass radius — sized so labels stay within margin
        cx = S - M + M // 2  # horizontal centre of the margin strip
        cy = M // 2          # vertical centre of the margin strip
        fs = max(11, int(r * 0.50))
        font = _load_font(fs)

        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=_COMPASS_FG, width=2)

        # N arrow (filled dark) pointing up
        aw = max(5, r // 4)
        draw.polygon([(cx, cy - r + 4), (cx - aw, cy), (cx + aw, cy)],
                     fill=_NORTH_FILL)
        # S arrow (light) pointing down
        draw.polygon([(cx, cy + r - 4), (cx - aw, cy), (cx + aw, cy)],
                     fill=_SOUTH_FILL, outline=_COMPASS_FG)

        # Cardinal labels
        off = r + fs + 4
        cardinals = [("N", 0, -off), ("S", 0, off), ("E", off, 0), ("W", -off, 0)]
        for label, dx, dy in cardinals:
            is_facing = label.lower() == facing.lower()
            color = _ENT_ARC if is_facing else _COMPASS_FG
            bb = draw.textbbox((0, 0), label, font=font)
            lw = bb[2] - bb[0]
            lh = bb[3] - bb[1]
            draw.text((cx + dx - lw / 2, cy + dy - lh / 2),
                      label, fill=color, font=font)

    # ── 9. Legend ─────────────────────────────────────────────────────────────

    def _draw_legend(self, draw: ImageDraw.ImageDraw,
                      layout: Layout, S: int, M: int) -> None:
        room_types = list(dict.fromkeys(r.room_type for r in layout.rooms))
        if not room_types:
            return

        swatch = max(16, M // 10)
        pad_v  = max(6,  M // 28)
        pad_h  = 10
        fs     = max(11, swatch - 2)
        font   = _load_font(fs)

        # Measure widest label to decide column width
        col_w = 160
        for rt in room_types:
            label = ROOM_LABELS.get(rt, rt.replace("_", " ").title())
            bb = draw.textbbox((0, 0), label, font=font)
            col_w = max(col_w, bb[2] - bb[0] + swatch + pad_h + 20)

        avail_w = S - 2 * M
        n_cols  = max(1, avail_w // col_w)
        n_rows  = math.ceil(len(room_types) / n_cols)

        legend_h = n_rows * (swatch + pad_v)
        start_x  = M
        start_y  = S - M + (M - legend_h) // 2

        for i, rt in enumerate(room_types):
            col = i % n_cols
            row = i // n_cols
            x = start_x + col * col_w
            y = start_y + row * (swatch + pad_v)

            color = ROOM_COLORS.get(rt, (210, 210, 210))
            draw.rectangle([x, y, x + swatch, y + swatch],
                           fill=color, outline=_WALL, width=1)

            label = ROOM_LABELS.get(rt, rt.replace("_", " ").title())
            bb    = draw.textbbox((0, 0), label, font=font)
            lh    = bb[3] - bb[1]
            draw.text((x + swatch + 6, y + (swatch - lh) // 2),
                      label, fill=_LABEL, font=font)
