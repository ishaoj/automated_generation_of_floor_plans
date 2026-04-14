"""
CubiCasa5k Dataset Loader

Loads and parses the CubiCasa5k dataset of real floor plans.
Dataset contains ~5000 floor plans from Finnish apartments with detailed annotations.

Actual dataset structure (from Zenodo download):
  data_dir/
    train.txt          (list of folder paths for training)
    val.txt
    test.txt
    high_quality/
      1/
        model.svg      (floor plan SVG with room annotations)
        F1_scaled.png  (rendered image)
      2/
        ...

SVG format:
  Rooms are encoded as <g class="Space RoomType"> groups
  containing a <polygon points="y1,x1 y2,x2 ..."/> child
  (Note: CubiCasa SVGs store points as y,x not x,y)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.dom import minidom

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


# Mapping from CubiCasa5k room types (CamelCase) to our system's room types
CUBICASA_TO_ROOM_TYPE = {
    "Bedroom": "bedroom",
    "LivingRoom": "living_room",
    "Lounge": "living_room",
    "Kitchen": "kitchen",
    "Bath": "bathroom",
    "Sauna": "bathroom",
    "Dining": "dining",
    "EatingArea": "dining",
    "HallWay": "corridor",
    "DraughtLobby": "corridor",
    "Entry": "corridor",
    "Storage": "storage",
    "Closet": "storage",
    "DressingRoom": "storage",
    "Garage": "parking",
    "CarPort": "parking",
    "Utility": "utility",
    "ServiceRoom": "utility",
    "Outdoor": "balcony",

    # Skipped types (return None to exclude)
    "Wall": None,
    "Railing": None,
    "Stairs": None,
    "StairWell": None,
    "Undefined": None,
    "UserDefined": None,
    "Background": None,
    "Hall": None,
    "OpenToBelow": None,
}

# Scale factor: CubiCasa SVGs use pixel coordinates.
# We convert to approximate feet using a typical scale (1 pixel ≈ 0.033 feet at 300 DPI).
# For training we just normalize to [0, 1], so exact conversion doesn't matter.
PIXEL_TO_FEET = 0.033


class CubiCasaDataset(Dataset):
    """
    PyTorch Dataset for CubiCasa5k floor plans.

    Returns dicts with:
        'rooms':     List of room dicts (type, polygon, area, centroid, bbox)
        'walls':     List of wall segment dicts
        'doors':     List of door position dicts
        'windows':   List of window position dicts
        'plot_dims': (width, height) in feet
        'svg_path':  Path to source SVG
        'image':     Optional rendered floor plan (numpy H×W×3 uint8)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        include_images: bool = False,
        filter_non_residential: bool = True,
        min_rooms: int = 2,
        max_rooms: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.include_images = include_images
        self.filter_non_residential = filter_non_residential
        self.min_rooms = min_rooms
        self.max_rooms = max_rooms

        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} floor plans for '{split}' split")

    # ------------------------------------------------------------------
    # Sample discovery
    # ------------------------------------------------------------------

    def _load_samples(self) -> List[Dict]:
        svg_paths = self._discover_svg_paths()
        if not svg_paths:
            raise FileNotFoundError(
                f"No floor plan SVG files found in {self.data_dir}. "
                "Please download and extract the CubiCasa5k dataset from "
                "https://zenodo.org/record/2613548"
            )

        samples = []
        for svg_path in svg_paths:
            try:
                sample = self._parse_svg(svg_path)
                if sample is None:
                    continue
                n_rooms = len(sample["rooms"])
                if n_rooms < self.min_rooms or n_rooms > self.max_rooms:
                    continue
                if self.filter_non_residential and self._is_commercial(sample):
                    continue
                samples.append(sample)
            except Exception as e:
                print(f"  Warning: could not parse {svg_path}: {e}")

        return samples

    def _discover_svg_paths(self) -> List[Path]:
        """
        Find model.svg files.

        Priority order:
        1. train.txt / val.txt / test.txt split files (official CubiCasa5k structure)
        2. Recursive glob for model.svg (auto-split by 80/10/10)
        3. Legacy flat svg/ directory (old assumption, kept as fallback)
        """
        # --- Strategy 1: official txt split files ---
        txt_file = self.data_dir / f"{self.split}.txt"
        if txt_file.exists():
            return self._paths_from_txt(txt_file)

        # --- Strategy 2: recursive glob for model.svg ---
        all_svgs = sorted(self.data_dir.rglob("model.svg"))
        if all_svgs:
            return self._split_paths(all_svgs)

        # --- Strategy 3: legacy flat svg/ directory ---
        svg_dir = self.data_dir / "svg"
        if svg_dir.exists():
            all_svgs = sorted(svg_dir.glob("*.svg"))
            return self._split_paths(all_svgs)

        return []

    def _paths_from_txt(self, txt_file: Path) -> List[Path]:
        """Load SVG paths listed in a CubiCasa5k split txt file."""
        paths = []
        with open(txt_file) as f:
            for line in f:
                folder = line.strip().lstrip("/")  # strip leading slash
                if not folder:
                    continue
                svg_path = self.data_dir / folder / "model.svg"
                if svg_path.exists():
                    paths.append(svg_path)
                else:
                    print(f"  Warning: missing {svg_path}")
        return paths

    def _split_paths(self, all_svgs: List[Path]) -> List[Path]:
        """80/10/10 train/val/test split on a sorted list of paths."""
        n = len(all_svgs)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if self.split == "train":
            return all_svgs[:n_train]
        elif self.split == "val":
            return all_svgs[n_train: n_train + n_val]
        else:
            return all_svgs[n_train + n_val:]

    # ------------------------------------------------------------------
    # SVG parsing
    # ------------------------------------------------------------------

    def _parse_svg(self, svg_path: Path) -> Optional[Dict]:
        """
        Parse a CubiCasa5k model.svg file.

        SVG structure:
          <svg viewBox="0 0 W H">
            <g class="Space Bedroom">
              <polygon points="y1,x1 y2,x2 ..."/>
              ...
            </g>
            ...
          </svg>

        Note: CubiCasa SVG point format is (y, x), not (x, y).
        """
        try:
            doc = minidom.parse(str(svg_path))
        except Exception as e:
            print(f"  Warning: XML parse error in {svg_path}: {e}")
            return None

        svg_el = doc.getElementsByTagName("svg")[0]

        # Get canvas dimensions from viewBox or width/height attributes
        viewbox = svg_el.getAttribute("viewBox")
        if viewbox:
            parts = viewbox.split()
            width_px = float(parts[2])
            height_px = float(parts[3])
        else:
            width_px = float(svg_el.getAttribute("width") or 1000)
            height_px = float(svg_el.getAttribute("height") or 1000)

        rooms: List[Dict] = []
        walls: List[Dict] = []
        doors: List[Dict] = []
        windows: List[Dict] = []

        for g in doc.getElementsByTagName("g"):
            cls = g.getAttribute("class")

            if cls.startswith("Space "):
                # class is e.g. "Space Bedroom" or "Space Closet WalkIn"
                # room type is always the second word
                parts = cls.split()
                room_type_raw = parts[1] if len(parts) > 1 else ""
                room_type = CUBICASA_TO_ROOM_TYPE.get(room_type_raw)
                if room_type is None:
                    continue
                room_data = self._extract_room(g, room_type, width_px, height_px)
                if room_data:
                    rooms.append(room_data)

            elif g.getAttribute("id") == "Wall":
                for line in g.getElementsByTagName("line"):
                    wall = self._parse_wall(line, width_px, height_px)
                    if wall:
                        walls.append(wall)

            elif g.getAttribute("id") == "Door":
                for poly in g.getElementsByTagName("polygon"):
                    door = self._parse_opening_polygon(poly, width_px, height_px, "door")
                    if door:
                        doors.append(door)

            elif g.getAttribute("id") == "Window":
                for poly in g.getElementsByTagName("polygon"):
                    win = self._parse_opening_polygon(poly, width_px, height_px, "window")
                    if win:
                        windows.append(win)

        if not rooms:
            return None

        plot_width_ft = width_px * PIXEL_TO_FEET
        plot_height_ft = height_px * PIXEL_TO_FEET

        return {
            "rooms": rooms,
            "walls": walls,
            "doors": doors,
            "windows": windows,
            "plot_dims": (plot_width_ft, plot_height_ft),
            "svg_path": str(svg_path),
        }

    def _extract_room(
        self, g_el, room_type: str, width_px: float, height_px: float
    ) -> Optional[Dict]:
        """Extract room bounding box from a Space group element."""
        # Find the direct polygon child (points are in y,x order)
        poly = None
        for child in g_el.childNodes:
            if child.nodeName == "polygon":
                poly = child
                break
        if poly is None:
            return None

        points_str = poly.getAttribute("points").strip()
        if not points_str:
            return None

        xs, ys = [], []
        for token in points_str.split():
            token = token.strip()
            if not token:
                continue
            try:
                if "," in token:
                    # SVG points are standard "x,y"
                    x_str, y_str = token.split(",", 1)
                    xs.append(float(x_str))
                    ys.append(float(y_str))
                else:
                    # Space-separated pairs handled externally; skip odd tokens
                    pass
            except ValueError:
                continue

        if len(xs) < 3:
            return None

        polygon_pts = list(zip(xs, ys))  # (x, y) pairs
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        area_px = self._polygon_area(polygon_pts)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)  # (x, y, w, h)

        # Normalise to [0, 1]
        bbox_norm = (
            x_min / width_px,
            y_min / height_px,
            (x_max - x_min) / width_px,
            (y_max - y_min) / height_px,
        )

        area_ft = area_px * (PIXEL_TO_FEET ** 2)

        # Convert bbox to feet (same units as plot_dims)
        bbox_ft = (
            x_min * PIXEL_TO_FEET,
            y_min * PIXEL_TO_FEET,
            (x_max - x_min) * PIXEL_TO_FEET,
            (y_max - y_min) * PIXEL_TO_FEET,
        )
        centroid_ft = (centroid[0] * PIXEL_TO_FEET, centroid[1] * PIXEL_TO_FEET)

        return {
            "type": room_type,
            "polygon": polygon_pts,
            "centroid": centroid_ft,
            "area": area_ft,
            "bbox": bbox_ft,
            "bbox_norm": bbox_norm,
        }

    def _parse_wall(self, line_el, width_px: float, height_px: float) -> Optional[Dict]:
        try:
            x1 = float(line_el.getAttribute("x1") or 0) * PIXEL_TO_FEET
            y1 = float(line_el.getAttribute("y1") or 0) * PIXEL_TO_FEET
            x2 = float(line_el.getAttribute("x2") or 0) * PIXEL_TO_FEET
            y2 = float(line_el.getAttribute("y2") or 0) * PIXEL_TO_FEET
            return {"start": (x1, y1), "end": (x2, y2)}
        except Exception:
            return None

    def _parse_opening_polygon(
        self, poly_el, width_px: float, height_px: float, opening_type: str
    ) -> Optional[Dict]:
        try:
            pts_str = poly_el.getAttribute("points").strip()
            if not pts_str:
                return None
            xs, ys = [], []
            for token in pts_str.split():
                if "," in token:
                    x_s, y_s = token.split(",", 1)
                    xs.append(float(x_s))
                    ys.append(float(y_s))
            if not xs:
                return None
            cx = (sum(xs) / len(xs)) * PIXEL_TO_FEET
            cy = (sum(ys) / len(ys)) * PIXEL_TO_FEET
            return {"type": opening_type, "position": (cx, cy)}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _polygon_area(points: List[Tuple[float, float]]) -> float:
        """Shoelace formula for polygon area."""
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0

    # ------------------------------------------------------------------
    # Residential filter
    # ------------------------------------------------------------------

    def _is_commercial(self, sample: Dict) -> bool:
        types = {r["type"] for r in sample["rooms"]}
        return not ("bedroom" in types or "bathroom" in types)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx].copy()
        if self.include_images:
            sample["image"] = self._render_floor_plan(sample)
        return sample

    # ------------------------------------------------------------------
    # Optional visualisation
    # ------------------------------------------------------------------

    def _render_floor_plan(self, sample: Dict, size: int = 512) -> np.ndarray:
        plot_w, plot_h = sample["plot_dims"]
        scale = min(size / plot_w, size / plot_h)
        img_w = max(1, int(plot_w * scale))
        img_h = max(1, int(plot_h * scale))

        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)

        room_colors = {
            "bedroom": "#FFE4E1",
            "living_room": "#E0F0FF",
            "kitchen": "#FFE4B5",
            "bathroom": "#E6E6FA",
            "dining": "#F0FFF0",
            "storage": "#F5F5DC",
            "utility": "#F5F5DC",
            "balcony": "#E0FFFF",
            "parking": "#DCDCDC",
            "corridor": "#FFF8DC",
        }

        for room in sample["rooms"]:
            color = room_colors.get(room["type"], "#FFFFFF")
            scaled = [(x * scale, y * scale) for x, y in room["polygon"]]
            draw.polygon(scaled, fill=color, outline="black")

        for wall in sample["walls"]:
            x1, y1 = wall["start"]
            x2, y2 = wall["end"]
            draw.line(
                [(x1 * scale, y1 * scale), (x2 * scale, y2 * scale)],
                fill="black",
                width=2,
            )

        for door in sample["doors"]:
            x, y = door["position"]
            r = 3
            draw.ellipse(
                [(x * scale - r, y * scale - r), (x * scale + r, y * scale + r)],
                fill="red",
            )

        for win in sample["windows"]:
            x, y = win["position"]
            r = 3
            draw.ellipse(
                [(x * scale - r, y * scale - r), (x * scale + r, y * scale + r)],
                fill="blue",
            )

        return np.array(img)


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of floor-plan dicts into a batch dict."""
    result = {
        "rooms": [s["rooms"] for s in batch],
        "walls": [s["walls"] for s in batch],
        "doors": [s["doors"] for s in batch],
        "windows": [s["windows"] for s in batch],
        "plot_dims": [s["plot_dims"] for s in batch],
        "svg_path": [s["svg_path"] for s in batch],
    }
    if "image" in batch[0]:
        result["images"] = [s["image"] for s in batch]
    return result
