import os
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Any

class ConcurrencyMode(Enum):
    PESSIMISTIC = "pessimistic" # Focus on Locks (Shared/Unique)
    OPTIMISTIC = "optimistic"   # Focus on Versions (vX)

class CellState(Enum):
    IDLE = "idle"
    READ_LOCKED = "read_locked"   # Shared lock
    WRITE_LOCKED = "write_locked" # Exclusive lock
    CONFLICT = "conflict"         # Optimistic conflict/retry due to version mismatch

@dataclass
class CellMetadata:
    id: int
    centroid: Tuple[int, int]
    state: CellState = CellState.IDLE
    versions: tuple[Any] = (0,)
    label: str = ""

class NilVecTheme:
    """Color palette matching the paper's technical diagrams."""
    COLORS = {
        CellState.IDLE: (245, 245, 245),         # Light gray
        CellState.READ_LOCKED: (180, 230, 180),  # Green (reader-only)
        CellState.WRITE_LOCKED: (255, 235, 150), # Yellow (writer-only)
        CellState.CONFLICT: (255, 170, 170),     # Red (conflict/retry)
    }
    STROKES = {
        CellState.IDLE: (200, 200, 200),
        CellState.READ_LOCKED: (30, 140, 30),
        CellState.WRITE_LOCKED: (180, 140, 0),
        CellState.CONFLICT: (200, 40, 40),
    }

def generate_concurrency_viz(
    width: int,
    height: int,
    cells: List[CellMetadata],
    output_path: str,
    mode: ConcurrencyMode = ConcurrencyMode.PESSIMISTIC,
    dpi: int = 300,
    border_width: float = 1.5
):
    """
    Generates a high-resolution Voronoi diagram with thick, clean boundaries.
    """
    scale = dpi / 72.0
    s_width = int(width * scale)
    s_height = int(height * scale)
    t_px = max(1, int(border_width * scale)) # Scaled border thickness

    img = Image.new("RGB", (s_width, s_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 1. Pixel mapping
    pixel_map = {}
    for x in range(s_width):
        for y in range(s_height):
            lx, ly = x / scale, y / scale
            best_cell = min(cells, key=lambda c: (lx-c.centroid[0])**2 + (ly-c.centroid[1])**2)
            pixel_map[(x, y)] = best_cell

    # 2. Render background and thick symmetric boundaries
    # Priority for stroke selection: CONFLICT > WRITE > READ > IDLE
    prio = {
        CellState.CONFLICT: 3,
        CellState.WRITE_LOCKED: 2,
        CellState.READ_LOCKED: 1,
        CellState.IDLE: 0
    }

    half_t = t_px // 2
    for x in range(s_width):
        for y in range(s_height):
            cell = pixel_map[(x, y)]

            # Symmetric boundary check: look in a small window around the pixel
            # to ensure borders are centered on the edge and consistent on all sides.
            is_boundary = False
            # We use a set of cell IDs to track neighbors
            neighbor_ids = {cell.id}

            # Check horizontal and vertical neighborhood
            for offset in range(-half_t, half_t + 1):
                if x + offset >= 0 and x + offset < s_width:
                    neighbor = pixel_map[(x + offset, y)]
                    if neighbor.id != cell.id:
                        is_boundary = True
                        neighbor_ids.add(neighbor.id)
                if y + offset >= 0 and y + offset < s_height:
                    neighbor = pixel_map[(x, y + offset)]
                    if neighbor.id != cell.id:
                        is_boundary = True
                        neighbor_ids.add(neighbor.id)

            if is_boundary:
                # Map IDs back to objects for state lookup
                id_to_cell = {c.id: c for c in cells}
                best_cell = max([id_to_cell[id] for id in neighbor_ids], key=lambda c: prio[c.state])
                stroke = NilVecTheme.STROKES[best_cell.state]

                # OPTIMISTIC mode uses scaled dashed boundaries
                dash_len = int(12 * scale)
                if mode == ConcurrencyMode.OPTIMISTIC and (x + y) % dash_len > (dash_len / 2):
                    img.putpixel((x, y), (255, 255, 255))
                else:
                    img.putpixel((x, y), stroke)
            else:
                bg_color = NilVecTheme.COLORS[cell.state]
                img.putpixel((x, y), bg_color)

    # 3. Annotate with scaled Concurrency Info
    try:
        font_path = "/System/Library/Fonts/SFNSMono.ttf"
        font_main = ImageFont.truetype(font_path, int(13 * scale))
        font_sub = ImageFont.truetype(font_path, int(9 * scale))
    except:
        font_main = font_sub = ImageFont.load_default()

    for cell in cells:
        cx, cy = int(cell.centroid[0] * scale), int(cell.centroid[1] * scale)
        r = int(3 * scale)
        dot_color = NilVecTheme.STROKES[cell.state]
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=dot_color)

        if mode == ConcurrencyMode.OPTIMISTIC:
            # OPTIMISTIC focuses on versions
            v_text = " ".join(f"v{version}" for version in cell.versions)
            draw.text((cx + int(8 * scale), cy - int(6 * scale)), v_text, fill=(40, 40, 40), font=font_main)

            if cell.state == CellState.CONFLICT:
                draw.text((cx + int(8 * scale), cy + int(10 * scale)), "Retry", fill=dot_color, font=font_sub)
        else:
            # PESSIMISTIC focuses on explicit locks
            label = cell.label
            if not label:
                if cell.state == CellState.READ_LOCKED: label = "Read Lock"
                elif cell.state == CellState.WRITE_LOCKED: label = "Write Lock"
                elif cell.state == CellState.CONFLICT: label = "Conflict"

            if label:
                draw.text((cx + int(8 * scale), cy - int(6 * scale)), label, fill=dot_color, font=font_main)

    img.save(output_path, dpi=(dpi, dpi))
    print(f"Saved {mode.value} visualization to {output_path} ({s_width}x{s_height})")

if __name__ == "__main__":
    os.makedirs("paper/plots", exist_ok=True)

    # Standard query list of cells to visualize
    query_list = [
        CellMetadata(0, (120, 100), CellState.READ_LOCKED, versions=(12,)),
        CellMetadata(1, (350, 150), CellState.IDLE, versions=(8,)),
        CellMetadata(2, (180, 380), CellState.WRITE_LOCKED, versions=(14, 15)),
        CellMetadata(3, (450, 420), CellState.CONFLICT, versions=(5, 6)),
        CellMetadata(4, (100, 250), CellState.IDLE, versions=(10,)),
        CellMetadata(5, (500, 100), CellState.IDLE, versions=(10,)),
    ]

    # Output both distinct files
    generate_concurrency_viz(
        600, 500, query_list,
        "paper/plots/voronoi_pessimistic.png",
        ConcurrencyMode.PESSIMISTIC
    )

    generate_concurrency_viz(
        600, 500, query_list,
        "paper/plots/voronoi_optimistic.png",
        ConcurrencyMode.OPTIMISTIC
    )
