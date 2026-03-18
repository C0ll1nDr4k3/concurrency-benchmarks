import os
import numpy as np
from scipy.spatial import Voronoi
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import xml.etree.ElementTree as ET


class ConcurrencyMode(Enum):
    PESSIMISTIC = "pessimistic"
    OPTIMISTIC = "optimistic"


class CellState(Enum):
    IDLE = "idle"
    READ_LOCKED = "read_locked"
    WRITE_LOCKED = "write_locked"
    CONFLICT = "conflict"


@dataclass
class CellMetadata:
    id: int
    centroid: Tuple[int, int]
    state: CellState = CellState.IDLE
    versions: tuple = (0,)
    label: str = ""


class NilVecTheme:
    COLORS = {
        CellState.IDLE: (245, 245, 245),
        CellState.READ_LOCKED: (180, 230, 180),
        CellState.WRITE_LOCKED: (255, 235, 150),
        CellState.CONFLICT: (255, 170, 170),
    }
    STROKES = {
        CellState.IDLE: (200, 200, 200),
        CellState.READ_LOCKED: (30, 140, 30),
        CellState.WRITE_LOCKED: (180, 140, 0),
        CellState.CONFLICT: (200, 40, 40),
    }


def _rgb(t: Tuple[int, int, int]) -> str:
    return f"rgb({t[0]},{t[1]},{t[2]})"


def generate_concurrency_viz(
    width: int,
    height: int,
    cells: List[CellMetadata],
    output_path: str,
    mode: ConcurrencyMode = ConcurrencyMode.PESSIMISTIC,
    border_width: float = 1.5,
):
    n = len(cells)
    points = np.array([c.centroid for c in cells], dtype=float)

    # Mirror points across all four edges so every original region is finite
    mirrored = np.vstack(
        [
            points,
            np.column_stack([-points[:, 0], points[:, 1]]),
            np.column_stack([2 * width - points[:, 0], points[:, 1]]),
            np.column_stack([points[:, 0], -points[:, 1]]),
            np.column_stack([points[:, 0], 2 * height - points[:, 1]]),
        ]
    )

    vor = Voronoi(mirrored)

    prio = {
        CellState.CONFLICT: 3,
        CellState.WRITE_LOCKED: 2,
        CellState.READ_LOCKED: 1,
        CellState.IDLE: 0,
    }

    svg = ET.Element(
        "svg",
        attrib={
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )

    # Filled regions
    for i, cell in enumerate(cells):
        region_idx = vor.point_region[i]
        verts_idx = vor.regions[region_idx]
        if -1 in verts_idx or not verts_idx:
            continue
        verts = vor.vertices[verts_idx]
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in verts)
        ET.SubElement(
            svg,
            "polygon",
            attrib={
                "points": pts_str,
                "fill": _rgb(NilVecTheme.COLORS[cell.state]),
                "stroke": "none",
            },
        )

    # Ridge lines between pairs of original cells only
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if p1 >= n or p2 >= n or v1 < 0 or v2 < 0:
            continue
        best = max([cells[p1], cells[p2]], key=lambda c: prio[c.state])
        x1, y1 = vor.vertices[v1]
        x2, y2 = vor.vertices[v2]
        attrs = {
            "x1": f"{x1:.2f}",
            "y1": f"{y1:.2f}",
            "x2": f"{x2:.2f}",
            "y2": f"{y2:.2f}",
            "stroke": _rgb(NilVecTheme.STROKES[best.state]),
            "stroke-width": str(border_width),
        }
        if mode == ConcurrencyMode.OPTIMISTIC:
            attrs["stroke-dasharray"] = "6,6"
        ET.SubElement(svg, "line", attrib=attrs)

    # Centroid dots and labels
    for cell in cells:
        cx, cy = cell.centroid
        dot_color = _rgb(NilVecTheme.STROKES[cell.state])
        ET.SubElement(
            svg,
            "circle",
            attrib={
                "cx": str(cx),
                "cy": str(cy),
                "r": "3",
                "fill": dot_color,
            },
        )

        if mode == ConcurrencyMode.OPTIMISTIC:
            v_text = " ".join(str(v) for v in cell.versions)
            ET.SubElement(
                svg,
                "text",
                attrib={
                    "x": str(cx + 8),
                    "y": str(cy + 4),
                    "font-family": "monospace",
                    "font-size": "13",
                    "fill": "rgb(40,40,40)",
                },
            ).text = v_text
            if cell.state == CellState.CONFLICT:
                ET.SubElement(
                    svg,
                    "text",
                    attrib={
                        "x": str(cx + 8),
                        "y": str(cy + 14),
                        "font-family": "monospace",
                        "font-size": "9",
                        "fill": dot_color,
                    },
                ).text = "Retry"
        else:
            label = cell.label
            if not label:
                if cell.state == CellState.READ_LOCKED:
                    label = "Read"
                elif cell.state == CellState.WRITE_LOCKED:
                    label = "Write"
                elif cell.state == CellState.CONFLICT:
                    label = "Conflict"
            if label:
                ET.SubElement(
                    svg,
                    "text",
                    attrib={
                        "x": str(cx + 8),
                        "y": str(cy + 4),
                        "font-family": "monospace",
                        "font-size": "13",
                        "fill": dot_color,
                    },
                ).text = label

    ET.indent(svg)
    ET.ElementTree(svg).write(output_path, xml_declaration=True, encoding="unicode")
    print(f"Saved {mode.value} visualization to {output_path}")


if __name__ == "__main__":
    os.makedirs("paper/plots", exist_ok=True)

    query_list = [
        CellMetadata(0, (120, 100), CellState.READ_LOCKED, versions=("v12",)),
        CellMetadata(1, (350, 150), CellState.IDLE, versions=("v8",)),
        CellMetadata(2, (180, 380), CellState.WRITE_LOCKED, versions=("v14→v15",)),
        CellMetadata(3, (450, 420), CellState.CONFLICT, versions=("v5→v6",)),
        CellMetadata(4, (100, 250), CellState.IDLE, versions=("v10",)),
        CellMetadata(5, (500, 100), CellState.IDLE, versions=("v10",)),
    ]

    generate_concurrency_viz(
        600,
        500,
        query_list,
        "paper/plots/voronoi_pessimistic.svg",
        ConcurrencyMode.PESSIMISTIC,
    )

    generate_concurrency_viz(
        600,
        500,
        query_list,
        "paper/plots/voronoi_optimistic.svg",
        ConcurrencyMode.OPTIMISTIC,
    )
