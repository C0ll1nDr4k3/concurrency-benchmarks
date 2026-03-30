import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


class ConcurrencyMode(Enum):
    PESSIMISTIC = "pessimistic"
    OPTIMISTIC = "optimistic"


class LayerState(Enum):
    IDLE = "idle"
    READ_LOCKED = "read_locked"
    WRITE_LOCKED = "write_locked"
    CONFLICT = "conflict"


@dataclass
class HNSWLayer:
    layer: int
    state: LayerState = LayerState.IDLE
    versions: tuple = (0,)  # used in optimistic mode


@dataclass
class HNSWNode:
    id: int
    max_layer: int  # highest layer this node belongs to


@dataclass
class HNSWEdge:
    src: int
    dst: int
    layer: int


class NilVecTheme:
    BG = {
        LayerState.IDLE: (248, 248, 250),
        LayerState.READ_LOCKED: (210, 240, 210),
        LayerState.WRITE_LOCKED: (255, 243, 190),
        LayerState.CONFLICT: (255, 210, 210),
    }
    STROKE = {
        LayerState.IDLE: (170, 170, 180),
        LayerState.READ_LOCKED: (30, 140, 30),
        LayerState.WRITE_LOCKED: (180, 140, 0),
        LayerState.CONFLICT: (200, 40, 40),
    }
    BORDER = {
        LayerState.IDLE: (200, 200, 210),
        LayerState.READ_LOCKED: (60, 160, 60),
        LayerState.WRITE_LOCKED: (200, 160, 0),
        LayerState.CONFLICT: (210, 50, 50),
    }
    NODE_FILL = (255, 255, 255)
    PROMO_LINE = (190, 190, 200)
    TEXT_DARK = (40, 40, 40)
    TEXT_LAYER = (80, 80, 100)


def generate_hnsw_viz(
    width: int,
    height: int,
    layers: List[HNSWLayer],
    nodes: List[HNSWNode],
    edges: List[HNSWEdge],
    output_path: str,
    mode: ConcurrencyMode = ConcurrencyMode.PESSIMISTIC,
    dpi: int = 300,
):
    scale = dpi / 72.0
    sw = int(width * scale)
    sh = int(height * scale)

    img = Image.new("RGB", (sw, sh), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_path = "/System/Library/Fonts/SFNSMono.ttf"
        font_main = ImageFont.truetype(font_path, int(10 * scale))
        font_label = ImageFont.truetype(font_path, int(8.5 * scale))
    except:
        font_main = font_label = ImageFont.load_default()

    num_layers = max(l.layer for l in layers) + 1
    layer_map = {l.layer: l for l in layers}

    layer_nodes: List[List[HNSWNode]] = [
        [n for n in nodes if n.max_layer >= layer] for layer in range(num_layers)
    ]

    margin_x = int(40 * scale)
    margin_top = int(20 * scale)
    margin_bot = int(20 * scale)
    layer_gap = int(12 * scale)
    total_gap = layer_gap * (num_layers - 1)
    layer_h = int((sh - margin_top - margin_bot - total_gap) / num_layers)
    oval_rx = int(26 * scale)
    oval_ry = int(13 * scale)

    def layer_y_range(layer: int) -> Tuple[int, int]:
        idx = num_layers - 1 - layer  # layer 0 at bottom
        top = margin_top + idx * (layer_h + layer_gap)
        return top, top + layer_h

    def node_x(node_id: int, layer: int) -> int:
        ids = sorted(n.id for n in layer_nodes[layer])
        pos = ids.index(node_id)
        usable = sw - 2 * margin_x
        if len(ids) == 1:
            return sw // 2
        return margin_x + int(pos * usable / (len(ids) - 1))

    def node_center(node_id: int, layer: int) -> Tuple[int, int]:
        top, bot = layer_y_range(layer)
        return node_x(node_id, layer), (top + bot) // 2

    # 1. Layer bands
    for lyr in layers:
        top, bot = layer_y_range(lyr.layer)
        bg = NilVecTheme.BG[lyr.state]
        border = NilVecTheme.BORDER[lyr.state]
        stroke = NilVecTheme.STROKE[lyr.state]

        draw.rounded_rectangle(
            [margin_x // 2, top, sw - margin_x // 2, bot],
            radius=int(6 * scale),
            fill=bg,
            outline=border,
            width=max(2, int(1.2 * scale)),
        )

        if mode == ConcurrencyMode.PESSIMISTIC:
            if lyr.state == LayerState.READ_LOCKED:
                state_str = "Read Lock"
            elif lyr.state == LayerState.WRITE_LOCKED:
                state_str = "Write Lock"
            elif lyr.state == LayerState.CONFLICT:
                state_str = "Conflict"
            else:
                state_str = ""
            lbl = f"Layer {lyr.layer}" + (f"  [{state_str}]" if state_str else "")
            draw.text(
                (margin_x // 2 + int(6 * scale), top + int(5 * scale)),
                lbl,
                fill=stroke if state_str else NilVecTheme.TEXT_LAYER,
                font=font_label,
            )
        else:
            v_str = " ".join(f"v{v}" for v in lyr.versions)
            retry = "  retry" if lyr.state == LayerState.CONFLICT else ""
            lbl = f"Layer {lyr.layer}  [{v_str}{retry}]"
            draw.text(
                (margin_x // 2 + int(6 * scale), top + int(5 * scale)),
                lbl,
                fill=stroke if lyr.state != LayerState.IDLE else NilVecTheme.TEXT_LAYER,
                font=font_label,
            )

    # 2. Promotion lines (dashed, between layer bands)
    dash_on = int(5 * scale)
    dash_off = int(4 * scale)
    for node in nodes:
        for layer in range(node.max_layer):
            x0, y0 = node_center(node.id, layer)
            x1, y1 = node_center(node.id, layer + 1)
            total = abs(y0 - y1)
            d, drawing = 0, True
            while d < total:
                seg = min(dash_on if drawing else dash_off, total - d)
                if drawing:
                    ya = int(y0 + d * (y1 - y0) / total)
                    yb = int(y0 + (d + seg) * (y1 - y0) / total)
                    draw.line(
                        [(x0, ya), (x0, yb)],
                        fill=NilVecTheme.PROMO_LINE,
                        width=max(1, int(1 * scale)),
                    )
                d += seg
                drawing = not drawing

    # 3. Edges (color from their layer's state)
    for edge in edges:
        lyr = layer_map[edge.layer]
        color = NilVecTheme.STROKE[lyr.state]
        x0, y0 = node_center(edge.src, edge.layer)
        x1, y1 = node_center(edge.dst, edge.layer)

        if mode == ConcurrencyMode.OPTIMISTIC:
            total = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if total == 0:
                continue
            dash_len = int(10 * scale)
            steps = max(1, int(total / dash_len))
            drawing = True
            for i in range(steps + 1):
                t0 = i / (steps + 1)
                t1 = min((i + 0.5) / (steps + 1), 1.0)
                if drawing:
                    draw.line(
                        [
                            (int(x0 + t0 * (x1 - x0)), int(y0 + t0 * (y1 - y0))),
                            (int(x0 + t1 * (x1 - x0)), int(y0 + t1 * (y1 - y0))),
                        ],
                        fill=color,
                        width=max(1, int(1.2 * scale)),
                    )
                drawing = not drawing
        else:
            draw.line([(x0, y0), (x1, y1)], fill=color, width=max(1, int(1.5 * scale)))

    # 4. Node ovals (neutral white; layer color comes from band)
    for node in nodes:
        for layer in range(node.max_layer + 1):
            cx, cy = node_center(node.id, layer)
            lyr = layer_map[layer]
            stroke = NilVecTheme.STROKE[lyr.state]
            draw.ellipse(
                [cx - oval_rx, cy - oval_ry, cx + oval_rx, cy + oval_ry],
                fill=NilVecTheme.NODE_FILL,
                outline=stroke,
                width=max(1, int(1.6 * scale)),
            )

    img.save(output_path, dpi=(dpi, dpi))
    print(f"Saved {mode.value} HNSW visualization to {output_path} ({sw}x{sh})")


if __name__ == "__main__":
    os.makedirs("paper/plots", exist_ok=True)

    layers = [
        HNSWLayer(0, LayerState.WRITE_LOCKED, versions=(14, 15)),
        HNSWLayer(1, LayerState.READ_LOCKED, versions=(8,)),
        HNSWLayer(2, LayerState.IDLE, versions=(3,)),
    ]

    nodes = [
        HNSWNode(0, max_layer=0),
        HNSWNode(1, max_layer=2),
        HNSWNode(2, max_layer=0),
        HNSWNode(3, max_layer=1),
        HNSWNode(4, max_layer=0),
        HNSWNode(5, max_layer=1),
        HNSWNode(6, max_layer=0),
        HNSWNode(7, max_layer=2),
    ]

    edges = [
        HNSWEdge(0, 1, 0),
        HNSWEdge(1, 2, 0),
        HNSWEdge(2, 3, 0),
        HNSWEdge(3, 4, 0),
        HNSWEdge(4, 5, 0),
        HNSWEdge(5, 6, 0),
        HNSWEdge(6, 7, 0),
        HNSWEdge(0, 4, 0),
        HNSWEdge(2, 6, 0),
        HNSWEdge(1, 3, 1),
        HNSWEdge(3, 5, 1),
        HNSWEdge(5, 7, 1),
        HNSWEdge(1, 7, 1),
        HNSWEdge(1, 7, 2),
    ]

    for m in ConcurrencyMode:
        generate_hnsw_viz(
            600,
            480,
            layers,
            nodes,
            edges,
            f"paper/plots/hnsw_{m.value}.png",
            mode=m,
        )
