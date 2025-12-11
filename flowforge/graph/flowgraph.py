from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from ..geometry.primitives import GeometryOutput, ShapePrimitive


@dataclass
class FlowNode:
    id: str
    shape: str
    text: str
    out: Optional[str] = None
    out_yes: Optional[str] = None
    out_no: Optional[str] = None


@dataclass
class FlowGraph:
    nodes: Dict[str, FlowNode] = field(default_factory=dict)
    start_node: Optional[str] = None
    orientation: str = "top-down"  # or "left-right"


def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def infer_orientation(geometry: GeometryOutput) -> str:
    """
    Determine orientation by majority nearest-neighbor direction.
    """
    shapes = geometry.shapes
    if len(shapes) < 2:
        return "top-down"
    down_votes = 0
    right_votes = 0
    # naive: nearest neighbor by y-forward and x-forward
    for s in shapes:
        scx, scy = _center(s.bbox)
        best_down = None
        best_right = None
        for t in shapes:
            if t.id == s.id:
                continue
            tcx, tcy = _center(t.bbox)
            if tcy > scy:
                dy = tcy - scy
                if best_down is None or dy < best_down:
                    best_down = dy
            if tcx > scx:
                dx = tcx - scx
                if best_right is None or dx < best_right:
                    best_right = dx
        if best_down is not None:
            down_votes += 1
        if best_right is not None:
            right_votes += 1
    return "top-down" if down_votes >= right_votes else "left-right"


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def build_flowgraph(
    geometry: GeometryOutput,
    yes_no_hints: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    forced_orientation: Optional[str] = None,
) -> FlowGraph:
    """
    Deterministic adjacency construction:
    - orientation via infer_orientation
    - for each node: default 'out' = nearest in-forward-direction with axis overlap
    - for decision diamonds: pick left and right nearest with y-overlap (TD) or up/down when LR
    - yes_no_hints: optional {'sX': {'left':'YES','right':'NO'}} overlays
    """
    shapes = geometry.shapes
    orientation = forced_orientation if forced_orientation else infer_orientation(geometry)
    id_to_shape: Dict[str, ShapePrimitive] = {s.id: s for s in shapes}
    nodes: Dict[str, FlowNode] = {
        s.id: FlowNode(id=s.id, shape=s.shape_type or "unknown", text=s.text or "") for s in shapes
    }

    # Helper: directional nearest neighbors with overlap
    for s in shapes:
        sx1, sy1, sx2, sy2 = s.bbox
        scx, scy = _center(s.bbox)
        best = None  # (dist, tid)
        for t in shapes:
            if t.id == s.id:
                continue
            tx1, ty1, tx2, ty2 = t.bbox
            tcx, tcy = _center(t.bbox)
            if orientation == "top-down":
                if tcy <= scy:
                    continue
                # require x-overlap
                if _overlap_1d(sx1, sx2, tx1, tx2) <= 0.15 * min(sx2 - sx1, tx2 - tx1):
                    continue
                dist = tcy - scy
            else:
                if tcx <= scx:
                    continue
                # require y-overlap
                if _overlap_1d(sy1, sy2, ty1, ty2) <= 0.15 * min(sy2 - sy1, ty2 - ty1):
                    continue
                dist = tcx - scx
            if best is None or dist < best[0]:
                best = (dist, t.id)
        if best:
            nodes[s.id].out = best[1]

    # Decision branch candidates
    for s in shapes:
        if (s.shape_type or "") != "decision":
            continue
        sx1, sy1, sx2, sy2 = s.bbox
        scx, scy = _center(s.bbox)
        left_best = None
        right_best = None
        for t in shapes:
            if t.id == s.id:
                continue
            tx1, ty1, tx2, ty2 = t.bbox
            tcx, tcy = _center(t.bbox)
            if orientation == "top-down":
                # y-overlap required
                if _overlap_1d(sy1, sy2, ty1, ty2) <= 0.15 * min(sy2 - sy1, ty2 - ty1):
                    continue
                # classify left/right by center
                if tcx < scx:
                    # left
                    dist = scx - tcx
                    if left_best is None or dist < left_best[0]:
                        left_best = (dist, t.id)
                else:
                    dist = tcx - scx
                    if right_best is None or dist < right_best[0]:
                        right_best = (dist, t.id)
            else:
                # left-right orientation → use up/down branches
                if _overlap_1d(sx1, sx2, tx1, tx2) <= 0.15 * min(sx2 - sx1, tx2 - tx1):
                    continue
                if tcy < scy:
                    # "left" slot repurposed as up
                    dist = scy - tcy
                    if left_best is None or dist < left_best[0]:
                        left_best = (dist, t.id)
                else:
                    dist = tcy - scy
                    if right_best is None or dist < right_best[0]:
                        right_best = (dist, t.id)

        # Assign provisional
        if left_best:
            nodes[s.id].out_no = left_best[1]  # default left->NO
        if right_best:
            nodes[s.id].out_yes = right_best[1]  # default right->YES
        # Optional override from OCR hints
        if yes_no_hints and s.id in yes_no_hints:
            hints = yes_no_hints[s.id]
            # If hints swap, honor them
            if hints.get("left") == "YES" and left_best and right_best:
                nodes[s.id].out_yes = left_best[1]
                nodes[s.id].out_no = right_best[1]
            elif hints.get("right") == "NO" and left_best and right_best:
                nodes[s.id].out_yes = left_best[1]
                nodes[s.id].out_no = right_best[1]

    # Start node: choose the node with no incoming candidates; fallback to min y
    incoming: Dict[str, int] = {nid: 0 for nid in nodes}
    for nid, n in nodes.items():
        for tid in (n.out, n.out_yes, n.out_no):
            if tid and tid in incoming:
                incoming[tid] += 1
    start = None
    for nid, cnt in incoming.items():
        if cnt == 0:
            start = nid
            break
    if start is None and shapes:
        # fallback: top-most
        start = min(shapes, key=lambda s: _center(s.bbox)[1])[0].id if hasattr(min(shapes, key=lambda s: _center(s.bbox)[1]), 'id') else shapes[0].id

    return FlowGraph(nodes=nodes, start_node=start, orientation=orientation)


def to_json(flow: FlowGraph) -> str:
    out = {
        "orientation": flow.orientation,
        "start_node": flow.start_node,
        "nodes": {
            nid: {
                "id": n.id,
                "shape": n.shape,
                "text": n.text,
                "out": n.out,
                "out_yes": n.out_yes,
                "out_no": n.out_no,
            }
            for nid, n in flow.nodes.items()
        },
    }
    import json
    return json.dumps(out, indent=2, ensure_ascii=False)


def to_mermaid(flow: FlowGraph) -> str:
    orient = "TD" if flow.orientation != "left-right" else "LR"
    lines: List[str] = [f"flowchart {orient}"]
    def node_label(n: FlowNode) -> str:
        txt = (n.text or "").replace('"', "'")
        if n.shape == "decision":
            return f"{{{txt}}}"
        if n.shape == "terminator":
            return f"([{txt}])"
        if n.shape == "input_output":
            return f"[/{txt}/]"
        return f"[{txt}]"
    for n in flow.nodes.values():
        lines.append(f"    {n.id}{node_label(n)}")
    # Edges
    for n in flow.nodes.values():
        if n.out:
            lines.append(f"    {n.id} --> {n.out}")
        if n.out_yes:
            lines.append(f"    {n.id} --|YES|--> {n.out_yes}")
        if n.out_no:
            lines.append(f"    {n.id} --|NO|--> {n.out_no}")
    return "\n".join(lines)

