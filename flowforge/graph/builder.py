import json
from typing import Any, Dict, List, Union
from .schema import Node, Edge, SemanticSkeleton
from ..geometry.primitives import GeometryOutput
from dataclasses import asdict


def _normalize_edge_keys(edge_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON keys 'from'/'to' into dataclass fields 'from_id'/'to_id'.
    Leaves other keys as-is.
    """
    normalized = dict(edge_dict)
    # Common synonyms from loosely prompted outputs
    if "source" in normalized and "from" not in normalized:
        normalized["from"] = normalized.pop("source")
    if "target" in normalized and "to" not in normalized:
        normalized["to"] = normalized.pop("target")
    if "from" in normalized:
        normalized["from_id"] = normalized.pop("from")
    if "to" in normalized:
        normalized["to_id"] = normalized.pop("to")
    return normalized


def parse_semantic_json(raw_json: Union[str, Dict[str, Any]]) -> SemanticSkeleton:
    """
    Parse the semantic JSON string or dict into typed dataclasses.
    """
    data: Dict[str, Any]
    if isinstance(raw_json, str):
        data = json.loads(raw_json)
    else:
        data = raw_json

    nodes: List[Node] = []
    for n in data.get("nodes", []):
        nodes.append(
            Node(
                id=n.get("id"),
                approx_position=n.get("approx_position", {"row": 0, "col": 0}),
                inferred_shape=n.get("inferred_shape", "unknown"),
                text_summary=n.get("text_summary", ""),
                role=n.get("role"),
            )
        )

    edges: List[Edge] = []
    for e in data.get("edges", []):
        e_norm = _normalize_edge_keys(e)
        edges.append(
            Edge(
                from_id=e_norm.get("from_id"),
                to_id=e_norm.get("to_id"),
                label=e_norm.get("label"),
                possible_labels=e_norm.get("possible_labels"),
            )
        )

    # Normalize layout orientation if Graphviz-like fields were returned
    layout: Dict[str, Any] = data.get("layout", {}) or {}
    rankdir = layout.get("rankdir")
    if isinstance(rankdir, str):
        rd = rankdir.upper()
        if rd.startswith("LR"):
            layout["orientation"] = "left-right"
        elif rd.startswith("TB") or rd.startswith("TD"):
            layout["orientation"] = "top-down"
    # Default orientation if still missing
    if "orientation" not in layout:
        layout["orientation"] = layout.get("orientation", "top-down")

    return SemanticSkeleton(
        layout=layout,
        nodes=nodes,
        edges=edges,
        notes=data.get("notes", []),
    )


def _approx_positions_from_bboxes(shapes: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Produce coarse row/col from bbox ordering (top-down, left-right).
    Simple bucketing by y with tolerance.
    """
    # Collect (id, bbox)
    items: List[Dict[str, Any]] = []
    for s in shapes:
        sid = s.get("id")
        bbox = s.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        items.append({"id": sid, "x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0})

    # Sort by y then x
    items.sort(key=lambda r: (r["y"], r["x"]))

    # Row bucketing by y distance threshold
    rows: List[List[Dict[str, Any]]] = []
    y_tol = None
    if items:
        ys = [it["y"] for it in items]
        # Adaptive tolerance: 5% of median vertical span or 30px minimum
        y_tol = max(30.0, (max(ys) - min(ys)) * 0.05)

    for it in items:
        placed = False
        for row in rows:
            if abs(row[0]["y"] - it["y"]) <= (y_tol or 30.0):
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])

    # Assign row/col indices
    pos: Dict[str, Dict[str, int]] = {}
    for r_idx, row in enumerate(rows):
        row.sort(key=lambda r: r["x"])
        for c_idx, it in enumerate(row):
            pos[it["id"]] = {"row": r_idx, "col": c_idx}
    return pos


def build_from_geometry_and_cleanup(geometry: GeometryOutput, cleanup_json: Union[str, Dict[str, Any]]) -> SemanticSkeleton:
    """
    Combine geometry primitives with LLM cleanup output into a SemanticSkeleton.
    Expected cleanup schema:
    {
      "mode": "cleanup",
      "orientation": "top-down" | "left-right" | null,
      "node_refinements": [{"id": "s0", "text": "...", "inferred_shape": "process"|null, "role": "start"|"end"|null}, ...],
      "edges": [{"from":"s0","to":"s1","label":"YES"}, ...],
      "notes": [...]
    }
    """
    data: Dict[str, Any]
    if isinstance(cleanup_json, str):
        data = json.loads(cleanup_json)
    else:
        data = cleanup_json

    # Map refinements by id
    refinements: Dict[str, Dict[str, Any]] = {}
    for r in data.get("node_refinements", []) or []:
        rid = r.get("id")
        if rid:
            refinements[rid] = r

    # Compute approx positions from geometry bboxes
    shape_dicts = [s.to_dict() for s in geometry.shapes]
    approx_pos = _approx_positions_from_bboxes(shape_dicts)

    # Nodes: one per geometry shape
    nodes: List[Node] = []
    for s in geometry.shapes:
        sid = s.id
        ref = refinements.get(sid, {})
        inferred_shape = ref.get("inferred_shape") or s.shape_type or "unknown"
        text_summary = ref.get("text")
        if text_summary is None or text_summary == "":
            text_summary = s.text or ""
        role = ref.get("role")
        nodes.append(
            Node(
                id=sid,
                approx_position=approx_pos.get(sid, {"row": 0, "col": 0}),
                inferred_shape=inferred_shape,
                text_summary=text_summary,
                role=role,
            )
        )

    # Edges: take from cleanup if provided; otherwise none (or could fall back to candidates)
    edges: List[Edge] = []
    for e in data.get("edges", []) or []:
        e_norm = _normalize_edge_keys(e)
        edges.append(
            Edge(
                from_id=e_norm.get("from_id"),
                to_id=e_norm.get("to_id"),
                label=e_norm.get("label"),
                possible_labels=e_norm.get("possible_labels"),
            )
        )

    layout: Dict[str, Any] = {}
    orientation = data.get("orientation")
    if orientation in ("top-down", "left-right"):
        layout["orientation"] = orientation
    else:
        layout["orientation"] = "top-down"

    return SemanticSkeleton(
        layout=layout,
        nodes=nodes,
        edges=edges,
        notes=data.get("notes", []),
    )


def build_from_geometry(geometry: GeometryOutput) -> SemanticSkeleton:
    """
    Deterministic graph from geometry+OCR only.
    - Nodes: 1:1 with geometry.shapes
    - Edges: from geometry.connectors (labels optional)
    - Layout: orientation defaults to top-down
    """
    shape_dicts = [s.to_dict() for s in geometry.shapes]
    approx_pos = _approx_positions_from_bboxes(shape_dicts)

    nodes: List[Node] = []
    for s in geometry.shapes:
        sid = s.id
        nodes.append(
            Node(
                id=sid,
                approx_position=approx_pos.get(sid, {"row": 0, "col": 0}),
                inferred_shape=s.shape_type or "unknown",
                text_summary=s.text or "",
                role=None,
            )
        )

    edges: List[Edge] = []
    for c in geometry.connectors:
        edges.append(
            Edge(
                from_id=c.from_id,
                to_id=c.to_id,
                label=c.label,
                possible_labels=None,
            )
        )

    layout: Dict[str, Any] = {"orientation": "top-down"}
    return SemanticSkeleton(layout=layout, nodes=nodes, edges=edges, notes=[])


def skeleton_to_json(skel: SemanticSkeleton) -> str:
    """
    Serialize SemanticSkeleton to canonical JSON text.
    """
    obj = {
        "layout": skel.layout,
        "nodes": [
            {
                "id": n.id,
                "approx_position": n.approx_position,
                "inferred_shape": n.inferred_shape,
                "text_summary": n.text_summary,
                "role": n.role,
            }
            for n in skel.nodes
        ],
        "edges": [
            {
                "from": e.from_id,
                "to": e.to_id,
                "label": e.label,
                "possible_labels": e.possible_labels,
            }
            for e in skel.edges
        ],
        "notes": skel.notes,
    }
    return json.dumps(obj, indent=2, ensure_ascii=False)


