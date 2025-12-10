import json
from typing import Any, Dict, List, Union
from .schema import Node, Edge, SemanticSkeleton


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


