# flowforge/geometry/primitives.py

from typing import Any, Dict, List, Optional, Tuple


class ShapePrimitive:
    """
    A single detected shape node in the flowchart.
    """

    def __init__(
        self,
        id: str,
        bbox: Tuple[int, int, int, int],
        shape_type: str,
        text: Optional[str] = None,
        confidence: float = 1.0,
    ):
        self.id = id  # e.g. "s1"
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.shape_type = shape_type  # "process" | "decision" | "terminator" | "unknown" | ...
        self.text = text  # OCR text (raw)
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "shape_type": self.shape_type,
            "text": self.text,
            "confidence": self.confidence,
        }


class ConnectorPrimitive:
    """
    A detected arrow/connector between two shapes.
    """

    def __init__(
        self,
        id: str,
        from_id: str,
        to_id: str,
        label: Optional[str] = None,
        points: Optional[List[Tuple[int, int]]] = None,
        confidence: float = 1.0,
    ):
        self.id = id  # e.g. "c5"
        self.from_id = from_id  # shape id
        self.to_id = to_id  # shape id
        self.label = label  # OCR above arrow
        self.points = points or []  # polyline
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from": self.from_id,
            "to": self.to_id,
            "label": self.label,
            "points": [list(p) for p in self.points],
            "confidence": self.confidence,
        }


class GeometryOutput:
    """
    The complete set of geometric primitives extracted from the image.
    """

    def __init__(
        self,
        shapes: List[ShapePrimitive],
        connectors: List[ConnectorPrimitive],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.shapes = shapes  # [ShapePrimitive]
        self.connectors = connectors  # [ConnectorPrimitive]
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shapes": [s.to_dict() for s in self.shapes],
            "connectors": [c.to_dict() for c in self.connectors],
            "metadata": self.metadata,
        }

