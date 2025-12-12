# flowforge/geometry/raster_detector.py

from typing import Optional, List, Dict, Any, Tuple
from .primitives import ShapePrimitive, ConnectorPrimitive, GeometryOutput
import os
import math
import numpy as np


def _shape_centers(shapes: List[ShapePrimitive]) -> Dict[str, Tuple[float, float]]:
    return {shape.id: ((shape.bbox[0] + shape.bbox[2]) / 2.0, (shape.bbox[1] + shape.bbox[3]) / 2.0) for shape in shapes}


def _closest_shape(point: Tuple[float, float], centers: Dict[str, Tuple[float, float]]) -> Optional[str]:
    best_id = None
    best_dist = float("inf")
    px, py = point
    for sid, (cx, cy) in centers.items():
        dist = math.hypot(cx - px, cy - py)
        if dist < best_dist:
            best_dist = dist
            best_id = sid
    return best_id


def _build_connectors_hough(image: np.ndarray, shapes: List[ShapePrimitive]) -> List[ConnectorPrimitive]:
    import cv2
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(image)[0]

    centers = _shape_centers(shapes)
    connectors: List[ConnectorPrimitive] = []
    seen: set[Tuple[str, str]] = set()
    idx = 0
    if lines is None:
        return connectors
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if math.hypot(x2 - x1, y2 - y1) < 20:
            continue
        start = (x1, y1)
        end = (x2, y2)
        from_id = _closest_shape(start, centers)
        to_id = _closest_shape(end, centers)
        if not from_id or not to_id or from_id == to_id:
            continue
        edge_key = tuple(sorted((from_id, to_id)))
        if edge_key in seen:
            continue
        seen.add(edge_key)
        connectors.append(
            ConnectorPrimitive(
                id=f"c{idx}",
                from_id=from_id,
                to_id=to_id,
                label=None,
                points=[start, end],
                confidence=0.55,
            )
        )
        idx += 1
    print(f"[Hough] produced {len(connectors)} connectors")
    return connectors
import numpy as np


def detect_geometry(image_path: str, params: Optional[Dict[str, Any]] = None) -> GeometryOutput:
    """Robust minimal detector for flowcharts (fixed version)."""

    shapes: List[ShapePrimitive] = []
    connectors: List[ConnectorPrimitive] = []
    detector_name = "simple-adaptive"

    # IMPORTANT FIX: define width/height early
    width, height = 0, 0

    try:
        import cv2

        def _circularity(contour: np.ndarray) -> float:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                return 0.0
            area = cv2.contourArea(contour)
            return (4 * math.pi * area) / (perimeter * perimeter)

        def _is_diamond(contour: np.ndarray) -> bool:
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if w <= 0 or h <= 0:
                return False
            ratio = min(w, h) / max(w, h)
            if ratio < 0.7:
                return False
            angle = abs(rect[2])
            if angle > 90:
                angle = 180 - angle
            return abs(angle - 45) <= 15

        def _is_connector(contour: np.ndarray, w: int, h: int) -> bool:
            circ = _circularity(contour)
            if circ < 0.55 or max(w, h) > 40:
                return False
            return True

        def _is_terminator(w: int, h: int, contour: np.ndarray) -> bool:
            circ = _circularity(contour)
            ratio = max(w, h) / (min(w, h) or 1)
            if circ > 0.72 and min(w, h) > 30:
                return True
            if 1.5 <= ratio <= 2.5 and min(w, h) > 20:
                return True
            return False

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("cv2.imread returned None")

        height, width = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            5
        )

        contours, hierarchy = cv2.findContours(
            thr,
            cv2.RETR_TREE,        # <-- MUST NOT CHANGE
            cv2.CHAIN_APPROX_SIMPLE
        )

        MIN_AREA = 120
        MIN_W, MIN_H = 20, 20

        idx = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_W or h < MIN_H:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            n = len(approx)

            if _is_connector(cnt, w, h):
                shape_type = "connector"
            elif _is_terminator(w, h, cnt):
                shape_type = "terminator"
            elif n == 4:
                shape_type = "decision" if _is_diamond(cnt) else "process"
            else:
                shape_type = "unknown"

            shapes.append(
                ShapePrimitive(
                    id=f"s{idx}",
                    bbox=(x, y, x+w, y+h),
                    shape_type=shape_type,
                    text=None,
                    confidence=0.8,
                )
            )
            idx += 1
        connectors = _build_connectors_hough(thr, shapes)

        detector_name = "simple-adaptive"

    except Exception as e:
        shapes = [
            ShapePrimitive(
                id="s0",
                bbox=(0, 0, 100, 100),
                shape_type="unknown",
                text=None,
                confidence=0.1,
            )
        ]
        detector_name = f"error: {e}"

    metadata = {
        "source": os.path.basename(image_path),
        "detector": detector_name,
        "image_size": {"width": width, "height": height},    # always safe now
    }

    return GeometryOutput(shapes=shapes, connectors=connectors, metadata=metadata)
