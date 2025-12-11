# flowforge/geometry/raster_detector.py

from typing import Optional, List, Dict, Any
from .primitives import ShapePrimitive, ConnectorPrimitive, GeometryOutput
import os
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

        def _is_diamond(contour: np.ndarray) -> bool:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width <= 0 or height <= 0:
                return False
            ratio = min(width, height) / max(width, height)
            if ratio < 0.6:
                return False
            angle = abs(rect[2])
            if angle > 90:
                angle = 180 - angle
            return abs(angle - 45) <= 15

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

            if n == 4:
                shape_type = "process"
                if _is_diamond(cnt):
                    shape_type = "decision"
            elif n >= 6:
                shape_type = "terminator"
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
