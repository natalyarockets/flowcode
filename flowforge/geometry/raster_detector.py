from typing import Optional, List, Tuple
from .primitives import ShapePrimitive, ConnectorPrimitive, GeometryOutput
import os
import math


def detect_geometry(image_path: str) -> GeometryOutput:
    """
    Basic geometry extractor.
    - If OpenCV is available: detect rectangles, diamonds, and ovals via contour analysis.
    - Otherwise: return a single unknown shape covering the image (placeholder).
    """
    width: Optional[int] = None
    height: Optional[int] = None

    # Try to read image dimensions (Pillow optional)
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as im:
            width, height = im.size
    except Exception:
        pass

    detector_name = "placeholder"
    shapes: List[ShapePrimitive] = []
    connectors: List[ConnectorPrimitive] = []

    # Try OpenCV path
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = cv2.imread(image_path)
        if img is not None:
            if width is None or height is None:
                h, w = img.shape[:2]
                width, height = int(w), int(h)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Adaptive threshold handles varied lighting/backgrounds
            thr = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
            )
            # Clean noise
            kernel = np.ones((3, 3), np.uint8)
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            shape_idx = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:  # discard tiny specks
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                bbox: Tuple[int, int, int, int] = (int(x), int(y), int(x + w), int(y + h))

                # Classify shape
                shape_type = "unknown"
                confidence = 0.5

                if len(approx) == 4:
                    # Rectangle vs diamond via minAreaRect angle
                    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
                    angle = rect[2]
                    # Normalize angle to [-45, 45]
                    if angle < -45:
                        angle = angle + 90
                    angle_abs = abs(angle)

                    # If near axis-aligned, call it a rectangle (process)
                    if angle_abs <= 15:
                        shape_type = "process"
                        confidence = 0.75
                    # If rotated notably, call it a diamond (decision)
                    elif 25 <= angle_abs <= 65:
                        shape_type = "decision"
                        confidence = 0.7
                    else:
                        # Fallback heuristic by aspect ratio to bias toward process
                        ar = w / float(h) if h > 0 else 1.0
                        if 0.6 <= ar <= 1.6:
                            shape_type = "process"
                            confidence = 0.6
                        else:
                            shape_type = "unknown"
                            confidence = 0.5
                else:
                    # Oval / terminator via circularity
                    # circularity = 4*pi*Area / Perimeter^2
                    if peri > 0:
                        circularity = (4.0 * math.pi * area) / (peri * peri)
                    else:
                        circularity = 0.0
                    if circularity > 0.7 and len(approx) >= 6:
                        shape_type = "terminator"
                        confidence = 0.65

                shapes.append(
                    ShapePrimitive(
                        id=f"s{shape_idx}",
                        bbox=bbox,
                        shape_type=shape_type,
                        text=None,
                        confidence=confidence,
                    )
                )
                shape_idx += 1

            detector_name = "opencv-basic"
        else:
            # Could not read image; fall back
            raise RuntimeError("cv2.imread returned None")

    except Exception:
        # Fallback placeholder if OpenCV path fails or not installed
        if width is None or height is None:
            width, height = 0, 0
        shapes = [
            ShapePrimitive(
                id="s0",
                bbox=(0, 0, int(width), int(height)),
                shape_type="unknown",
                text=None,
                confidence=0.1,
            )
        ]
        connectors = []
        detector_name = "placeholder"

    metadata = {
        "source": os.path.basename(image_path),
        "detector": detector_name,
        "image_size": {"width": width, "height": height},
    }

    return GeometryOutput(shapes=shapes, connectors=connectors, metadata=metadata)

