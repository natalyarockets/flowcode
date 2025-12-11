# flowforge/geometry/raster_detector.py

from typing import Optional, List, Tuple
from .primitives import ShapePrimitive, ConnectorPrimitive, GeometryOutput
import os
import math


def detect_geometry(image_path: str) -> GeometryOutput:
    """
    Enhanced geometry extractor for thin-line flowcharts.
    - Uses Otsu threshold + morphological closing to solidify thin strokes
    - Uses RETR_TREE to capture nested contours
    - Classifies rectangles (process), diamonds (decision), ovals/circles (terminator)
    - Detects small circles via HoughCircles (often connectors)
    - Applies simple non-maximum suppression (NMS) on overlapping boxes
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
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Close small gaps in thin strokes
            kernel = np.ones((3, 3), np.uint8)
            thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Contours with hierarchy to capture nested shapes
            contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Dynamic area threshold
            img_area = float(width * height)
            min_area = max(150.0, 0.00015 * img_area)

            rects: List[Tuple[int, int, int, int, str, float]] = []  # x1,y1,x2,y2,type,conf

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri <= 0:
                    continue
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                bbox: Tuple[int, int, int, int] = (int(x), int(y), int(x + w), int(y + h))

                shape_type = "unknown"
                confidence = 0.5

                if len(approx) == 4:
                    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
                    angle = rect[2]
                    if angle < -45:
                        angle = angle + 90
                    angle_abs = abs(angle)
                    ar = w / float(h) if h > 0 else 1.0
                    if angle_abs <= 15:
                        shape_type = "process"
                        confidence = 0.78
                    elif 25 <= angle_abs <= 65:
                        shape_type = "decision"
                        confidence = 0.75
                    else:
                        if 0.6 <= ar <= 1.6:
                            shape_type = "process"
                            confidence = 0.6
                else:
                    # Oval / terminator via circularity
                    circularity = (4.0 * math.pi * area) / (peri * peri)
                    if circularity > 0.7 and len(approx) >= 6:
                        shape_type = "terminator"
                        confidence = 0.68

                rects.append((bbox[0], bbox[1], bbox[2], bbox[3], shape_type, confidence))

            # Hough circle detection for small circles (often connectors)
            try:
                hc = cv2.medianBlur(gray, 3)
                circles = cv2.HoughCircles(
                    hc,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=max(10, int(min(width, height) * 0.02)),
                    param1=100,
                    param2=20,
                    minRadius=int(min(width, height) * 0.008),
                    maxRadius=int(min(width, height) * 0.03),
                )
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for c in circles[0, :]:
                        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                        rects.append((cx - r, cy - r, cx + r, cy + r, "terminator", 0.6))
            except Exception:
                pass

            # Non-maximum suppression on rects to reduce duplicates
            def iou(a, b) -> float:
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                iw = max(0, inter_x2 - inter_x1)
                ih = max(0, inter_y2 - inter_y1)
                inter = iw * ih
                area_a = max(0, (ax2 - ax1) * (ay2 - ay1))
                area_b = max(0, (bx2 - bx1) * (by2 - by1))
                union = area_a + area_b - inter + 1e-6
                return inter / union

            rects_sorted = sorted(rects, key=lambda r: (r[5], (r[2] - r[0]) * (r[3] - r[1])), reverse=True)
            picked: List[Tuple[int, int, int, int, str, float]] = []
            for r in rects_sorted:
                if all(iou((r[0], r[1], r[2], r[3]), (p[0], p[1], p[2], p[3])) < 0.45 for p in picked):
                    picked.append(r)

            # Materialize shapes
            for idx, (x1, y1, x2, y2, stype, conf) in enumerate(picked):
                shapes.append(
                    ShapePrimitive(
                        id=f"s{idx}",
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        shape_type=stype,
                        text=None,
                        confidence=float(conf),
                    )
                )

            detector_name = "opencv-enhanced"
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

