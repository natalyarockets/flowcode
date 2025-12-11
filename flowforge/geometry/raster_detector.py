# flowforge/geometry/raster_detector.py

from typing import Optional, List, Tuple, Dict, Any
from .primitives import ShapePrimitive, ConnectorPrimitive, GeometryOutput
import os
import math


def detect_geometry(image_path: str, params: Optional[Dict[str, Any]] = None) -> GeometryOutput:
    """
    Geometry extractor for flowcharts.

    - Uses Otsu threshold + morphological closing to solidify thin strokes.
    - Uses RETR_TREE to capture nested contours.
    - Classifies rectangles (process), diamonds (decision), ovals/circles (terminator).
    - Applies non-maximum suppression (NMS) on overlapping boxes.
    - Uses simple adjacency (nearest neighbor down/right) to propose connectors.

    `params` is an optional calibration dict from a vision-LLM, e.g.:

        {
          "orientation": "top-down",
          "median_shape_width": 50,
          "median_shape_height": 30,
          "shape_types_present": ["decision", "process", "terminator", "connector"],
          "arrow_thickness_px": 2,
          "estimated_node_count": 20,
          "arrow_style": "triangle-head"
        }
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

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("cv2.imread returned None")

        if width is None or height is None:
            h, w = img.shape[:2]
            width, height = int(w), int(h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thr = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Close small gaps in thin strokes
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Contours with hierarchy to capture nested shapes
        contours, hierarchy = cv2.findContours(
            thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        img_area = float(width * height)

        # --- adaptive size thresholds -----------------------------------
        BASE_MIN_W, BASE_MIN_H = 40, 40
        BASE_MIN_AREA = max(150.0, 0.00015 * img_area)

        mw = int(params.get("median_shape_width", BASE_MIN_W)) if params else BASE_MIN_W
        mh = int(params.get("median_shape_height", BASE_MIN_H)) if params else BASE_MIN_H

        # Clamp to at least base mins so bogus small estimates don't kill us.
        mw = max(mw, BASE_MIN_W)
        mh = max(mh, BASE_MIN_H)

        # Require shapes to be at least ~60% of median in each dimension.
        MIN_W = int(0.6 * mw)
        MIN_H = int(0.6 * mh)

        # Area threshold based on median size, but never below base.
        median_area = float(mw * mh)
        min_area = max(BASE_MIN_AREA, 0.25 * median_area)

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

            # Minimum size filter
            if w < MIN_W or h < MIN_H:
                continue

            shape_type = "unknown"
            confidence = 0.5

            if len(approx) == 4:
                rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
                angle = rect[2]
                if angle < -45:
                    angle = angle + 90
                angle_abs = abs(angle)
                ar = w / float(h) if h > 0 else 1.0

                # Rectangles and diamonds in manuals are often skewed.
                if angle_abs <= 20:
                    shape_type = "process"
                    confidence = 0.75
                elif 20 < angle_abs <= 70:
                    shape_type = "decision"
                    confidence = 0.72
                else:
                    # Fallback: treat as process if roughly rectangular.
                    if 0.5 <= ar <= 4.5:
                        shape_type = "process"
                        confidence = 0.6
            else:
                # Oval / terminator via circularity
                circularity = (4.0 * math.pi * area) / (peri * peri)
                if circularity > 0.7 and len(approx) >= 6:
                    shape_type = "terminator"
                    confidence = 0.68

            # Aspect-ratio plausibility checks (relaxed)
            ar = w / float(h) if h > 0 else 1.0
            if shape_type == "process" and not (0.6 <= ar <= 5.0):
                continue
            if shape_type == "decision" and not (0.5 <= ar <= 1.6):
                continue
            if shape_type == "terminator" and not (0.7 <= ar <= 5.0):
                continue

            rects.append((bbox[0], bbox[1], bbox[2], bbox[3], shape_type, confidence))

        # --- merge touching / overlapping boxes -------------------------
        def _is_close(
            a: Tuple[int, int, int, int],
            b: Tuple[int, int, int, int],
            pad: int = 5,
        ) -> bool:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            return not (
                ax2 < bx1 - pad
                or bx2 < ax1 - pad
                or ay2 < by1 - pad
                or by2 < ay1 - pad
            )

        merged: List[List] = []
        for (x1, y1, x2, y2, st, cf) in rects:
            merged_flag = False
            for m in merged:
                if _is_close((x1, y1, x2, y2), (m[0], m[1], m[2], m[3])):
                    # Expand bounding box
                    m[0] = min(m[0], x1)
                    m[1] = min(m[1], y1)
                    m[2] = max(m[2], x2)
                    m[3] = max(m[3], y2)
                    # Keep higher confidence/type
                    if cf > m[5]:
                        m[4] = st
                        m[5] = cf
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append([x1, y1, x2, y2, st, cf])
        rects = [(m[0], m[1], m[2], m[3], m[4], m[5]) for m in merged]

        # --- NMS on merged rects ----------------------------------------
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

        rects_sorted = sorted(
            rects,
            key=lambda r: (r[5], (r[2] - r[0]) * (r[3] - r[1])),
            reverse=True,
        )
        picked: List[Tuple[int, int, int, int, str, float]] = []
        for r in rects_sorted:
            if all(iou((r[0], r[1], r[2], r[3]), (p[0], p[1], p[2], p[3])) < 0.6 for p in picked):
                picked.append(r)

        # --- remove nested tiny boxes inside much larger ones -----------
        filtered: List[Tuple[int, int, int, int, str, float]] = []
        for r in picked:
            ra = max(1, (r[2] - r[0]) * (r[3] - r[1]))
            keep = True
            for p in picked:
                if r is p:
                    continue
                pa = max(1, (p[2] - p[0]) * (p[3] - p[1]))
                if pa <= ra:
                    continue
                if iou((r[0], r[1], r[2], r[3]), (p[0], p[1], p[2], p[3])) > 0.8 and (ra / pa) < 0.2:
                    keep = False
                    break
            if keep:
                filtered.append(r)
        picked = filtered

        # --- materialize shapes ----------------------------------------
        allowed_types = {"process", "decision", "terminator"}
        for idx, (x1, y1, x2, y2, stype, conf) in enumerate(picked):
            if stype in allowed_types:
                shapes.append(
                    ShapePrimitive(
                        id=f"s{idx}",
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        shape_type=stype,
                        text=None,
                        confidence=float(conf),
                    )
                )

        # Cap to 3× estimated node count (if provided)
        if params and shapes:
            est = int(params.get("estimated_node_count") or 0)
            if est > 0 and len(shapes) > 3 * est:
                shapes = sorted(
                    shapes,
                    key=lambda s: (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]),
                    reverse=True,
                )[: 3 * est]

        # --- build simple adjacency (connectors) ------------------------
        def center(b):
            x1, y1, x2, y2 = b
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

        def overlap_1d(a1, a2, b1, b2):
            return max(0, min(a2, b2) - max(a1, b1))

        id_to_shape = {s.id: s for s in shapes}
        shape_items = list(id_to_shape.items())

        for sid, s in shape_items:
            sx1, sy1, sx2, sy2 = s.bbox
            scx, scy = center(s.bbox)
            best_down = None  # (dist, tid)
            best_right = None

            for tid, t in shape_items:
                if tid == sid:
                    continue
                tx1, ty1, tx2, ty2 = t.bbox
                tcx, tcy = center(t.bbox)

                # Down candidate: t below s with x-overlap
                if tcy > scy and overlap_1d(sx1, sx2, tx1, tx2) > 0.2 * min(
                    sx2 - sx1, tx2 - tx1
                ):
                    dist = tcy - scy
                    if best_down is None or dist < best_down[0]:
                        best_down = (dist, tid)

                # Right candidate: t right of s with y-overlap
                if tcx > scx and overlap_1d(sy1, sy2, ty1, ty2) > 0.2 * min(
                    sy2 - sy1, ty2 - ty1
                ):
                    dist = tcx - scx
                    if best_right is None or dist < best_right[0]:
                        best_right = (dist, tid)

            sc = (int(scx), int(scy))
            if best_down:
                tid = best_down[1]
                t = id_to_shape[tid]
                tc = (int(0.5 * (t.bbox[0] + t.bbox[2])), int(0.5 * (t.bbox[1] + t.bbox[3])))
                connectors.append(
                    ConnectorPrimitive(
                        id=f"c_{sid}_{tid}_v",
                        from_id=sid,
                        to_id=tid,
                        label=None,
                        points=[sc, tc],
                        confidence=0.4,
                    )
                )
            if best_right:
                tid = best_right[1]
                t = id_to_shape[tid]
                tc = (int(0.5 * (t.bbox[0] + t.bbox[2])), int(0.5 * (t.bbox[1] + t.bbox[3])))
                connectors.append(
                    ConnectorPrimitive(
                        id=f"c_{sid}_{tid}_h",
                        from_id=sid,
                        to_id=tid,
                        label=None,
                        points=[sc, tc],
                        confidence=0.4,
                    )
                )

        detector_name = "opencv-calibrated"

    except Exception:
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
