from typing import List, Tuple, Dict, Optional
from ..geometry.primitives import GeometryOutput, ShapePrimitive, ConnectorPrimitive


def _safe_imports():
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        return pytesseract, Image
    except Exception as e:
        return None, None


def _crop_bbox(img, bbox: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    return img.crop((x1, y1, x2, y2))


def annotate_ocr(image_path: str, geometry: GeometryOutput) -> GeometryOutput:
    """
    Populate ShapePrimitive.text using Tesseract OCR per-shape crop.
    Graceful no-op if pytesseract or PIL is unavailable.
    """
    pytesseract, Image = _safe_imports()
    if pytesseract is None or Image is None:
        # Return unchanged geometry if OCR not available
        return geometry

    try:
        full = Image.open(image_path)
    except Exception:
        return geometry

    new_shapes: List[ShapePrimitive] = []
    for s in geometry.shapes:
        try:
            crop = _crop_bbox(full, s.bbox)
            # Basic config geared for thin black text on white
            text = pytesseract.image_to_string(
                crop, config="--psm 6 -l eng"
            )
            text = text.strip()
        except Exception:
            text = s.text
        new_shapes.append(
            ShapePrimitive(
                id=s.id,
                bbox=s.bbox,
                shape_type=s.shape_type,
                text=text or s.text,
                confidence=s.confidence,
            )
        )

    # Optionally, detect YES/NO near connectors later.
    return GeometryOutput(shapes=new_shapes, connectors=geometry.connectors, metadata=geometry.metadata)


def detect_yes_no_near_decisions(image_path: str, geometry: GeometryOutput) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Heuristic: look for 'YES'/'NO' text near the left/right sides of decision diamonds.
    Returns mapping: {shape_id: {'left': 'YES'|'NO'|None, 'right': 'YES'|'NO'|None}}
    """
    pytesseract, Image = _safe_imports()
    hints: Dict[str, Dict[str, Optional[str]]] = {}
    if pytesseract is None or Image is None:
        return hints
    try:
        full = Image.open(image_path)
    except Exception:
        return hints

    W, H = full.size
    for s in geometry.shapes:
        if (s.shape_type or "") != "decision":
            continue
        x1, y1, x2, y2 = s.bbox
        pad_x = max(10, int(0.05 * W))
        left_roi = (max(0, x1 - pad_x), y1, x1, y2)
        right_roi = (x2, y1, min(W, x2 + pad_x), y2)
        def read_roi(b):
            if b[2] <= b[0] or b[3] <= b[1]:
                return ""
            try:
                crop = full.crop(b)
                t = pytesseract.image_to_string(crop, config="--psm 6 -l eng").strip().upper()
                return t
            except Exception:
                return ""
        ltxt = read_roi(left_roi)
        rtxt = read_roi(right_roi)
        l = "YES" if "YES" in ltxt else ("NO" if "NO" in ltxt else None)
        r = "YES" if "YES" in rtxt else ("NO" if "NO" in rtxt else None)
        if l or r:
            hints[s.id] = {"left": l, "right": r}
    return hints

