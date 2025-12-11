from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from ..geometry.primitives import GeometryOutput, ShapePrimitive


@dataclass
class OCRConfig:
    lang: str = "eng"
    psm: int = 6
    oem: int = 3
    whitelist: Optional[str] = None
    preprocess: bool = True


def _safe_imports():
    try:
        import pytesseract  # type: ignore
        from PIL import Image, ImageFilter, ImageOps  # type: ignore
        return pytesseract, Image, ImageFilter, ImageOps
    except Exception:
        return None, None, None, None


def _crop_bbox(img, bbox: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    return img.crop((x1, y1, x2, y2))


def _preprocess_crop(crop, ImageFilter, ImageOps, config: OCRConfig):
    gray = crop.convert("L")
    if config.preprocess:
        gray = ImageOps.autocontrast(gray)
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return gray


def _tesseract_config(config: OCRConfig) -> str:
    parts = [f"--psm {config.psm}", f"--oem {config.oem}", "-l", config.lang]
    if config.whitelist:
        parts.append(f"-c tessedit_char_whitelist={config.whitelist}")
    return " ".join(parts)


def annotate_ocr(image_path: str, geometry: GeometryOutput, *, config: Optional[OCRConfig] = None) -> GeometryOutput:
    pytesseract, Image, ImageFilter, ImageOps = _safe_imports()
    if pytesseract is None or Image is None:
        return geometry

    config = config or OCRConfig()
    try:
        with Image.open(image_path) as full:
            full = full.convert("RGB")
            new_shapes: List[ShapePrimitive] = []
            tess_config = _tesseract_config(config)
            for shape in geometry.shapes:
                text = shape.text
                try:
                    crop = _crop_bbox(full, shape.bbox)
                    processed = _preprocess_crop(crop, ImageFilter, ImageOps, config)
                    text = pytesseract.image_to_string(processed, config=tess_config).strip()
                except Exception:
                    pass
                new_shapes.append(
                    ShapePrimitive(
                        id=shape.id,
                        bbox=shape.bbox,
                        shape_type=shape.shape_type,
                        text=text or shape.text,
                        confidence=shape.confidence,
                    )
                )
    except Exception:
        return geometry

    return GeometryOutput(shapes=new_shapes, connectors=geometry.connectors, metadata=geometry.metadata)


def detect_yes_no_near_decisions(
    image_path: str,
    geometry: GeometryOutput,
    *,
    config: Optional[OCRConfig] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    pytesseract, Image, ImageFilter, ImageOps = _safe_imports()
    hints: Dict[str, Dict[str, Optional[str]]] = {}
    if pytesseract is None or Image is None:
        return hints

    config = config or OCRConfig()
    tess_config = _tesseract_config(config)
    try:
        with Image.open(image_path) as full:
            full = full.convert("RGB")
            W, H = full.size
            for shape in geometry.shapes:
                if (shape.shape_type or "") != "decision":
                    continue
                x1, y1, x2, y2 = shape.bbox
                pad_x = max(10, int(0.05 * W))
                left_roi = (max(0, x1 - pad_x), y1, x1, y2)
                right_roi = (x2, y1, min(W, x2 + pad_x), y2)

                def read_roi(box):
                    if box[2] <= box[0] or box[3] <= box[1]:
                        return ""
                    try:
                        crop = full.crop(box)
                        processed = _preprocess_crop(crop, ImageFilter, ImageOps, config)
                        return pytesseract.image_to_string(processed, config=tess_config).strip().upper()
                    except Exception:
                        return ""

                ltxt = read_roi(left_roi)
                rtxt = read_roi(right_roi)
                l = "YES" if "YES" in ltxt else ("NO" if "NO" in ltxt else None)
                r = "YES" if "YES" in rtxt else ("NO" if "NO" in rtxt else None)
                if l or r:
                    hints[shape.id] = {"left": l, "right": r}
    except Exception:
        pass

    return hints

