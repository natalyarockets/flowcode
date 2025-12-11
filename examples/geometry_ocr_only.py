import argparse
import json
import sys
import os

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flowforge.geometry.raster_detector import detect_geometry
from flowforge.ocr.tesseract_ocr import annotate_ocr


def main():
    parser = argparse.ArgumentParser(description="Run geometry + OCR and print primitives JSON.")
    parser.add_argument("--image", required=True, help="Path to flowchart image")
    parser.add_argument("--out", default=None, help="Optional path to save JSON")
    args = parser.parse_args()

    g = detect_geometry(args.image)
    g = annotate_ocr(args.image, g)
    data = g.to_dict()
    text = json.dumps(data, indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote geometry+ocr to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

