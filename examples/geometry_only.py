import argparse
import json
import sys
import os

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flowforge.geometry.raster_detector import detect_geometry


def main():
    parser = argparse.ArgumentParser(description="Run geometry detector and print primitives JSON.")
    parser.add_argument("--image", required=True, help="Path to flowchart image")
    parser.add_argument("--out", default=None, help="Optional path to save primitives JSON")
    args = parser.parse_args()

    geometry = detect_geometry(args.image)
    data = geometry.to_dict()
    text = json.dumps(data, indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote geometry to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

