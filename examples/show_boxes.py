import argparse
import sys
import os

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flowforge.geometry.raster_detector import detect_geometry
from flowforge.geometry.visualize import draw_geometry


def main():
    parser = argparse.ArgumentParser(description="Detect shapes and render bounding boxes overlay.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="debug_boxes.jpg", help="Path to save the overlay image")
    args = parser.parse_args()

    geometry = detect_geometry(args.image)
    out_path = draw_geometry(args.image, geometry, args.out)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()


