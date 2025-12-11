# examples/example1_basic.py

import sys
import os
import argparse

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flowforge import FlowchartExtractor
from flowforge.config import FlowforgeConfig
from flowforge.geometry.raster_detector import detect_geometry
from flowforge.geometry.visualize import draw_geometry


def main():
    parser = argparse.ArgumentParser(description="End-to-end example: geometry overlay + semantic JSON.")
    parser.add_argument("--image", required=True, help="Path to a flowchart image")
    parser.add_argument("--overlay", default="debug_boxes.jpg", help="Path to save bounding-box overlay image")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE"), help="OpenAI API base (optional)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Missing OpenAI API key. Use --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(2)

    # 1) Geometry detection and overlay
    geometry = detect_geometry(args.image)
    out_path = draw_geometry(args.image, geometry, args.overlay)
    print(f"Saved overlay: {out_path}")

    # 2) Hybrid pipeline (geometry → semantic → graph)
    config = FlowforgeConfig(
        semantic_provider="openai",
        semantic_model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    extractor = FlowchartExtractor(config)
    skeleton = extractor.extract(args.image)

    # 3) Print parsed semantic skeleton (dataclass)
    print(skeleton)


if __name__ == "__main__":
    main()


