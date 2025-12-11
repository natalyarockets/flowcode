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
from flowforge.graph.builder import skeleton_to_json


def main():
    parser = argparse.ArgumentParser(description="End-to-end example: geometry overlay + semantic JSON.")
    parser.add_argument("--image", required=True, help="Path to a flowchart image")
    parser.add_argument("--overlay", default="debug_boxes.jpg", help="Path to save bounding-box overlay image")
    parser.add_argument("--out-json", default="graph.json", help="Path to save deterministic graph JSON")
    parser.add_argument("--out-json-mod", default="graph_modified.json", help="Path to save LLM-reviewed graph JSON")
    parser.add_argument("--out-mermaid", default="graph.mmd", help="Path to save Mermaid flowchart")
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

    # 2) Deterministic graph (geometry → OCR → graph)
    config = FlowforgeConfig(
        semantic_provider="openai",
        semantic_model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    extractor = FlowchartExtractor(config)
    skeleton = extractor.extract(args.image)
    with open(args.out_json, "w", encoding="utf-8") as f:
        f.write(skeleton_to_json(skeleton))
    print(f"Wrote deterministic graph JSON: {args.out_json}")

    # 3) LLM review to refine the graph
    revised = extractor.review_with_llm(args.image, skeleton)
    with open(args.out_json_mod, "w", encoding="utf-8") as f:
        from flowforge.graph.builder import skeleton_to_json
        f.write(skeleton_to_json(revised))
    print(f"Wrote LLM-reviewed graph JSON: {args.out_json_mod}")

    # 4) Mermaid export
    with open(args.out_mermaid, "w", encoding="utf-8") as f:
        f.write(extractor.to_mermaid(revised))
    print(f"Wrote Mermaid: {args.out_mermaid}")


if __name__ == "__main__":
    main()


