# examples/example1_basic.py

import sys
import os
import argparse

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flowforge import FlowchartExtractor
from flowforge.config import FlowforgeConfig
from flowforge.geometry.visualize import draw_geometry


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
    extractor = FlowchartExtractor(FlowforgeConfig(semantic_provider="openai", semantic_model=args.model, api_key=args.api_key, api_base=args.api_base))
    fg = extractor.extract_flowgraph(args.image)
    # Draw overlays from current geometry pass inside extractor? We still draw using original image + boxes:
    # Rebuild overlay via pipeline steps
    # For simplicity, do a light re-run to draw: extract geometry locally
    from flowforge.geometry.raster_detector import detect_geometry
    from flowforge.ocr.tesseract_ocr import annotate_ocr
    g = detect_geometry(args.image)
    g = annotate_ocr(args.image, g)
    out_path = draw_geometry(args.image, g, args.overlay)
    print(f"Saved overlay: {out_path}")

    # 2) Deterministic graph JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        f.write(extractor.flowgraph_to_json(fg))
    print(f"Wrote deterministic graph JSON: {args.out_json}")

    # 3) LLM review to refine the graph
    revised_json = extractor.semantic_model.review_graph(args.image, extractor.flowgraph_to_json(fg))
    with open(args.out_json_mod, "w", encoding="utf-8") as f:
        f.write(revised_json)
    print(f"Wrote LLM-reviewed graph JSON: {args.out_json_mod}")

    # 4) Mermaid export
    with open(args.out_mermaid, "w", encoding="utf-8") as f:
        # If LLM returned JSON, convert to FlowGraph for Mermaid
        from flowforge.graph.flowgraph import FlowGraph, FlowNode
        import json as _json
        obj = _json.loads(revised_json)
        # construct FlowGraph
        nodes = {nid: FlowNode(id=ndata["id"], shape=ndata.get("shape","unknown"), text=ndata.get("text",""), out=ndata.get("out"), out_yes=ndata.get("out_yes"), out_no=ndata.get("out_no")) for nid, ndata in obj.get("nodes", {}).items()}
        from flowforge.graph.flowgraph import FlowGraph as _FG, to_mermaid as _fg_to_mermaid
        fg2 = _FG(nodes=nodes, start_node=obj.get("start_node"), orientation=obj.get("orientation","top-down"))
        f.write(_fg_to_mermaid(fg2))
    print(f"Wrote Mermaid: {args.out_mermaid}")


if __name__ == "__main__":
    main()


