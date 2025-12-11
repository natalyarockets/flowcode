# examples/example1_basic.py

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

from flowforge import FlowchartExtractor
from flowforge.config import FlowforgeConfig
from flowforge.geometry.raster_detector import detect_geometry
from flowforge.geometry.visualize import draw_geometry
from flowforge.ocr.tesseract_ocr import OCRConfig, annotate_ocr, detect_yes_no_near_decisions
from flowforge.graph.flowgraph import FlowGraph as FlowGraphClass, FlowNode, build_flowgraph, to_mermaid


@dataclass
class FlowchartRunResult:
    """Structured data leaking out of the flowchart run."""

    deterministic_json: str
    reviewed_json: str
    mermaid: str
    overlay_path: Optional[str]


def run_flowchart_pipeline(
    image_path: str,
    *,
    config: Optional[FlowforgeConfig] = None,
    extractor: Optional[FlowchartExtractor] = None,
    overlay_path: Optional[str] = "debug_boxes.jpg",
    deterministic_json_path: Optional[str] = "graph.json",
    reviewed_json_path: Optional[str] = "graph_modified.json",
    mermaid_path: Optional[str] = "graph.mmd",
    ocr_config: Optional[OCRConfig] = None,
) -> FlowchartRunResult:
    """
    Run the full pipeline and return the generated artifacts.

    The caller can optionally pass in their own `FlowchartExtractor` or configuration.
    All filesystem writes are gated behind an explicit output path to keep this friendly for
    document-processing workflows that only need the strings.
    """

    config = config or FlowforgeConfig()
    extractor = extractor or FlowchartExtractor(config)

    params = None
    try:
        calibration = extractor.semantic_model.calibrate(image_path)
        params = json.loads(calibration)
    except Exception:
        params = None

    geometry = detect_geometry(image_path, params=params)
    geometry = annotate_ocr(image_path, geometry, config=ocr_config)
    yesno = detect_yes_no_near_decisions(image_path, geometry, config=ocr_config)
    forced_orientation = params.get("orientation") if isinstance(params, dict) else None
    flowgraph = build_flowgraph(
        geometry,
        yes_no_hints=yesno,
        forced_orientation=forced_orientation,
    )

    deterministic_json = extractor.flowgraph_to_json(flowgraph)
    reviewed_json = extractor.semantic_model.review_graph(image_path, deterministic_json)

    reviewed_obj = json.loads(reviewed_json)
    mermaid_graph = FlowGraphClass(
        nodes={
            nid: FlowNode(
                id=ndata["id"],
                shape=ndata.get("shape", "unknown"),
                text=ndata.get("text", ""),
                out=ndata.get("out"),
                out_yes=ndata.get("out_yes"),
                out_no=ndata.get("out_no"),
            )
            for nid, ndata in reviewed_obj.get("nodes", {}).items()
        },
        start_node=reviewed_obj.get("start_node"),
        orientation=reviewed_obj.get("orientation", "top-down"),
    )
    mermaid_text = to_mermaid(mermaid_graph)

    if overlay_path:
        draw_geometry(image_path, geometry, overlay_path)

    if deterministic_json_path:
        with open(deterministic_json_path, "w", encoding="utf-8") as dest:
            dest.write(deterministic_json)

    if reviewed_json_path:
        with open(reviewed_json_path, "w", encoding="utf-8") as dest:
            dest.write(reviewed_json)

    if mermaid_path:
        with open(mermaid_path, "w", encoding="utf-8") as dest:
            dest.write(mermaid_text)

    return FlowchartRunResult(
        deterministic_json=deterministic_json,
        reviewed_json=reviewed_json,
        mermaid=mermaid_text,
        overlay_path=overlay_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the Flowforge pipeline on one image.")
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
        parser.error("Missing OpenAI API key. Use --api-key or set OPENAI_API_KEY.")

    config = FlowforgeConfig(
        semantic_provider="openai",
        semantic_model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )

    result = run_flowchart_pipeline(
        args.image,
        config=config,
        overlay_path=args.overlay,
        deterministic_json_path=args.out_json,
        reviewed_json_path=args.out_json_mod,
        mermaid_path=args.out_mermaid,
    )

    print(f"Saved overlay: {result.overlay_path}")
    print(f"Wrote deterministic graph JSON: {args.out_json}")
    print(f"Wrote LLM-reviewed graph JSON: {args.out_json_mod}")
    print(f"Wrote Mermaid: {args.out_mermaid}")


if __name__ == "__main__":
    main()


