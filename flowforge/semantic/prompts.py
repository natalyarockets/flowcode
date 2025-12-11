"""Shared prompts for semantic calibration and FlowGraph review."""


def calibrate_prompt() -> str:
    """Return the base calibration prompt for orientation + node stats."""

    return """
You are a vision model. Output ONLY this JSON with integers where appropriate:
{
  "orientation": "top-down" | "left-right" | "radial" | "swimlane",
  "median_shape_width": <int>,
  "median_shape_height": <int>,
  "shape_types_present": ["decision"|"process"|"terminator"|"connector", ...],
  "arrow_thickness_px": <int>,
  "estimated_node_count": <int>,
  "arrow_style": "triangle-head" | "line-only" | "block" | "none"
}
Guidance:
- Estimate typical node width/height in pixels (not text boxes).
- Give a reasonable approximate node count (rounded integer).
""".strip()


def review_prompt(graph_json: str) -> str:
    """Return the FlowGraph review prompt that includes the latest graph JSON."""

    return f"""
You are an expert at reading flowcharts. Review and revise the provided FlowGraph JSON to better match the image.
Return ONLY a single JSON object in the SAME FlowGraph schema:
{{
  "orientation": "top-down" | "left-right",
  "start_node": "<id or null>",
  "nodes": {{
    "<id>": {{"id":"<id>","shape":"process|decision|terminator|input_output|unknown","text":"...","out":"<id|null>","out_yes":"<id|null>","out_no":"<id|null>"}},
    ...
  }}
}}
Constraints:
- Keep node ids identical; do not invent or remove ids.
- You may edit node text, shape, orientation, start_node, and branch pointers (out/out_yes/out_no).
- Prefer minimal edits; only change what is clearly wrong.

FlowGraph JSON to review:
{graph_json}
""".strip()
