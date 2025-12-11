# flowforge/semantic/openai_adapter.py

import base64
import json
import requests
from .base import SemanticModel
from ..utils.json_sanitize import safe_json_extract, strip_code_fences


class OpenAISemanticModel(SemanticModel):
    """
    Semantic model using any OpenAI-compatible vision model.
    """

    def __init__(self, model, api_key, api_base=None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def describe(self, image_path, geometry=None):
        image_b64 = self._encode_image(image_path)

        geometry_json = None
        geometry_count = None
        shape_ids_preview = None
        if geometry is not None:
            try:
                gd = geometry.to_dict()
                geometry_json = json.dumps(gd, indent=2)
                geometry_count = len(gd.get("shapes", []))
                shape_ids_preview = [s.get("id") for s in gd.get("shapes", [])][:30]
            except Exception:
                geometry_json = None
                geometry_count = None

        # Prompt: if geometry provided → return cleanup JSON only. Otherwise produce full schema.
        if geometry_json:
            prompt = f"""
You are a diagram-understanding model. Return ONLY one JSON object (no prose, no markdown fences).
Task: CLEAN UP the provided geometry+OCR. DO NOT invent or remove shapes. Do NOT output pixel coordinates.

Required JSON keys (cleanup schema):
- mode: "cleanup"
- orientation: "top-down" | "left-right" | null
- node_refinements: [
    {{
      "id": "<shape id>",              // MUST match exactly one GEOMETRY.shapes[].id
      "text": "<cleaned text>",        // cleaned OCR text (empty string if unknown)
      "inferred_shape": "<override or null>", // null to keep geometry.shape_type
      "role": "start" | "end" | null
    }}, ...
  ]
- edges: [
    {{"from": "<shape id>", "to": "<shape id or null>", "label": "YES|NO|... or null"}}, ...
  ]
- notes: [string, ...]

STRICT RULES:
- Produce EXACTLY one node_refinement entry for EACH shape in GEOMETRY.shapes.
- Use the shape 'id' EXACTLY (e.g., s0 → s0).
- Do NOT add or remove shapes.
- Edges only between known shape ids; if unsure set "to" to null or omit label.
{("- GEOMETRY shape count: " + str(geometry_count)) if geometry_count is not None else ""}
{("- Example first ids: " + str(shape_ids_preview)) if shape_ids_preview else ""}

GEOMETRY:
{geometry_json}
"""
        else:
            # Fallback to legacy "full schema" behavior for semantic-only mode
            prompt = """
You are a diagram-understanding model. Return ONLY one JSON object (no prose, no markdown fences).

Required JSON keys: layout, nodes, edges, notes.
...
"""

        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        # The model should produce JSON as a string in message content.
        content = response.json()["choices"][0]["message"]["content"]

        # Clean to ensure we return raw JSON text
        cleaned = safe_json_extract(content)
        if cleaned is not None:
            return cleaned

        # Fallback: strip fences and return raw if cannot validate
        return strip_code_fences(content)


    def review_graph(self, image_path, graph_json):
        image_b64 = self._encode_image(image_path)
        prompt = f"""
You are an expert at reading flowcharts. Review and revise the provided Flowchart Graph JSON to better match the image.
Return ONLY a single JSON object with keys: layout, nodes, edges, notes (same schema).

Constraints:
- Keep node ids identical; do not invent or remove ids.
- You may edit node text, inferred_shape, role, orientation, and edges (add/remove/update labels).
- Prefer minimal edits; only change what is clearly wrong.

GRAPH:
{graph_json}
"""
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        cleaned = safe_json_extract(content)
        return cleaned if cleaned is not None else strip_code_fences(content)


