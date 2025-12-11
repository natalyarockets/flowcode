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
        if geometry is not None:
            try:
                geometry_json = json.dumps(geometry.to_dict(), indent=2)
            except Exception:
                geometry_json = None

        prompt = """
You are a diagram-understanding model. Return ONLY one JSON object (no prose, no markdown fences).

Required JSON keys: layout, nodes, edges, notes.

Schema:
- layout: {
    "orientation": "top-down" | "left-right",
    "swimlanes": boolean,
    "estimated_rows": integer >= 1
  }
- nodes: [
    {
      "id": "n0" | "n1" | ...,
      "approx_position": {"row": int >=0, "col": int >=0},
      "inferred_shape": "process" | "decision" | "terminator" | "input_output" | "connector" | "subprocess",
      "text_summary": string,
      "role": "start" | "end" (optional)
    }, ...
  ]
- edges: [
    {"from": "<node id>", "to": "<node id or null>", "label": string or null, "possible_labels": [string] or null},
    ...
  ]
- notes: [string, ...]

Rules:
- IDs referenced in edges must exist in nodes (unless "to" is null if uncertain).
- Use approx_position as a coarse grid; do NOT output pixel coords.
- Do NOT output Graphviz keys like "rankdir", "node", "edge", "source", "target".
- Output strict JSON only.
"""
        if geometry_json:
            prefix = (
                "Use the provided geometric primitives as ground truth for shapes and connectors. "
                "Correct OCR errors and assign semantic meaning but do NOT invent new shapes.\n\n"
                f"GEOMETRY:\n{geometry_json}\n\n"
            )
            prompt = prefix + prompt

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


