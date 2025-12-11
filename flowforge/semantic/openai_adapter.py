# flowforge/semantic/openai_adapter.py

import base64
import json
import requests
from .base import SemanticModel
from ..utils.json_sanitize import safe_json_extract, strip_code_fences


class OpenAISemanticModel(SemanticModel):
    """
    OpenAI-compatible model that performs final review of FlowGraph JSON.
    """

    def __init__(self, model, api_key, api_base=None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def review_graph(self, image_path, graph_json):
        image_b64 = self._encode_image(image_path)
        prompt = f"""
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


