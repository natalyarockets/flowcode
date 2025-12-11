import json
import base64
import requests
from .base import SemanticModel
from ..utils.json_sanitize import safe_json_extract, strip_code_fences


class OllamaSemanticModel(SemanticModel):
    def __init__(self, model="qwen2.5vl", api_url="http://localhost:11434", timeout=300, verbose=False):
        self.model = model
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose

    def describe(self, image_path, geometry=None):
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

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

        if geometry_json:
            prompt = f"""
You are a diagram-understanding model. Return ONLY one JSON object (no prose).
Task: CLEAN UP the provided geometry+OCR. Do NOT invent or remove shapes.

Required JSON keys (cleanup schema):
- mode: "cleanup"
- orientation: "top-down" | "left-right" | null
- node_refinements: [
    {{
      "id": "<shape id>",
      "text": "<cleaned text>",
      "inferred_shape": "<override or null>",
      "role": "start" | "end" | null
    }}, ...
  ]
- edges: [
    {{"from": "<shape id>", "to": "<shape id or null>", "label": "YES|NO|... or null"}}, ...
  ]
- notes: [string, ...]

STRICT RULES:
- Produce EXACTLY one node_refinement entry for EACH shape in GEOMETRY.shapes.
- Use the shape 'id' EXACTLY.
- Edges only between known shape ids; if unsure set "to" to null.
{("- GEOMETRY shape count: " + str(geometry_count)) if geometry_count is not None else ""}
{("- Example first ids: " + str(shape_ids_preview)) if shape_ids_preview else ""}

GEOMETRY:
{geometry_json}
"""
        else:
            prompt = (
                "You are a diagram-understanding model. "
                "Extract a JSON object describing the flowchart. "
                "Output ONLY valid JSON."
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "stream": False
        }


        url = f"{self.api_url}/api/chat"
        r = requests.post(url, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            print("SERVER RESPONSE:", r.text)
        r.raise_for_status()

        data = r.json()
        raw = data["message"]["content"]

        cleaned = safe_json_extract(raw)
        if cleaned:
            return cleaned
        return strip_code_fences(raw)

    def review_graph(self, image_path, graph_json):
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            "You are an expert at reading flowcharts. Review and revise the provided Flowchart Graph JSON "
            "to better match the image. Return ONLY a single JSON object with keys: layout, nodes, edges, notes. "
            "Constraints: keep node ids identical; you may edit node text, inferred_shape, role, orientation, and edges; "
            "prefer minimal edits.\n\nGRAPH:\n" + graph_json
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "stream": False
        }

        url = f"{self.api_url}/api/chat"
        r = requests.post(url, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            print("SERVER RESPONSE:", r.text)
        r.raise_for_status()
        data = r.json()
        raw = data["message"]["content"]
        cleaned = safe_json_extract(raw)
        return cleaned if cleaned else strip_code_fences(raw)
