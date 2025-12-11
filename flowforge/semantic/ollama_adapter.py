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

    # Removed legacy describe(); we only perform final review now.

    def review_graph(self, image_path, graph_json):
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            "You are an expert at reading flowcharts. Review and revise the provided FlowGraph JSON "
            "to better match the image. Return ONLY a single JSON object in the SAME FlowGraph schema:\n"
            "{\n"
            '  "orientation": "top-down" | "left-right",\n'
            '  "start_node": "<id or null>",\n'
            '  "nodes": { "<id>": {"id":"<id>","shape":"process|decision|terminator|input_output|unknown","text":"...","out":"<id|null>","out_yes":"<id|null>","out_no":"<id|null>"} }\n'
            "}\n"
            "Constraints: keep node ids identical; edit text/shape/orientation/pointers if needed; prefer minimal edits.\n\n"
            "GRAPH:\n" + graph_json
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
