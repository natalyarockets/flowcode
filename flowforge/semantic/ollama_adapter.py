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
        if geometry is not None:
            try:
                geometry_json = json.dumps(geometry.to_dict(), indent=2)
            except Exception:
                geometry_json = None

        intro = (
            "You are a diagram-understanding model. "
            "Extract a JSON object describing the flowchart. "
            "Output ONLY valid JSON."
        )
        if geometry_json:
            intro = (
                "You are a diagram-understanding model. "
                "Use the provided geometric primitives as ground truth for shapes and connectors. "
                "Correct OCR errors and assign semantic meaning but do NOT invent new shapes. "
                "Output ONLY valid JSON.\n\n"
                f"GEOMETRY:\n{geometry_json}\n\n"
            )
        prompt = intro

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
