import json
import base64
import requests
from .base import SemanticModel
from .prompts import calibrate_prompt, review_prompt
from ..utils.json_sanitize import safe_json_extract, strip_code_fences


class OllamaSemanticModel(SemanticModel):
    def __init__(self, model="qwen2.5vl", api_url="http://localhost:11434", timeout=300, verbose=False):
        self.model = model
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose

    def calibrate(self, image_path):
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = calibrate_prompt()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt, "images": [img_b64]},
            ],
            "stream": False,
        }
        url = f"{self.api_url}/api/chat"
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        raw = data["message"]["content"]
        cleaned = safe_json_extract(raw)
        return cleaned if cleaned else strip_code_fences(raw)

    # Removed legacy describe(); we only perform final review now.

    def review_graph(self, image_path, graph_json):
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = review_prompt(graph_json)

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
