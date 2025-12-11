from typing import Optional

from typing import Optional

from ..semantic.openai_adapter import OpenAISemanticModel
from ..semantic.ollama_adapter import OllamaSemanticModel
from ..geometry.raster_detector import detect_geometry
from ..ocr.tesseract_ocr import annotate_ocr, OCRConfig
from ..graph.flowgraph import build_flowgraph as build_fg, to_json as fg_to_json, to_mermaid as fg_to_mermaid
from ..ocr.tesseract_ocr import detect_yes_no_near_decisions


class FlowchartExtractor:
    def __init__(self, config):
        self.config = config
        self.semantic_model = self._load_semantic_model()

    def _load_semantic_model(self):
        if self.config.semantic_provider == "openai":
            return OpenAISemanticModel(
                model=self.config.semantic_model,
                api_key=self.config.api_key,
                api_base=self.config.api_base,
            )
        elif self.config.semantic_provider == "ollama":
            return OllamaSemanticModel(model=self.config.semantic_model)
        else:
            raise ValueError("Unknown semantic provider")

    # Legacy extract() removed to simplify API; use extract_flowgraph()

    # New flowgraph-oriented helpers
    def extract_flowgraph(
        self,
        image_path,
        *,
        ocr_config: Optional[OCRConfig] = None,
    ):
        clean_image = image_path
        # Calibrate if semantic model available
        params = None
        try:
            params_text = self.semantic_model.calibrate(clean_image)
            import json as _json
            params = _json.loads(params_text)
        except Exception:
            params = None
        geometry = detect_geometry(clean_image, params=params)
        geometry = annotate_ocr(clean_image, geometry, config=ocr_config)
        yesno = detect_yes_no_near_decisions(clean_image, geometry, config=ocr_config)
        forced_orientation = None
        if params and isinstance(params.get("orientation"), str):
            forced_orientation = params["orientation"]
        return build_fg(geometry, yes_no_hints=yesno, forced_orientation=forced_orientation)

    def flowgraph_to_json(self, fg):
        return fg_to_json(fg)

    def flowgraph_to_mermaid(self, fg):
        return fg_to_mermaid(fg)


