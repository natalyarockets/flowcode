from ..semantic.openai_adapter import OpenAISemanticModel
from ..semantic.ollama_adapter import OllamaSemanticModel
from ..geometry.raster_detector import detect_geometry
from ..ocr.tesseract_ocr import annotate_ocr
from ..graph.builder import parse_semantic_json, build_from_geometry_and_cleanup, build_from_geometry, skeleton_to_json
from ..graph.mermaid import to_mermaid
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

    def extract(self, image_path):
        clean_image = image_path  # preprocessing currently inlined/no-op
        geometry = detect_geometry(clean_image)
        geometry = annotate_ocr(clean_image, geometry)
        # Deterministic graph only
        return build_from_geometry(geometry)

    def review_with_llm(self, image_path, skel):
        """
        Ask LLM to review deterministic graph against image and return revised graph.
        """
        graph_json = skeleton_to_json(skel)
        revised_json = self.semantic_model.review_graph(image_path, graph_json)
        # Reuse legacy parser to parse full graph schema
        revised = parse_semantic_json(revised_json)
        return revised

    def to_mermaid(self, skel):
        return to_mermaid(skel)

    # New flowgraph-oriented helpers
    def extract_flowgraph(self, image_path):
        clean_image = image_path
        geometry = detect_geometry(clean_image)
        geometry = annotate_ocr(clean_image, geometry)
        yesno = detect_yes_no_near_decisions(clean_image, geometry)
        return build_fg(geometry, yes_no_hints=yesno)

    def flowgraph_to_json(self, fg):
        return fg_to_json(fg)

    def flowgraph_to_mermaid(self, fg):
        return fg_to_mermaid(fg)


