from ..semantic.openai_adapter import OpenAISemanticModel
from ..semantic.ollama_adapter import OllamaSemanticModel
from ..geometry.preprocess import preprocess_image
from ..geometry.raster_detector import detect_geometry
from ..graph.builder import parse_semantic_json


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
        clean_image = preprocess_image(image_path)
        geometry = detect_geometry(clean_image)
        semantic_json = self.semantic_model.describe(clean_image, geometry=geometry)
        semantic_skeleton = parse_semantic_json(semantic_json)
        return semantic_skeleton


