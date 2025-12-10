import os

try:
    # Optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

class FlowforgeConfig:
    """
    Central configuration object.

    For now, only the semantic model backend is configurable.
    Later, geometry, OCR, and vector settings plug in here.
    """

    def __init__(
        self,
        semantic_provider="openai",
        semantic_model="gpt-4o-mini",
        api_base=None,
        api_key=None
    ):
        self.semantic_provider = semantic_provider
        self.semantic_model = semantic_model
        # Fallback to env if not provided
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")


