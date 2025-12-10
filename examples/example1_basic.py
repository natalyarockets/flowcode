import sys
import os

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flowforge import FlowchartExtractor
from flowforge.config import FlowforgeConfig


config = FlowforgeConfig(
    semantic_provider="openai",
    semantic_model="gpt-4o-mini",
    api_key="YOUR_API_KEY"
)

extractor = FlowchartExtractor(config)
result = extractor.extract("example_flowchart.jpg")

print(result)


