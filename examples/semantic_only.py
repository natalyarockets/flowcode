import argparse
import json
import sys
import os

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


from flowforge.semantic.openai_adapter import OpenAISemanticModel
from flowforge.semantic.ollama_adapter import OllamaSemanticModel


def main():
    parser = argparse.ArgumentParser(description="Generate semantic JSON from a flowchart image.")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"], help="Semantic backend")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-4o-mini or qwen2-vl)")
    parser.add_argument("--api-key", default=None, help="API key for OpenAI-compatible providers")
    parser.add_argument("--api-base", default=None, help="API base for OpenAI-compatible providers")
    parser.add_argument("--api-url", default=None, help="Base URL for Ollama server (default: http://localhost:11434)")
    parser.add_argument("--image", required=True, help="Path to flowchart image")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds (Ollama)")
    parser.add_argument("--verbose", action="store_true", help="Print streaming progress (Ollama)")
    args = parser.parse_args()

    if args.provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        api_base = args.api_base or os.getenv("OPENAI_API_BASE")
        if not api_key:
            print("Error: Missing OpenAI API key. Set --api-key or OPENAI_API_KEY (supports .env).", file=sys.stderr)
            sys.exit(2)
        model = OpenAISemanticModel(model=args.model, api_key=api_key, api_base=api_base)
    else:
        model = OllamaSemanticModel(
            model=args.model,
            api_url=args.api_url or "http://localhost:11434",
            timeout=args.timeout,
            verbose=args.verbose,
        )

    raw = model.describe(args.image)

    # Try to pretty-print JSON; fall back to raw text if not valid JSON
    try:
        parsed = json.loads(raw)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception:
        print(raw)


if __name__ == "__main__":
    main()


