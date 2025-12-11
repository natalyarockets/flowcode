Flowchart Library
=================

Flowforge is a minimal toolkit for extracting structured flowcharts from bitmap images.
It combines a lightweight contour-based geometry detector, Tesseract OCR helpers, and
a semantic review step powered by an LLM backend (OpenAI or Ollama). The included example
pipeline shows how to run the full stack: geometry → OCR → graph builders → LLM refinement.

Installation
------------

1. Create a virtual environment that matches your project Python version.

2. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Install `tesseract` on your machine so that `pytesseract` can talk to it. On macOS:

   ```bash
   brew install tesseract
   ```

Configuration
-------------

- Set `OPENAI_API_KEY` (or pass `--api-key` to the example) before running any script that calls
  the OpenAI semantic backend. `OPENAI_API_BASE` can be provided for Azure-hosted endpoints.
- The default semantic model is `gpt-4o-mini`, but you can override it via `FlowforgeConfig`.

Usage
-----

The `examples/example1_basic.py` script demonstrates the full flow:

```bash
python examples/example1_basic.py --image path/to/flowchart.jpg --api-key $OPENAI_API_KEY
```

It produces:

- `debug_boxes.jpg`: overlay of the detected geometry.
- `graph.json`: deterministic graph built from geometry + OCR.
- `graph_modified.json`: LLM-reviewed version of the graph.
- `graph.mmd`: Mermaid notation for easy visualization.

Key components
--------------

- `flowforge.geometry`: Detects shapes using OpenCV, constrained to large contours.
- `flowforge.ocr`: Wraps Tesseract to assign text to shapes and to help with yes/no hints.
- `flowforge.graph`: Builds a `FlowGraph` from geometry/ocr and can export JSON or Mermaid.
- `flowforge.pipeline`: High-level `FlowchartExtractor` that orchestrates calibration, geometry,
  OCR, graph building, and semantic checks.
- `flowforge.config`: Centralizes configuration for the semantic backend.

Development
-----------

- Run the unit tests with `pytest`.
- Update dependencies in `requirements.txt` and re-run `pip install -r requirements.txt` when needed.
- Geometry tweaks happen in `flowforge/geometry/raster_detector.py`; OCR helpers live in `flowforge/ocr`.

License
-------

This repository does not yet include a license file. Add one before publishing.

