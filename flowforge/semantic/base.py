class SemanticModel:
    """
    Minimal interface for final LLM review only.
    """

    def calibrate(self, image_path):
        """
        Return small JSON with global detection parameters:
        {
          "orientation": "top-down" | "left-right" | "radial" | "swimlane",
          "median_shape_width": int,
          "median_shape_height": int,
          "shape_types_present": [string],
          "arrow_thickness_px": int,
          "estimated_node_count": int,
          "arrow_style": "triangle-head" | "line-only" | "block" | "none"
        }
        """
        raise NotImplementedError("Subclasses must implement calibrate().")

    def review_graph(self, image_path, graph_json):
        """
        Review a canonical FlowGraph JSON against the image and return a revised FlowGraph JSON.
        """
        raise NotImplementedError("Subclasses must implement review_graph().")

