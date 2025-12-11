class SemanticModel:
    """
    Minimal interface for final LLM review only.
    """

    def review_graph(self, image_path, graph_json):
        """
        Review a canonical FlowGraph JSON against the image and return a revised FlowGraph JSON.
        """
        raise NotImplementedError("Subclasses must implement review_graph().")

