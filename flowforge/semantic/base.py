class SemanticModel:
    """
    Abstract base class for semantic flowchart description models.
    All adapters must implement `describe(image) -> str` returning JSON string.
    """

    def describe(self, image, geometry=None):
        """
        Describe a flowchart image and return a JSON string.
        geometry: Optional GeometryOutput providing detected primitives to guide the model.
        """
        raise NotImplementedError("Subclasses must implement describe().")


