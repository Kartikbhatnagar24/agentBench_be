from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        # This automatically downloads the model weights from HF on the first run
        self.model = CrossEncoder(model_name)

    def invoke(self, pairs: list[list[str]]) -> list[float]:
        """
        Evaluates query-document pairs.
        Returns a list of float scores representing the relevance.
        """
        if not pairs:
            return []
        # predict returns numpy floats, convert them to standard Python floats
        scores = self.model.predict(pairs)
        return [float(score) for score in scores]

# Export the reranker instance
ranker = BGEReranker()
