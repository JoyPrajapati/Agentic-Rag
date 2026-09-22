"""Cross-encoder reranker for precision improvement."""

from typing import List, Tuple

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved documents using a cross-encoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str]],
        top_k: int = 3,
    ) -> List[Tuple[str, str, float]]:
        """Rerank (doc_id, text) pairs. Returns [(doc_id, text, score), ...]."""
        if not documents:
            return []

        pairs = [[query, text] for _, text in documents]
        scores = self.model.predict(pairs)

        ranked = [
            (doc_id, text, float(score))
            for (doc_id, text), score in zip(documents, scores)
        ]
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:top_k]