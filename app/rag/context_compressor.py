from typing import Any

from app.evaluation.metrics.cosine_similarity import CosineSimilarity
from app.retrieval.embedder import Embedder


class ContextCompressor:
    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.8,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(
                "similarity_threshold must be between 0 and 1"
            )

        self._embedder = embedder
        self._similarity = CosineSimilarity()
        self._similarity_threshold = similarity_threshold

    def compress(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not results:
            raise ValueError("Results cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._embedder.embed(query)

        compressed: list[dict[str, Any]] = []
        selected_vectors: list[list[float]] = []

        for result in results:
            text = result["metadata"].get("text")

            if not text:
                raise ValueError(
                    "Result metadata must contain text"
                )

            vector = self._embedder.embed(text)

            query_similarity = self._similarity.calculate(
                query_vector,
                vector,
            )

            if query_similarity < self._similarity_threshold:
                continue

            is_duplicate = any(
                self._similarity.calculate(
                    vector,
                    selected_vector,
                ) >= self._similarity_threshold
                for selected_vector in selected_vectors
            )

            if is_duplicate:
                continue

            compressed.append(result)
            selected_vectors.append(vector)

            if len(compressed) >= top_k:
                break

        return compressed