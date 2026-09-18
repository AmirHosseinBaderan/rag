from typing import Any

from app.evaluation.metrics.cosine_similarity import CosineSimilarity
from app.retrieval.embedder import Embedder


class SemanticReranker:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._similarity = CosineSimilarity()

    def rerank(
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

        reranked: list[dict[str, Any]] = []

        for result in results:
            metadata = result["metadata"]
            text = metadata.get("text")

            if not text:
                raise ValueError("Result metadata must contain text")

            document_vector = self._embedder.embed(text)

            rerank_score = self._similarity.calculate(
                query_vector,
                document_vector,
            )

            reranked.append(
                {
                    **result,
                    "rerank_score": rerank_score,
                }
            )

        reranked.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]