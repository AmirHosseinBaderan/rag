from typing import Any


class HybridSearch:
    def __init__(
        self,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> None:
        if semantic_weight < 0:
            raise ValueError("semantic_weight cannot be negative")

        if keyword_weight < 0:
            raise ValueError("keyword_weight cannot be negative")

        if semantic_weight + keyword_weight == 0:
            raise ValueError("At least one weight must be greater than zero")

        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight

    def combine(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        scores: dict[str, dict[str, Any]] = {}

        for result in semantic_results:
            scores[result["id"]] = {
                "semantic_score": result["score"],
                "keyword_score": 0.0,
            }

        for result in keyword_results:
            if result["id"] not in scores:
                scores[result["id"]] = {
                    "semantic_score": 0.0,
                    "keyword_score": result["score"],
                }
            else:
                scores[result["id"]]["keyword_score"] = result["score"]

        results = []

        for document_id, score in scores.items():
            final_score = (
                score["semantic_score"] * self._semantic_weight
                + score["keyword_score"] * self._keyword_weight
            )

            results.append(
                {
                    "id": document_id,
                    "score": final_score,
                    "semantic_score": score["semantic_score"],
                    "keyword_score": score["keyword_score"],
                }
            )

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results[:top_k]