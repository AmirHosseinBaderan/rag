from typing import Any


class MultiQueryRetriever:
    def __init__(
        self,
        retriever,
        query_generator,
    ) -> None:
        self._retriever = retriever
        self._query_generator = query_generator

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        queries = self._query_generator.generate(query)

        results_by_id: dict[str, dict[str, Any]] = {}

        for generated_query in queries:
            results = self._retriever.retrieve(
                query=generated_query,
                top_k=top_k,
            )

            for result in results:
                result_id = result["id"]

                existing = results_by_id.get(result_id)

                if existing is None or result["score"] > existing["score"]:
                    results_by_id[result_id] = result

        results = list(results_by_id.values())

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results[:top_k]