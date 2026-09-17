import re
from typing import Any


class KeywordSearch:
    def search(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_terms = self._tokenize(query)

        results = []

        for document in documents:
            document_terms = self._tokenize(document["text"])

            if not document_terms:
                score = 0.0
            else:
                matches = sum(
                    1
                    for term in query_terms
                    if term in document_terms
                )
                score = matches / len(query_terms)

            results.append(
                {
                    "id": document["id"],
                    "score": score,
                    "metadata": document,
                }
            )

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())