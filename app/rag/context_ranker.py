from typing import Any


class ContextRanker:
    def rank(
        self,
        results: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be greater than zero")

        ranked = sorted(
            results,
            key=lambda result: result["score"],
            reverse=True,
        )

        if top_n is not None:
            return ranked[:top_n]

        return ranked