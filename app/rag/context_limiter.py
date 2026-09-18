from typing import Any


class ContextLimiter:
    def __init__(self, max_characters: int) -> None:
        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than zero"
            )

        self._max_characters = max_characters

    def limit(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        limited: list[dict[str, Any]] = []
        total_characters = 0

        for result in results:
            text = result["metadata"].get("text")

            if not text:
                raise ValueError(
                    "Result metadata must contain text"
                )

            text_length = len(text)

            if total_characters + text_length > self._max_characters:
                break

            limited.append(result)
            total_characters += text_length

        return limited