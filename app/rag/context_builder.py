from typing import Any


class ContextBuilder:
    def build(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        if not results:
            return ""

        parts: list[str] = []

        for result in results:
            metadata = result["metadata"]

            source = metadata.get(
                "source",
                "unknown",
            )

            text = metadata.get(
                "text",
                "",
            )

            parts.append(
                f"[Source: {source}]\n"
                f"{text}"
            )

        return "\n\n".join(parts)