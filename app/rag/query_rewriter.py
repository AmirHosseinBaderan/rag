from app.generation.llm import LLM


class QueryRewriter:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def rewrite(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        prompt = (
            "Rewrite the following user question into a clear "
            "and concise search query.\n\n"
            f"Question:\n{query}\n\n"
            "Search query:"
        )

        rewritten_query = self._llm.generate(prompt).strip()

        if not rewritten_query:
            raise ValueError("Rewritten query cannot be empty")

        return rewritten_query