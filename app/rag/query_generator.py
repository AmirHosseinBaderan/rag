from app.generation.llm import LLM


class QueryGenerator:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def generate(self, query: str) -> list[str]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        prompt = (
            "Generate multiple search queries for the following question.\n"
            "Each query must be on a separate line.\n"
            "Do not add numbering or explanations.\n\n"
            f"Question:\n{query}\n\n"
            "Search queries:"
        )

        response = self._llm.generate(prompt)

        queries = [
            line.strip()
            for line in response.splitlines()
            if line.strip()
        ]

        if not queries:
            raise ValueError("Generated queries cannot be empty")

        return queries