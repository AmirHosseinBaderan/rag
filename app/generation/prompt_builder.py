class PromptBuilder:
    def build(
        self,
        query: str,
        context: str,
    ) -> str:
        prompt = (
            "Answer the question using only "
            "the provided context.\n\n"
            f"Context:\n{context}"
        )

        prompt += (
            "\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        return prompt