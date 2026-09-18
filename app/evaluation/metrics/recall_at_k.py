class RecallAtK:
    def calculate(
        self,
        relevant_ids: list[str],
        retrieved_ids: list[str],
        k: int,
    ) -> float:
        if not relevant_ids:
            raise ValueError("Relevant IDs cannot be empty")

        if k <= 0:
            raise ValueError("k must be greater than zero")

        top_k = retrieved_ids[:k]

        retrieved_relevant = sum(
            1
            for document_id in relevant_ids
            if document_id in top_k
        )

        return retrieved_relevant / len(relevant_ids)