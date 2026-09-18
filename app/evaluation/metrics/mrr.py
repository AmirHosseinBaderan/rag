class MRR:
    def calculate(
        self,
        relevant_ids: list[list[str]],
        retrieved_ids: list[list[str]],
    ) -> float:
        if not relevant_ids:
            raise ValueError("Evaluation data cannot be empty")

        if len(relevant_ids) != len(retrieved_ids):
            raise ValueError(
                "Relevant IDs and retrieved IDs must have the same length"
            )

        reciprocal_ranks: list[float] = []

        for relevant, retrieved in zip(
            relevant_ids,
            retrieved_ids,
        ):
            reciprocal_rank = 0.0

            for rank, document_id in enumerate(
                retrieved,
                start=1,
            ):
                if document_id in relevant:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_ranks.append(reciprocal_rank)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)