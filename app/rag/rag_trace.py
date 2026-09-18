from dataclasses import dataclass


@dataclass(frozen=True)
class RAGTrace:
    query: str
    retrieved_count: int
    final_context_count: int
    retrieval_duration_ms: float
    generation_duration_ms: float

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Query cannot be empty")

        if (
            self.retrieved_count < 0
            or self.final_context_count < 0
        ):
            raise ValueError("Counts cannot be negative")

        if (
            self.retrieval_duration_ms < 0
            or self.generation_duration_ms < 0
        ):
            raise ValueError(
                "Durations cannot be negative"
            )