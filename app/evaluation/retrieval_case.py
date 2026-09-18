from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_ids: list[str]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Query cannot be empty")

        if not self.relevant_ids:
            raise ValueError("Relevant IDs cannot be empty")