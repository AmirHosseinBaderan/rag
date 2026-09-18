from dataclasses import dataclass


@dataclass(frozen=True)
class RAGSource:
    id: str
    document_id: str
    source: str
    score: float


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[RAGSource]