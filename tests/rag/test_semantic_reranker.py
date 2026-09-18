import pytest

from app.rag.semantic_reranker import SemanticReranker


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> list[float]:
        return self._vectors[text]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(
        {
            "question": [1.0, 0.0],
            "candidate-a": [0.0, 1.0],
            "candidate-b": [1.0, 0.0],
        }
    )


@pytest.fixture
def reranker(embedder: FakeEmbedder) -> SemanticReranker:
    return SemanticReranker(embedder)


def test_reranks_by_semantic_similarity(
    reranker: SemanticReranker,
) -> None:
    results = [
        {
            "id": "a",
            "score": 0.95,
            "metadata": {"text": "candidate-a"},
        },
        {
            "id": "b",
            "score": 0.70,
            "metadata": {"text": "candidate-b"},
        },
    ]

    reranked = reranker.rerank(
        query="question",
        results=results,
        top_k=2,
    )

    assert [result["id"] for result in reranked] == ["b", "a"]


def test_preserves_original_score_and_adds_rerank_score(
    reranker: SemanticReranker,
) -> None:
    results = [
        {
            "id": "a",
            "score": 0.95,
            "metadata": {"text": "candidate-a"},
        },
    ]

    reranked = reranker.rerank(
        query="question",
        results=results,
        top_k=1,
    )

    assert reranked[0]["score"] == 0.95
    assert reranked[0]["rerank_score"] == pytest.approx(0.0)


def test_applies_top_k(
    reranker: SemanticReranker,
) -> None:
    results = [
        {
            "id": "a",
            "score": 0.8,
            "metadata": {"text": "candidate-a"},
        },
        {
            "id": "b",
            "score": 0.7,
            "metadata": {"text": "candidate-b"},
        },
    ]

    reranked = reranker.rerank(
        query="question",
        results=results,
        top_k=1,
    )

    assert len(reranked) == 1
    assert reranked[0]["id"] == "b"


def test_rejects_empty_query(
    reranker: SemanticReranker,
) -> None:
    with pytest.raises(ValueError, match="Query cannot be empty"):
        reranker.rerank(
            query="",
            results=[],
            top_k=1,
        )


def test_rejects_empty_results(
    reranker: SemanticReranker,
) -> None:
    with pytest.raises(ValueError, match="Results cannot be empty"):
        reranker.rerank(
            query="question",
            results=[],
            top_k=1,
        )


def test_rejects_invalid_top_k(
    reranker: SemanticReranker,
) -> None:
    results = [
        {
            "id": "a",
            "score": 0.8,
            "metadata": {"text": "candidate-a"},
        },
    ]

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank(
            query="question",
            results=results,
            top_k=0,
        )