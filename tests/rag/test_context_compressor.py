import pytest

from app.rag.context_compressor import ContextCompressor


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
            "relevant": [1.0, 0.0],
            "irrelevant": [0.0, 1.0],
            "similar": [1.0, 0.0],
        }
    )


@pytest.fixture
def compressor(embedder: FakeEmbedder) -> ContextCompressor:
    return ContextCompressor(
        embedder=embedder,
        similarity_threshold=0.8,
    )


def test_keeps_semantically_relevant_context(
    compressor: ContextCompressor,
) -> None:
    results = [
        {
            "id": "relevant",
            "score": 0.8,
            "rerank_score": 0.9,
            "metadata": {
                "text": "relevant",
                "source": "doc.txt",
            },
        },
        {
            "id": "irrelevant",
            "score": 0.7,
            "rerank_score": 0.2,
            "metadata": {
                "text": "irrelevant",
                "source": "other.txt",
            },
        },
    ]

    compressed = compressor.compress(
        query="question",
        results=results,
        top_k=2,
    )

    assert [result["id"] for result in compressed] == ["relevant"]


def test_removes_duplicate_contexts(
    compressor: ContextCompressor,
) -> None:
    results = [
        {
            "id": "a",
            "score": 0.9,
            "metadata": {
                "text": "relevant",
            },
        },
        {
            "id": "b",
            "score": 0.8,
            "metadata": {
                "text": "similar",
            },
        },
    ]

    compressed = compressor.compress(
        query="question",
        results=results,
        top_k=2,
    )

    assert len(compressed) == 1
    assert compressed[0]["id"] == "a"


def test_preserves_result_metadata(
    compressor: ContextCompressor,
) -> None:
    result = {
        "id": "a",
        "score": 0.8,
        "rerank_score": 0.9,
        "metadata": {
            "text": "relevant",
            "source": "doc.txt",
            "document_id": "doc-1",
        },
    }

    compressed = compressor.compress(
        query="question",
        results=[result],
        top_k=1,
    )

    assert compressed[0]["score"] == 0.8
    assert compressed[0]["rerank_score"] == 0.9
    assert compressed[0]["metadata"] == result["metadata"]


def test_applies_top_k(
    compressor: ContextCompressor,
) -> None:
    results = [
        {
            "id": "a",
            "score": 0.9,
            "metadata": {"text": "relevant"},
        },
        {
            "id": "b",
            "score": 0.8,
            "metadata": {"text": "irrelevant"},
        },
    ]

    compressed = compressor.compress(
        query="question",
        results=results,
        top_k=1,
    )

    assert len(compressed) == 1


def test_rejects_empty_query(
    compressor: ContextCompressor,
) -> None:
    with pytest.raises(ValueError, match="Query cannot be empty"):
        compressor.compress(
            query="",
            results=[],
            top_k=1,
        )


def test_rejects_empty_results(
    compressor: ContextCompressor,
) -> None:
    with pytest.raises(ValueError, match="Results cannot be empty"):
        compressor.compress(
            query="question",
            results=[],
            top_k=1,
        )


def test_rejects_invalid_top_k(
    compressor: ContextCompressor,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        compressor.compress(
            query="question",
            results=[
                {
                    "id": "a",
                    "score": 0.8,
                    "metadata": {"text": "relevant"},
                }
            ],
            top_k=0,
        )


def test_rejects_invalid_similarity_threshold(
    embedder: FakeEmbedder,
) -> None:
    with pytest.raises(
        ValueError,
        match="similarity_threshold",
    ):
        ContextCompressor(
            embedder=embedder,
            similarity_threshold=1.5,
        )