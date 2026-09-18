import pytest

from app.evaluation.metrics.semantic_context_relevance import (
    SemanticContextRelevance,
)
from app.retrieval.embedder import Embedder


class FakeEmbedder(Embedder):
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    def embed(self, text: str) -> list[float]:
        return self._embeddings[text]


def test_identical_context_is_fully_relevant():
    embedder = FakeEmbedder(
        {
            "question": [1.0, 0.0],
            "relevant context": [1.0, 0.0],
        }
    )
    metric = SemanticContextRelevance(
        embedder=embedder,
    )

    result = metric.evaluate(
        question="question",
        contexts=["relevant context"],
    )

    assert result == pytest.approx(1.0)


def test_orthogonal_context_is_not_relevant():
    embedder = FakeEmbedder(
        {
            "question": [1.0, 0.0],
            "irrelevant context": [0.0, 1.0],
        }
    )
    metric = SemanticContextRelevance(
        embedder=embedder,
    )

    result = metric.evaluate(
        question="question",
        contexts=["irrelevant context"],
    )

    assert result == pytest.approx(0.0)


def test_multiple_contexts_return_average_similarity():
    embedder = FakeEmbedder(
        {
            "question": [1.0, 0.0],
            "relevant context": [1.0, 0.0],
            "irrelevant context": [0.0, 1.0],
        }
    )
    metric = SemanticContextRelevance(
        embedder=embedder,
    )

    result = metric.evaluate(
        question="question",
        contexts=[
            "relevant context",
            "irrelevant context",
        ],
    )

    assert result == pytest.approx(0.5)


def test_empty_question_is_rejected():
    embedder = FakeEmbedder(
        {
            "question": [1.0, 0.0],
        }
    )
    metric = SemanticContextRelevance(
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        metric.evaluate(
            question=" ",
            contexts=["context"],
        )


def test_empty_contexts_are_rejected():
    embedder = FakeEmbedder(
        {
            "question": [1.0, 0.0],
        }
    )
    metric = SemanticContextRelevance(
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="Contexts cannot be empty"):
        metric.evaluate(
            question="question",
            contexts=[],
        )