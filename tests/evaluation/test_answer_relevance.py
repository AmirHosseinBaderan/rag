import pytest

from app.evaluation.metrics.answer_relevance import AnswerRelevance


class FakeEmbedder:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    def embed(self, text: str) -> list[float]:
        return self._embeddings[text]


def test_calculates_answer_relevance():
    embedder = FakeEmbedder(
        {
            "What is Python?": [1.0, 0.0],
            "Python is a programming language.": [1.0, 0.0],
        }
    )

    metric = AnswerRelevance(embedder)

    score = metric.calculate(
        question="What is Python?",
        answer="Python is a programming language.",
    )

    assert score == pytest.approx(1.0)


def test_calculates_low_relevance_for_unrelated_answer():
    embedder = FakeEmbedder(
        {
            "What is Python?": [1.0, 0.0],
            "The weather is sunny today.": [0.0, 1.0],
        }
    )

    metric = AnswerRelevance(embedder)

    score = metric.calculate(
        question="What is Python?",
        answer="The weather is sunny today.",
    )

    assert score == pytest.approx(0.0)


def test_rejects_empty_question():
    metric = AnswerRelevance(FakeEmbedder({}))

    with pytest.raises(ValueError, match="Question cannot be empty"):
        metric.calculate(
            question="",
            answer="Some answer",
        )


def test_rejects_empty_answer():
    metric = AnswerRelevance(FakeEmbedder({}))

    with pytest.raises(ValueError, match="Answer cannot be empty"):
        metric.calculate(
            question="Some question",
            answer="",
        )