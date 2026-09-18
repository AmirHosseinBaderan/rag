import pytest

from app.evaluation.metrics.semantic_faithfulness import SemanticFaithfulness


class FakeEmbedder:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    def embed(self, text: str) -> list[float]:
        return self._embeddings[text]


def test_calculates_faithfulness_from_best_context_match():
    embedder = FakeEmbedder(
        {
            "Python was created by Guido.": [1.0, 0.0],
            "Python is a programming language.": [1.0, 0.0],
            "The sky is blue.": [0.0, 1.0],
        }
    )

    metric = SemanticFaithfulness(embedder)

    score = metric.calculate(
        answer="Python was created by Guido.",
        contexts=[
            "Python is a programming language.",
            "The sky is blue.",
        ],
    )

    assert score == pytest.approx(1.0)


def test_uses_best_matching_context_for_multiple_sentences():
    embedder = FakeEmbedder(
        {
            "Python is a programming language.": [1.0, 0.0],
            "Guido created Python.": [0.0, 1.0],
            "Python is popular.": [1.0, 0.0],
            "Guido created Python in 1991.": [0.0, 1.0],
            "Unrelated context.": [1.0, 1.0],
        }
    )

    metric = SemanticFaithfulness(embedder)

    score = metric.calculate(
        answer=(
            "Python is a programming language. "
            "Guido created Python."
        ),
        contexts=[
            "Python is popular.",
            "Guido created Python in 1991.",
        ],
    )

    assert score == pytest.approx(1.0)


def test_returns_zero_when_answer_has_no_matching_context():
    embedder = FakeEmbedder(
        {
            "The moon is made of cheese.": [1.0, 0.0],
            "Python is a programming language.": [0.0, 1.0],
        }
    )

    metric = SemanticFaithfulness(embedder)

    score = metric.calculate(
        answer="The moon is made of cheese.",
        contexts=[
            "Python is a programming language.",
        ],
    )

    assert score == pytest.approx(0.0)


def test_rejects_empty_answer():
    metric = SemanticFaithfulness(FakeEmbedder({}))

    with pytest.raises(ValueError, match="Answer cannot be empty"):
        metric.calculate(
            answer="",
            contexts=["Some context"],
        )


def test_rejects_empty_contexts():
    metric = SemanticFaithfulness(FakeEmbedder({}))

    with pytest.raises(ValueError, match="Contexts cannot be empty"):
        metric.calculate(
            answer="Some answer",
            contexts=[],
        )


def test_rejects_empty_context():
    metric = SemanticFaithfulness(FakeEmbedder({}))

    with pytest.raises(ValueError, match="Context cannot be empty"):
        metric.calculate(
            answer="Some answer",
            contexts=[""],
        )