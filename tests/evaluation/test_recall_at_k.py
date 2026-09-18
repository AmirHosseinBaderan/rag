import pytest

from app.evaluation.metrics.recall_at_k import RecallAtK


def test_calculates_recall():
    metric = RecallAtK()

    score = metric.calculate(
        relevant_ids=["a", "c"],
        retrieved_ids=["b", "c", "a", "d"],
        k=4,
    )

    assert score == 1.0


def test_calculates_partial_recall():
    metric = RecallAtK()

    score = metric.calculate(
        relevant_ids=["a", "c", "e"],
        retrieved_ids=["b", "c", "a", "d"],
        k=4,
    )

    assert score == pytest.approx(2 / 3)


def test_calculates_zero_recall():
    metric = RecallAtK()

    score = metric.calculate(
        relevant_ids=["a", "c"],
        retrieved_ids=["b", "d"],
        k=2,
    )

    assert score == 0.0


def test_uses_only_top_k_results():
    metric = RecallAtK()

    score = metric.calculate(
        relevant_ids=["a"],
        retrieved_ids=["b", "c", "a"],
        k=2,
    )

    assert score == 0.0


def test_rejects_empty_relevant_ids():
    metric = RecallAtK()

    with pytest.raises(
        ValueError,
        match="Relevant IDs cannot be empty",
    ):
        metric.calculate(
            relevant_ids=[],
            retrieved_ids=["a"],
            k=1,
        )


def test_rejects_invalid_k():
    metric = RecallAtK()

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        metric.calculate(
            relevant_ids=["a"],
            retrieved_ids=["a"],
            k=0,
        )