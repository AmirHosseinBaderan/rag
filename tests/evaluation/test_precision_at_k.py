import pytest

from app.evaluation.metrics.precision_at_k import PrecisionAtK


def test_calculates_precision():
    metric = PrecisionAtK()

    score = metric.calculate(
        relevant_ids=["a", "c"],
        retrieved_ids=["b", "c", "a", "d"],
        k=4,
    )

    assert score == 0.5


def test_calculates_full_precision():
    metric = PrecisionAtK()

    score = metric.calculate(
        relevant_ids=["a", "b", "c"],
        retrieved_ids=["a", "b", "c"],
        k=3,
    )

    assert score == 1.0


def test_calculates_zero_precision():
    metric = PrecisionAtK()

    score = metric.calculate(
        relevant_ids=["a", "c"],
        retrieved_ids=["b", "d"],
        k=2,
    )

    assert score == 0.0


def test_uses_only_top_k_results():
    metric = PrecisionAtK()

    score = metric.calculate(
        relevant_ids=["a"],
        retrieved_ids=["b", "c", "a"],
        k=2,
    )

    assert score == 0.0


def test_handles_fewer_results_than_k():
    metric = PrecisionAtK()

    score = metric.calculate(
        relevant_ids=["a", "b"],
        retrieved_ids=["a"],
        k=3,
    )

    assert score == pytest.approx(1 / 3)


def test_rejects_empty_relevant_ids():
    metric = PrecisionAtK()

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
    metric = PrecisionAtK()

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        metric.calculate(
            relevant_ids=["a"],
            retrieved_ids=["a"],
            k=0,
        )