import pytest

from app.evaluation.metrics.mrr import MRR


def test_calculates_mrr():
    metric = MRR()

    score = metric.calculate(
        relevant_ids=[
            ["a"],
            ["b"],
            ["c"],
        ],
        retrieved_ids=[
            ["a", "x", "y"],
            ["x", "b", "y"],
            ["x", "y", "c"],
        ],
    )

    assert score == pytest.approx(
        (1.0 + 0.5 + 1 / 3) / 3
    )


def test_assigns_zero_to_queries_without_relevant_result():
    metric = MRR()

    score = metric.calculate(
        relevant_ids=[["a"], ["b"], ["c"]],
        retrieved_ids=[["a", "x"], ["x", "y"], ["c", "z"]],
    )

    assert score == pytest.approx((1.0 + 0.0 + 1.0) / 3)


def test_returns_zero_when_no_relevant_result_exists():
    metric = MRR()

    score = metric.calculate(
        relevant_ids=[
            ["a"],
            ["b"],
        ],
        retrieved_ids=[
            ["x", "y"],
            ["x", "y"],
        ],
    )

    assert score == 0.0


def test_multiple_relevant_results_uses_first_one():
    metric = MRR()

    score = metric.calculate(
        relevant_ids=[["a", "b"]],
        retrieved_ids=[["x", "b", "a"]],
    )

    assert score == 0.5


def test_rejects_empty_evaluation():
    metric = MRR()

    with pytest.raises(
        ValueError,
        match="Evaluation data cannot be empty",
    ):
        metric.calculate(
            relevant_ids=[],
            retrieved_ids=[],
        )


def test_rejects_mismatched_lengths():
    metric = MRR()

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        metric.calculate(
            relevant_ids=[["a"]],
            retrieved_ids=[
                ["a"],
                ["b"],
            ],
        )