from app.evaluation.report import EvaluationReport
from app.evaluation.runner import EvaluationResult


def test_report_calculates_average_score() -> None:
    results = [
        EvaluationResult(
            question="Q1",
            expected_answer="A1",
            actual_answer="A1",
            score=1.0,
        ),
        EvaluationResult(
            question="Q2",
            expected_answer="A2",
            actual_answer="wrong",
            score=0.0,
        ),
        EvaluationResult(
            question="Q3",
            expected_answer="A3",
            actual_answer="A3",
            score=1.0,
        ),
    ]

    report = EvaluationReport(results)

    assert report.average_score == 2 / 3


def test_empty_report_has_zero_average_score() -> None:
    report = EvaluationReport([])

    assert report.average_score == 0.0