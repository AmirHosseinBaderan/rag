from app.evaluation.dataset import (
    EvaluationCase,
    EvaluationDataset,
)
from app.evaluation.exact_match import ExactMatch
from app.evaluation.report import EvaluationReport
from app.evaluation.runner import EvaluationRunner


class FakeRAG:
    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    def ask(self, query: str) -> str:
        return self._answers[query]


def test_runner_evaluates_all_cases() -> None:
    dataset = EvaluationDataset(
        cases=[
            EvaluationCase(
                question="What is FastAPI?",
                expected_answer="FastAPI is a Python web framework.",
            ),
            EvaluationCase(
                question="What is RAG?",
                expected_answer="RAG retrieves relevant context.",
            ),
        ]
    )

    rag = FakeRAG(
        answers={
            "What is FastAPI?":
                "FastAPI is a Python web framework.",
            "What is RAG?":
                "RAG retrieves relevant context.",
        }
    )

    runner = EvaluationRunner(
        rag=rag,
        evaluator=ExactMatch(),
    )

    report = runner.run(dataset)

    assert isinstance(report, EvaluationReport)
    assert len(report.results) == 2
    assert report.results[0].score == 1.0
    assert report.results[1].score == 1.0


def test_runner_returns_zero_for_wrong_answer() -> None:
    dataset = EvaluationDataset(
        cases=[
            EvaluationCase(
                question="What is FastAPI?",
                expected_answer="FastAPI is a Python web framework.",
            ),
        ]
    )

    rag = FakeRAG(
        answers={
            "What is FastAPI?":
                "FastAPI is a database.",
        }
    )

    runner = EvaluationRunner(
        rag=rag,
        evaluator=ExactMatch(),
    )

    report = runner.run(dataset)

    assert len(report.results) == 1
    assert report.results[0].score == 0.0


def test_runner_calculates_average_score() -> None:
    dataset = EvaluationDataset(
        cases=[
            EvaluationCase(
                question="Q1",
                expected_answer="A1",
            ),
            EvaluationCase(
                question="Q2",
                expected_answer="A2",
            ),
            EvaluationCase(
                question="Q3",
                expected_answer="A3",
            ),
        ]
    )

    rag = FakeRAG(
        answers={
            "Q1": "A1",
            "Q2": "wrong",
            "Q3": "A3",
        }
    )

    runner = EvaluationRunner(
        rag=rag,
        evaluator=ExactMatch(),
    )

    report = runner.run(dataset)

    assert report.average_score == 2 / 3