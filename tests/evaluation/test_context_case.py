import pytest

from app.evaluation.context_case import ContextEvaluationCase


def test_context_evaluation_case_stores_values() -> None:
    case = ContextEvaluationCase(
        question="What is FastAPI?",
        contexts=[
            "FastAPI is a Python web framework.",
            "FastAPI supports automatic API documentation.",
        ],
    )

    assert case.question == "What is FastAPI?"
    assert len(case.contexts) == 2


def test_context_evaluation_case_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        ContextEvaluationCase(
            question="",
            contexts=["Some context."],
        )


def test_context_evaluation_case_rejects_empty_contexts() -> None:
    with pytest.raises(ValueError):
        ContextEvaluationCase(
            question="What is FastAPI?",
            contexts=[],
        )


def test_context_evaluation_case_rejects_empty_context() -> None:
    with pytest.raises(ValueError):
        ContextEvaluationCase(
            question="What is FastAPI?",
            contexts=[""],
        )