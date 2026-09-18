from app.evaluation.result import EvaluationResult


def test_evaluation_result_stores_values() -> None:
    result = EvaluationResult(
        question="What is FastAPI?",
        expected_answer="FastAPI is a Python web framework.",
        actual_answer="FastAPI is a Python web framework.",
        score=1.0,
    )

    assert result.question == "What is FastAPI?"
    assert result.expected_answer == (
        "FastAPI is a Python web framework."
    )
    assert result.actual_answer == (
        "FastAPI is a Python web framework."
    )
    assert result.score == 1.0