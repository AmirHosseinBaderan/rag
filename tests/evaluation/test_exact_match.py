from app.evaluation.exact_match import ExactMatch


def test_exact_match_returns_one_for_equal_answers() -> None:
    evaluator = ExactMatch()

    score = evaluator.evaluate(
        expected="FastAPI is a Python web framework.",
        actual="FastAPI is a Python web framework.",
    )

    assert score == 1.0


def test_exact_match_returns_zero_for_different_answers() -> None:
    evaluator = ExactMatch()

    score = evaluator.evaluate(
        expected="FastAPI is a Python web framework.",
        actual="FastAPI is a web framework for Python.",
    )

    assert score == 0.0


def test_exact_match_ignores_surrounding_whitespace() -> None:
    evaluator = ExactMatch()

    score = evaluator.evaluate(
        expected="FastAPI is a Python web framework.",
        actual="  FastAPI is a Python web framework.  ",
    )

    assert score == 1.0