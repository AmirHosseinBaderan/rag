from app.evaluation.context_relevance import ContextRelevance


def test_relevant_context_returns_one() -> None:
    evaluator = ContextRelevance()

    score = evaluator.evaluate(
        question="What is FastAPI?",
        contexts=[
            "FastAPI is a Python web framework.",
        ],
    )

    assert score == 1.0


def test_irrelevant_context_returns_zero() -> None:
    evaluator = ContextRelevance()

    score = evaluator.evaluate(
        question="What is FastAPI?",
        contexts=[
            "Qdrant is a vector database.",
        ],
    )

    assert score == 0.0


def test_multiple_contexts_returns_relevance_ratio() -> None:
    evaluator = ContextRelevance()

    score = evaluator.evaluate(
        question="What is FastAPI?",
        contexts=[
            "FastAPI is a Python web framework.",
            "Qdrant is a vector database.",
        ],
    )

    assert score == 0.5


def test_empty_question_is_rejected() -> None:
    evaluator = ContextRelevance()

    try:
        evaluator.evaluate(
            question="",
            contexts=["Some context."],
        )
        assert False
    except ValueError:
        pass