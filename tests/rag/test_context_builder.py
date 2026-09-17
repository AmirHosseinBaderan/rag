from app.rag.context_builder import ContextBuilder


def test_context_builder_builds_context() -> None:
    builder = ContextBuilder()

    results = [
        {
            "id": "chunk-1",
            "score": 0.95,
            "metadata": {
                "source": "fastapi.md",
                "text": "FastAPI is a Python web framework.",
            },
        },
        {
            "id": "chunk-2",
            "score": 0.82,
            "metadata": {
                "source": "python.md",
                "text": "Python is a programming language.",
            },
        },
    ]

    context = builder.build(results)

    assert context == (
        "[Source: fastapi.md]\n"
        "FastAPI is a Python web framework.\n\n"
        "[Source: python.md]\n"
        "Python is a programming language."
    )


def test_context_builder_handles_empty_results() -> None:
    builder = ContextBuilder()

    context = builder.build([])

    assert context == ""