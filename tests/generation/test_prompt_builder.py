from app.generation.prompt_builder import PromptBuilder


def test_prompt_builder_builds_prompt() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        query="What is FastAPI?",
        context=(
            "[Source: fastapi.md]\n"
            "FastAPI is a Python web framework."
        ),
    )

    assert prompt == (
        "Answer the question using only "
        "the provided context.\n\n"
        "Context:\n"
        "[Source: fastapi.md]\n"
        "FastAPI is a Python web framework.\n\n"
        "Question:\n"
        "What is FastAPI?\n\n"
        "Answer:"
    )


def test_prompt_builder_handles_empty_context() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        query="What is FastAPI?",
        context="",
    )

    assert prompt == (
        "Answer the question using only "
        "the provided context.\n\n"
        "Context:\n\n\n"
        "Question:\n"
        "What is FastAPI?\n\n"
        "Answer:"
    )