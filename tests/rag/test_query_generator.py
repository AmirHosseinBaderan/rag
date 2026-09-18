import pytest

from app.rag.query_generator import QueryGenerator


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return (
            "FastAPI dependency injection\n"
            "FastAPI Depends mechanism\n"
            "dependency injection in FastAPI"
        )


def test_query_generator_returns_multiple_queries():
    generator = QueryGenerator(FakeLLM())

    result = generator.generate(
        "How does dependency injection work in FastAPI?"
    )

    assert result == [
        "FastAPI dependency injection",
        "FastAPI Depends mechanism",
        "dependency injection in FastAPI",
    ]


def test_query_generator_ignores_empty_lines():
    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return (
                "query one\n"
                "\n"
                "query two\n"
                "   \n"
            )

    generator = QueryGenerator(FakeLLM())

    result = generator.generate("test")

    assert result == [
        "query one",
        "query two",
    ]


def test_query_generator_rejects_empty_query():
    generator = QueryGenerator(FakeLLM())

    with pytest.raises(ValueError, match="Query cannot be empty"):
        generator.generate(" ")


def test_query_generator_rejects_empty_llm_response():
    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return " \n "

    generator = QueryGenerator(FakeLLM())

    with pytest.raises(ValueError, match="Generated queries cannot be empty"):
        generator.generate("test")