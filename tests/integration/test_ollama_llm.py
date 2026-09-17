import pytest

from app.generation.ollama_llm import OllamaLLM


@pytest.mark.integration
def test_ollama_generates_answer() -> None:
    llm = OllamaLLM(
        model="gemma3:1b",
        base_url="http://192.168.0.247:11434",
    )

    answer = llm.generate(
        "What is FastAPI? Answer in one short sentence."
    )

    assert isinstance(answer, str)
    assert answer.strip()