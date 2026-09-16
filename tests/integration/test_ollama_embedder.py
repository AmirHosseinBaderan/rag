import pytest

from app.retrieval.ollama_embedder import OllamaEmbedder


@pytest.mark.integration
def test_ollama_returns_real_embedding() -> None:
    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
        base_url="http://192.168.0.247:11434",
    )

    vector = embedder.embed(
        "FastAPI is a Python framework."
    )

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(
        isinstance(value, float)
        for value in vector
    )