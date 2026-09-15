from app.retrieval.local_embedder import LocalEmbedder


def test_local_embedder_returns_vector() -> None:
    embedder = LocalEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vector = embedder.embed(
        "FastAPI is a Python framework."
    )

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)