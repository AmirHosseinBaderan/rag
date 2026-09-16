import httpx

from app.retrieval.ollama_embedder import OllamaEmbedder


def create_response(
    status_code: int,
    json: dict,
) -> httpx.Response:
    request = httpx.Request(
        "POST",
        "http://127.0.0.1:11434/api/embed",
    )

    return httpx.Response(
        status_code,
        json=json,
        request=request,
    )


def test_ollama_embedder_returns_embedding(
    monkeypatch,
) -> None:
    def mock_post(*args, **kwargs):
        return create_response(
            200,
            {
                "embeddings": [
                    [0.1, 0.2, 0.3]
                ]
            },
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
    )

    vector = embedder.embed(
        "FastAPI is a Python framework."
    )

    assert vector == [0.1, 0.2, 0.3]


def test_ollama_embedder_uses_configured_model(
    monkeypatch,
) -> None:
    captured = {}

    def mock_post(*args, **kwargs):
        captured["url"] = args[0]
        captured["json"] = kwargs["json"]

        return create_response(
            200,
            {
                "embeddings": [
                    [0.1, 0.2, 0.3]
                ]
            },
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
        base_url="http://192.168.0.247:11434",
    )

    embedder.embed("hello")

    assert captured["url"] == (
        "http://192.168.0.247:11434/api/embed"
    )

    assert captured["json"] == {
        "model": "nomic-embed-text:latest",
        "input": "hello",
    }


def test_ollama_embedder_raises_for_http_error(
    monkeypatch,
) -> None:
    def mock_post(*args, **kwargs):
        return create_response(
            500,
            {"error": "internal server error"},
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
    )

    try:
        embedder.embed("hello")
        assert False
    except httpx.HTTPStatusError:
        pass