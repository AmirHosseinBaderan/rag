import httpx

from app.generation.ollama_llm import OllamaLLM


def create_response(
    status_code: int,
    json: dict,
) -> httpx.Response:
    request = httpx.Request(
        "POST",
        "http://192.168.0.247:11434/api/generate",
    )

    return httpx.Response(
        status_code,
        json=json,
        request=request,
    )


def test_ollama_llm_returns_response(
    monkeypatch,
) -> None:
    def mock_post(*args, **kwargs):
        return create_response(
            200,
            {
                "response": "FastAPI is a Python framework."
            },
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    llm = OllamaLLM(
        model="your-model",
    )

    answer = llm.generate(
        "What is FastAPI?"
    )

    assert answer == (
        "FastAPI is a Python framework."
    )


def test_ollama_llm_sends_correct_request(
    monkeypatch,
) -> None:
    captured = {}

    def mock_post(*args, **kwargs):
        captured["url"] = args[0]
        captured["json"] = kwargs["json"]

        return create_response(
            200,
            {
                "response": "answer"
            },
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    llm = OllamaLLM(
        model="llama3.2",
        base_url="http://192.168.0.247:11434",
    )

    llm.generate("What is RAG?")

    assert captured["url"] == (
        "http://192.168.0.247:11434/api/generate"
    )

    assert captured["json"] == {
        "model": "llama3.2",
        "prompt": "What is RAG?",
        "stream": False,
    }


def test_ollama_llm_raises_for_http_error(
    monkeypatch,
) -> None:
    def mock_post(*args, **kwargs):
        return create_response(
            500,
            {
                "error": "internal server error"
            },
        )

    monkeypatch.setattr(
        httpx,
        "post",
        mock_post,
    )

    llm = OllamaLLM(
        model="llama3.2",
    )

    try:
        llm.generate("hello")
        assert False
    except httpx.HTTPStatusError:
        pass