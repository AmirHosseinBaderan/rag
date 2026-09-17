import httpx

from app.generation.llm import LLM


class OllamaLLM(LLM):
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120.0,
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]