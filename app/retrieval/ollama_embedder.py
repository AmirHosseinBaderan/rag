import httpx
from app.retrieval.embedder import Embedder

class OllamaEmbedder(Embedder):
    def __init__(
            self,
            model:str,
            base_url:str="localhost:11434"
    ):
        self._model = model
        self._base_url = base_url

    def embed(self, text)-> list[float]:
        response = httpx.post(
            f"{self._base_url}/api/embed",
            json={
                "model":self._model,
                "input":text
            },
            timeout=60.0
        )

        response.raise_for_status()

        data = response.json()

        return data["embeddings"][0]