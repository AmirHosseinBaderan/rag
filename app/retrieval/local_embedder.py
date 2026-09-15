from sentence_transformers import SentenceTransformer

from app.retrieval.embedder import Embedder


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()