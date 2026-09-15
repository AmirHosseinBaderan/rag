from app.retrieval.embedder import Embedder


class FakeEmbedder(Embedder):
    def embed(self, text: str) -> list[float]:
        return [float(len(text))]


def test_embedder_returns_vector() -> None:
    embedder = FakeEmbedder()

    vector = embedder.embed("hello")

    assert isinstance(vector, list)
    assert vector == [5.0]