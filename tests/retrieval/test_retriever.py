from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever


class FakeEmbedder(Embedder):
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.received_vector = None
        self.received_top_k = None

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict,
    ) -> None:
        pass

    def search(
        self,
        vector: list[float],
        top_k: int,
    ) -> list[dict]:
        self.received_vector = vector
        self.received_top_k = top_k

        return [
            {
                "id": "chunk-1",
                "score": 0.95,
                "metadata": {
                    "document_id": "doc-1",
                    "text": "FastAPI is a Python framework.",
                },
            }
        ]


def test_retriever_embeds_query_and_searches() -> None:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="What is FastAPI?",
        top_k=1,
    )

    assert vector_store.received_vector == [
        1.0,
        0.0,
        0.0,
    ]

    assert vector_store.received_top_k == 1

    assert results == [
        {
            "id": "chunk-1",
            "score": 0.95,
            "metadata": {
                "document_id": "doc-1",
                "text": "FastAPI is a Python framework.",
            },
        }
    ]


def test_retriever_rejects_empty_query() -> None:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    try:
        retriever.retrieve(
            query="",
            top_k=1,
        )
        assert False
    except ValueError:
        pass


def test_retriever_rejects_invalid_top_k() -> None:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    try:
        retriever.retrieve(
            query="What is FastAPI?",
            top_k=0,
        )
        assert False
    except ValueError:
        pass