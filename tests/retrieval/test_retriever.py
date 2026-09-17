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
        self.last_score_threshold = None
        self.last_metadata_filter = None

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
        score_threshold:float|None = None,
        metadata_filter:dict|None = None
    ) -> list[dict]:
        self.received_vector = vector
        self.received_top_k = top_k
        self.last_score_threshold = score_threshold
        self.last_metadata_filter = metadata_filter

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

def test_retrieve_passes_score_threshold_to_vector_store():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    retriever.retrieve(
        query="test query",
        top_k=3,
        score_threshold=0.8,
    )

    assert vector_store.last_score_threshold == 0.8

def test_retrieve_passes_metadata_filter_to_vector_store():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    metadata_filter = {
        "category": "backend",
        "source": "fastapi.md",
    }

    retriever.retrieve(
        query="dependency injection",
        top_k=3,
        metadata_filter=metadata_filter,
    )

    assert vector_store.last_metadata_filter == metadata_filter