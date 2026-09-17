from app.domain.chunks.chunk import Chunk
from app.domain.documents.document import Document
from app.rag.document_indexer import DocumentIndexer
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self.received_texts = []

    def embed(self, text: str) -> list[float]:
        self.received_texts.append(text)

        return [1.0, 0.0, 0.0]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.items = []

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict,
    ) -> None:
        self.items.append(
            {
                "id": vector_id,
                "vector": vector,
                "metadata": metadata,
            }
        )

    def search(
        self,
        vector: list[float],
        top_k: int,
    ) -> list[dict]:
        return []


class FakeChunker:
    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        return [
            Chunk(
                id="doc-1-chunk-0",
                document_id=document.id,
                text="First chunk.",
                metadata=document.metadata,
            ),
            Chunk(
                id="doc-1-chunk-1",
                document_id=document.id,
                text="Second chunk.",
                metadata=document.metadata,
            ),
        ]


def test_document_indexer_indexes_chunks() -> None:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    chunker = FakeChunker()

    indexer = DocumentIndexer(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    document = Document(
        id="doc-1",
        content="Some document content.",
        metadata={
            "source": "test.md",
        },
    )

    indexer.index(document)

    assert embedder.received_texts == [
        "First chunk.",
        "Second chunk.",
    ]

    assert len(vector_store.items) == 2

    assert (
        vector_store.items[0]["id"]
        == "doc-1-chunk-0"
    )

    assert (
        vector_store.items[1]["id"]
        == "doc-1-chunk-1"
    )

    assert (
        vector_store.items[0]["metadata"]["document_id"]
        == "doc-1"
    )