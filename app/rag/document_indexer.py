from app.domain.documents.document import Document
from app.ingestion.chunker import Chunker
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore


class DocumentIndexer:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    def index(
        self,
        document: Document,
    ) -> None:
        chunks = self._chunker.chunk(document)

        for chunk in chunks:
            vector = self._embedder.embed(
                chunk.text
            )

            metadata = {
                **chunk.metadata,
                "document_id": chunk.document_id,
                "text": chunk.text,
            }

            self._vector_store.upsert(
                vector_id=chunk.id,
                vector=vector,
                metadata=metadata,
            )