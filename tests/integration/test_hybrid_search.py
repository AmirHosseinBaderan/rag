import pytest
from uuid import uuid4

from app.domain.documents.document import Document
from app.ingestion.chunker import Chunker
from app.retrieval.ollama_embedder import OllamaEmbedder
from app.retrieval.qdrant_vector_store import QdrantVectorStore
from app.retrieval.retriever import Retriever
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.hybrid_search import HybridSearch
from app.rag.document_indexer import DocumentIndexer


@pytest.mark.integration
def test_hybrid_search_combines_semantic_and_keyword_results():
    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
        base_url="http://192.168.0.247:11434",
    )

    vector_store = QdrantVectorStore(
        collection_name=f"test-hybrid-{uuid4()}",
        vector_size=768,
        host="192.168.0.247",
        port=6333,
    )

    documents = [
        Document(
            id="fastapi-doc",
            content=(
                "FastAPI is a Python web framework for building APIs. "
                "It uses Python type hints for request validation."
            ),
            metadata={"source": "fastapi.md"},
        ),
        Document(
            id="postgresql-doc",
            content=(
                "PostgreSQL is a relational database system "
                "used for storing structured data."
            ),
            metadata={"source": "postgresql.md"},
        ),
        Document(
            id="docker-doc",
            content=(
                "Docker packages applications into portable containers "
                "for deployment."
            ),
            metadata={"source": "docker.md"},
        ),
    ]

    chunker = Chunker(chunk_size=500, overlap=0)

    indexer = DocumentIndexer(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    for document in documents:
        indexer.index(document)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    query = "FastAPI Python API"

    semantic_results = retriever.retrieve(
        query=query,
        top_k=3,
    )

    keyword_search = KeywordSearch()

    chunks = []

    for document in documents:
        chunks.extend(chunker.chunk(document))

    keyword_results = keyword_search.search(
        query=query,
        documents=[
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
        top_k=3,
    )

    hybrid_search = HybridSearch()

    results = hybrid_search.combine(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=3,
    )

    assert len(results) == 3

    assert results[0]["id"] == "fastapi-doc-chunk-0"

    assert results[0]["semantic_score"] > 0
    assert results[0]["keyword_score"] > 0
    assert results[0]["score"] > 0