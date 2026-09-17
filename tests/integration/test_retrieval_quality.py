import pytest

from app.domain.documents.document import Document
from app.ingestion.chunker import Chunker
from app.retrieval.ollama_embedder import OllamaEmbedder
from app.retrieval.qdrant_vector_store import QdrantVectorStore
from app.retrieval.retriever import Retriever
from app.rag.document_indexer import DocumentIndexer


@pytest.mark.integration
def test_retrieval_returns_results_in_score_order():
    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
        base_url="http://192.168.0.247:11434",
    )

    vector_store = QdrantVectorStore(
        collection_name="test-retrieval-quality",
        vector_size=768,
        host="192.168.0.247",
        port=6333,
    )

    chunker = Chunker(
        chunk_size=200,
        overlap=0,
    )

    indexer = DocumentIndexer(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    document = Document(
        id="retrieval-quality-doc",
        content=(
            "FastAPI is a Python web framework for building APIs. "
            "It uses Python type hints for request validation and "
            "automatic API documentation. "
            "Qdrant is a vector database designed for similarity search. "
            "Docker packages applications into portable containers."
        ),
    )

    indexer.index(document)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="How does FastAPI validate request data?",
        top_k=3,
    )

    assert 0 < len(results) <= 3

    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)

    for result in results:
        assert "score" in result
        assert "metadata" in result
        assert "text" in result["metadata"]

@pytest.mark.integration
def test_top_k_controls_result_count_without_changing_best_match():
    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
        base_url="http://192.168.0.247:11434",
    )

    vector_store = QdrantVectorStore(
        collection_name="test-top-k",
        vector_size=768,
        host="192.168.0.247",
        port=6333,
    )

    chunker = Chunker(
        chunk_size=120,
        overlap=0,
    )

    indexer = DocumentIndexer(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    document = Document(
        id="top-k-doc",
        content=(
            "FastAPI is a modern Python web framework for building APIs. "
            "It uses Python type hints for request validation. "
            "Qdrant is a vector database for similarity search. "
            "Docker packages applications into portable containers. "
            "Redis is an in-memory data store used for caching."
        ),
    )

    indexer.index(document)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    query = "How does FastAPI validate request data?"

    results_1 = retriever.retrieve(
        query=query,
        top_k=1,
    )

    results_2 = retriever.retrieve(
        query=query,
        top_k=2,
    )

    results_3 = retriever.retrieve(
        query=query,
        top_k=3,
    )

    assert len(results_1) == 1
    assert len(results_2) == 2
    assert len(results_3) == 3

    assert results_1[0]["score"] == results_2[0]["score"]
    assert results_1[0]["score"] == results_3[0]["score"]

    scores = [result["score"] for result in results_3]

    assert scores == sorted(scores, reverse=True)