from uuid import uuid4

from app.retrieval.ollama_embedder import OllamaEmbedder
from app.retrieval.qdrant_vector_store import QdrantVectorStore
from app.retrieval.retriever import Retriever


def test_retriever_returns_matching_chunk() -> None:
    collection_name = f"rag-retriever-{uuid4()}"

    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
        base_url="http://192.168.0.247:11434",
    )

    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=768,
        host="192.168.0.247",
        port=6333,
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    documents = [
        (
            "chunk-fastapi",
            "FastAPI is a modern Python web framework "
            "for building APIs.",
        ),
        (
            "chunk-database",
            "PostgreSQL is a relational database system "
            "used for storing structured data.",
        ),
    ]

    for chunk_id, text in documents:
        vector = embedder.embed(text)

        vector_store.upsert(
            vector_id=chunk_id,
            vector=vector,
            metadata={
                "chunk_id": chunk_id,
                "document_id": "doc-1",
                "text": text,
            },
        )

    results = retriever.retrieve(
        query=documents[0][1],
        top_k=1,
    )

    assert len(results) == 1

    assert (
        results[0]["metadata"]["chunk_id"]
        == "chunk-fastapi"
    )

    assert (
        results[0]["metadata"]["text"]
        == documents[0][1]
    )