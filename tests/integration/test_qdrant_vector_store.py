from uuid import uuid4

from app.retrieval.qdrant_vector_store import QdrantVectorStore


def test_qdrant_can_upsert_and_search() -> None:
    collection_name = f"rag-test-{uuid4()}"

    store = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=768,
        host="192.168.0.247",
        port=6333,
    )

    vector = [0.0] * 768
    vector[0] = 1.0

    chunk_id = "doc-1-chunk-0"

    store.upsert(
        vector_id=chunk_id,
        vector=vector,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc-1",
            "source": "test.md",
            "text": "FastAPI is a Python framework.",
        },
    )

    results = store.search(
        vector=vector,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["chunk_id"] == chunk_id
    assert results[0]["metadata"]["document_id"] == "doc-1"
    assert results[0]["metadata"]["source"] == "test.md"