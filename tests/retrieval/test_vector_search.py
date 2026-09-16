from typing import Any

from app.retrieval.vector_store import VectorStore


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        self.items[vector_id] = {
            "vector": vector,
            "metadata": metadata,
        }

    def search(
        self,
        vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        return list(self.items.values())[:top_k]


def test_vector_store_can_upsert() -> None:
    store = FakeVectorStore()

    store.upsert(
        vector_id="chunk-1",
        vector=[0.1, 0.2, 0.3],
        metadata={
            "source": "fastapi.md",
        },
    )

    assert store.items["chunk-1"]["vector"] == [
        0.1,
        0.2,
        0.3,
    ]

    assert store.items["chunk-1"]["metadata"] == {
        "source": "fastapi.md",
    }


def test_vector_store_can_search() -> None:
    store = FakeVectorStore()

    store.upsert(
        vector_id="chunk-1",
        vector=[0.1, 0.2, 0.3],
        metadata={
            "source": "fastapi.md",
        },
    )

    results = store.search(
        vector=[0.1, 0.2, 0.3],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "fastapi.md"