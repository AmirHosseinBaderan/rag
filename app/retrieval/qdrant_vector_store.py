from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

from app.retrieval.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        host: str = "192.168.0.247",
        port: int = 6333,
    ) -> None:
        self._collection_name = collection_name

        self._client = QdrantClient(
            host=host,
            port=port,
        )

        self._ensure_collection(vector_size)

    def _ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        collections = self._client.get_collections()

        exists = any(
            collection.name == self._collection_name
            for collection in collections.collections
        )

        if not exists:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance="Cosine",
                ),
            )

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        payload = {
            **metadata,
            "vector_id": vector_id,
        }

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def search(
        self,
        vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=top_k,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "metadata": point.payload or {},
            }
            for point in results.points
        ]