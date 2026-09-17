from app.retrieval.embedder import Embedder
from  app.retrieval.vector_store import VectorStore

class Retriever:
    def __init__(self,
                 embedder:Embedder,
                 vector_store:VectorStore):
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
            self,
            query:str,
            top_k:int,
            score_threshold: float | None = None,
            metadata_filter: dict | None = None,
    )-> list[dict]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError(
                "score_threshold must be between 0 and 1"
            )

        vector = self._embedder.embed(query)
        return self._vector_store.search(
            vector=vector,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter
        )
        