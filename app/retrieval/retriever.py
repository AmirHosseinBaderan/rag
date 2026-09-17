from app.retrieval.embedder import Embedder
from  app.retrieval.vector_store import VectorStore

class Retriever:
    def __init__(self,
                 embedder:Embedder,
                 vector_store:VectorStore):
        self._embedder = embedder
        self._vector_store = vector_store

    def retriever(
            self,
            query:str,
            top_k:int
    )-> list[dict]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        vector = self._embedder.embed(query)
        return self._vector_store.search(
            vector=vector,
            top_k=top_k
        )
        