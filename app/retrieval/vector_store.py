from abc import ABC,abstractmethod
from typing import Any

class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        vector_id,
        vector:list[float],
        metadata:dict[str,Any]
    ):
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        vector:list[float],
        top_k:int
    )-> list[dict[str,Any]]:
        raise NotImplementedError