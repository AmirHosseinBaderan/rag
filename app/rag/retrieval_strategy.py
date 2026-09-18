from abc import ABC, abstractmethod
from typing import Any


class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError