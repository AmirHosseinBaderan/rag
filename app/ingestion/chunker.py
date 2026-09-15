from app.domain.chunks.chunk import Chunk
from app.domain.documents.document import Document


class Chunker:
    def __init__(
        self,
        chunk_size: int,
        overlap: int = 0,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(
                "Chunk size must be greater than zero"
            )

        if overlap < 0:
            raise ValueError(
                "Overlap cannot be negative"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "Overlap must be smaller than chunk size"
            )

        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []

        step = self._chunk_size - self._overlap

        for index, start in enumerate(
            range(0, len(document.content), step)
        ):
            text = document.content[
                start : start + self._chunk_size
            ]

            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{index}",
                    document_id=document.id,
                    text=text,
                    metadata=document.metadata.copy(),
                )
            )

        return chunks