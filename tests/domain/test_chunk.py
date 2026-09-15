from app.domain.chunks.chunk import Chunk


def test_chunk_creation() -> None:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        text="FastAPI is a Python framework.",
    )

    assert chunk.id == "chunk-1"
    assert chunk.document_id == "doc-1"
    assert chunk.text == "FastAPI is a Python framework."
    assert chunk.metadata == {}


def test_chunk_accepts_metadata() -> None:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        text="FastAPI is a Python framework.",
        metadata={
            "source": "fastapi.md",
            "chunk_index": 0,
        },
    )

    assert chunk.metadata["source"] == "fastapi.md"
    assert chunk.metadata["chunk_index"] == 0