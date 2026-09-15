from app.domain.documents.document import Document
from app.ingestion.chunker import Chunker


def test_short_document_creates_one_chunk() -> None:
    document = Document(
        id="doc-1",
        content="Hello world",
    )

    chunker = Chunker(chunk_size=100)

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].text == "Hello world"


def test_long_document_creates_multiple_chunks() -> None:
    document = Document(
        id="doc-1",
        content="abcdefghij",
    )

    chunker = Chunker(chunk_size=4)

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    assert chunks[0].text == "abcd"
    assert chunks[1].text == "efgh"
    assert chunks[2].text == "ij"


def test_chunk_ids_are_unique() -> None:
    document = Document(
        id="doc-1",
        content="abcdefghij",
    )

    chunker = Chunker(chunk_size=4)

    chunks = chunker.chunk(document)

    ids = [chunk.id for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_document_metadata_is_preserved() -> None:
    document = Document(
        id="doc-1",
        content="abcdefghij",
        metadata={
            "source": "test.md",
            "category": "backend",
        },
    )

    chunker = Chunker(chunk_size=4)

    chunks = chunker.chunk(document)

    assert chunks[0].metadata["source"] == "test.md"
    assert chunks[0].metadata["category"] == "backend"


def test_chunk_overlap() -> None:
    document = Document(
        id="doc-1",
        content="abcdefghij",
    )

    chunker = Chunker(
        chunk_size=4,
        overlap=2,
    )

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "abcd",
        "cdef",
        "efgh",
        "ghij",
        "ij",
    ]


def test_zero_overlap_behaves_like_simple_chunking() -> None:
    document = Document(
        id="doc-1",
        content="abcdefghij",
    )

    chunker = Chunker(
        chunk_size=4,
        overlap=0,
    )

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "abcd",
        "efgh",
        "ij",
    ]


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    try:
        Chunker(
            chunk_size=4,
            overlap=4,
        )
        assert False
    except ValueError:
        pass