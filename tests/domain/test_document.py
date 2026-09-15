from app.domain.documents.document import Document


def test_document_creation() -> None:
    document = Document(
        id="doc-1",
        content="FastAPI is a Python framework.",
    )

    assert document.id == "doc-1"
    assert document.content == "FastAPI is a Python framework."
    assert document.metadata == {}


def test_document_accepts_metadata() -> None:
    document = Document(
        id="doc-1",
        content="FastAPI is a Python framework.",
        metadata={
            "source": "fastapi.md",
            "category": "backend",
        },
    )

    assert document.metadata["source"] == "fastapi.md"
    assert document.metadata["category"] == "backend"