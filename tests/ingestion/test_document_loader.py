from pathlib import Path

from app.ingestion.document_loader import DocumentLoader


def test_load_document(tmp_path: Path) -> None:
    file_path = tmp_path / "fastapi.md"
    file_path.write_text(
        "# FastAPI\n\nFastAPI is a Python framework.",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    document = loader.load(file_path)

    assert document.id == "fastapi"
    assert document.content == (
        "# FastAPI\n\nFastAPI is a Python framework."
    )
    assert document.metadata["source"] == "fastapi.md"


def test_load_missing_file_raises_error(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "missing.md"

    loader = DocumentLoader()

    try:
        loader.load(file_path)
        assert False
    except FileNotFoundError:
        pass