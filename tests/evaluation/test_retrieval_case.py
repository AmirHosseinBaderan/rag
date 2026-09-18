import pytest

from app.evaluation.retrieval_case import RetrievalCase


def test_creates_retrieval_case():
    case = RetrievalCase(
        query="What is RAG?",
        relevant_ids=["doc-1", "doc-3"],
    )

    assert case.query == "What is RAG?"
    assert case.relevant_ids == ["doc-1", "doc-3"]


def test_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        RetrievalCase(
            query="",
            relevant_ids=["doc-1"],
        )


def test_rejects_empty_relevant_ids():
    with pytest.raises(
        ValueError,
        match="Relevant IDs cannot be empty",
    ):
        RetrievalCase(
            query="What is RAG?",
            relevant_ids=[],
        )