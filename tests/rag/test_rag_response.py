from app.rag.rag_response import RAGResponse, RAGSource


def test_rag_source_contains_source_information():
    source = RAGSource(
        id="doc-1-chunk-0",
        document_id="doc-1",
        source="python.txt",
        score=0.92,
    )

    assert source.id == "doc-1-chunk-0"
    assert source.document_id == "doc-1"
    assert source.source == "python.txt"
    assert source.score == 0.92


def test_rag_response_contains_answer_and_sources():
    source = RAGSource(
        id="doc-1-chunk-0",
        document_id="doc-1",
        source="python.txt",
        score=0.92,
    )

    response = RAGResponse(
        answer="Python is a programming language.",
        sources=[source],
    )

    assert response.answer == "Python is a programming language."
    assert len(response.sources) == 1
    assert response.sources[0] == source