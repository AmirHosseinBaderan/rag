from app.retrieval.keyword_search import KeywordSearch


def test_keyword_search_ranks_matching_documents_higher():
    search = KeywordSearch()

    documents = [
        {"id": "1", "text": "FastAPI is a Python web framework."},
        {"id": "2", "text": "PostgreSQL is a relational database."},
        {"id": "3", "text": "FastAPI supports automatic API documentation."},
    ]

    results = search.search(
        query="FastAPI Python",
        documents=documents,
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["id"] == "1"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["score"] > results[2]["score"]


def test_keyword_search_returns_only_top_k_results():
    search = KeywordSearch()

    documents = [
        {"id": "1", "text": "FastAPI Python framework"},
        {"id": "2", "text": "FastAPI API framework"},
        {"id": "3", "text": "FastAPI documentation"},
    ]

    results = search.search(
        query="FastAPI",
        documents=documents,
        top_k=2,
    )

    assert len(results) == 2


def test_keyword_search_rejects_empty_query():
    search = KeywordSearch()

    documents = [
        {"id": "1", "text": "FastAPI Python framework"},
    ]

    try:
        search.search("", documents, top_k=1)
        assert False
    except ValueError:
        pass