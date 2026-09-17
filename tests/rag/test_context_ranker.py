from app.rag.context_ranker import ContextRanker


def test_context_ranker_sorts_results_by_score():
    ranker = ContextRanker()

    results = [
        {
            "id": "chunk-1",
            "score": 0.4,
            "metadata": {"text": "Docker content"},
        },
        {
            "id": "chunk-2",
            "score": 0.9,
            "metadata": {"text": "FastAPI content"},
        },
        {
            "id": "chunk-3",
            "score": 0.7,
            "metadata": {"text": "Qdrant content"},
        },
    ]

    ranked = ranker.rank(results)

    assert [result["id"] for result in ranked] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]


def test_context_ranker_limits_results():
    ranker = ContextRanker()

    results = [
        {"id": "chunk-1", "score": 0.4, "metadata": {}},
        {"id": "chunk-2", "score": 0.9, "metadata": {}},
        {"id": "chunk-3", "score": 0.7, "metadata": {}},
    ]

    ranked = ranker.rank(results, top_n=2)

    assert len(ranked) == 2
    assert [result["id"] for result in ranked] == [
        "chunk-2",
        "chunk-3",
    ]


def test_context_ranker_returns_empty_for_empty_results():
    ranker = ContextRanker()

    assert ranker.rank([]) == []


def test_context_ranker_rejects_invalid_top_n():
    ranker = ContextRanker()

    results = [
        {"id": "chunk-1", "score": 0.5, "metadata": {}},
    ]

    try:
        ranker.rank(results, top_n=0)
        assert False
    except ValueError:
        pass