from app.retrieval.hybrid_search import HybridSearch


def test_hybrid_search_combines_semantic_and_keyword_scores():
    search = HybridSearch()

    semantic_results = [
        {"id": "1", "score": 0.90},
        {"id": "2", "score": 0.80},
    ]

    keyword_results = [
        {"id": "1", "score": 0.50},
        {"id": "3", "score": 1.00},
    ]

    results = search.combine(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=3,
    )

    assert len(results) == 3

    result_by_id = {result["id"]: result for result in results}

    assert "1" in result_by_id
    assert "2" in result_by_id
    assert "3" in result_by_id

    assert result_by_id["1"]["score"] == 0.74
    assert result_by_id["3"]["score"] == 0.40
    assert result_by_id["2"]["score"] == 0.48


def test_hybrid_search_ranks_by_combined_score():
    search = HybridSearch()

    semantic_results = [
        {"id": "1", "score": 0.90},
        {"id": "2", "score": 0.80},
    ]

    keyword_results = [
        {"id": "1", "score": 0.50},
        {"id": "3", "score": 1.00},
    ]

    results = search.combine(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=3,
    )

    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)