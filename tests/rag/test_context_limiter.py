import pytest

from app.rag.context_limiter import ContextLimiter


def create_result(
    result_id: str,
    text: str,
) -> dict:
    return {
        "id": result_id,
        "score": 1.0,
        "metadata": {
            "source": f"{result_id}.txt",
            "text": text,
        },
    }


def test_keeps_results_until_character_limit_is_reached():
    limiter = ContextLimiter(max_characters=20)

    results = [
        create_result("a", "1234567890"),
        create_result("b", "abcdefghij"),
        create_result("c", "extra"),
    ]

    limited = limiter.limit(results)

    assert [result["id"] for result in limited] == ["a", "b"]


def test_keeps_result_that_fits_exactly():
    limiter = ContextLimiter(max_characters=10)

    results = [
        create_result("a", "1234567890"),
        create_result("b", "extra"),
    ]

    limited = limiter.limit(results)

    assert [result["id"] for result in limited] == ["a"]


def test_preserves_result_order():
    limiter = ContextLimiter(max_characters=30)

    results = [
        create_result("a", "first"),
        create_result("b", "second"),
        create_result("c", "third"),
    ]

    limited = limiter.limit(results)

    assert [result["id"] for result in limited] == [
        "a",
        "b",
        "c",
    ]


def test_returns_empty_list_for_empty_results():
    limiter = ContextLimiter(max_characters=100)

    assert limiter.limit([]) == []


def test_rejects_non_positive_limit():
    with pytest.raises(
        ValueError,
        match="max_characters must be greater than zero",
    ):
        ContextLimiter(max_characters=0)


def test_rejects_result_without_text():
    limiter = ContextLimiter(max_characters=100)

    result = {
        "id": "a",
        "score": 1.0,
        "metadata": {},
    }

    with pytest.raises(
        ValueError,
        match="Result metadata must contain text",
    ):
        limiter.limit([result])