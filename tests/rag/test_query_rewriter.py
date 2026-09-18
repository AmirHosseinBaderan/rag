from app.rag.query_rewriter import QueryRewriter


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "rewritten query"


def test_rewriter_returns_rewritten_query():
    rewriter = QueryRewriter(FakeLLM())

    result = rewriter.rewrite("original query")

    assert result == "rewritten query"


def test_rewriter_rejects_empty_query():
    rewriter = QueryRewriter(FakeLLM())

    try:
        rewriter.rewrite(" ")
        assert False
    except ValueError as error:
        assert str(error) == "Query cannot be empty"