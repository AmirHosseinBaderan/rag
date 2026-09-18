from app.generation.llm import LLM
from app.rag.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever
from app.rag.rag import RAG
from app.generation.prompt_builder import PromptBuilder
from app.rag.context_ranker import ContextRanker


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        pass

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        return [
            {
                "id": "chunk-1",
                "score": 0.95,
                "metadata": {
                    "source": "fastapi.md",
                    "text": (
                        "FastAPI is a Python "
                        "web framework."
                    ),
                },
            }
        ]


class FakeLLM(LLM):
    def __init__(self) -> None:
        self.received_prompt = ""
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        self.last_prompt = prompt
        return "FastAPI is a Python web framework."

class RankingFakeRetriever(Retriever):
    def __init__(self) -> None:
        pass

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        return [
            {
                "id": "low",
                "score": 0.4,
                "metadata": {
                    "source": "low.md",
                    "text": "Low relevance context.",
                },
            },
            {
                "id": "high",
                "score": 0.9,
                "metadata": {
                    "source": "high.md",
                    "text": "High relevance context.",
                },
            },
        ]


def test_rag_generates_answer_from_retrieved_context() -> None:
    retriever = FakeRetriever()
    context_builder = ContextBuilder()
    llm = FakeLLM()
    prompt_builder = PromptBuilder()

    rag = RAG(
        retriever=retriever,
        context_ranker=ContextRanker(),
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    answer = rag.ask(
        "What is FastAPI?"
    )

    assert answer == (
        "FastAPI is a Python web framework."
    )

    assert (
        "[Source: fastapi.md]"
        in llm.received_prompt
    )

    assert (
        "FastAPI is a Python web framework."
        in llm.received_prompt
    )

    assert (
        "What is FastAPI?"
        in llm.received_prompt
    )

def test_rag_ranks_context_before_generation() -> None:
    llm = FakeLLM()
    retriever = RankingFakeRetriever()
    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()

    rag = RAG(
        retriever=retriever,
        context_ranker=ContextRanker(),
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    rag.ask(
        query="test",
        top_k=2,
    )

    assert llm.last_prompt

    assert llm.last_prompt.index(
        "High relevance context."
    ) < llm.last_prompt.index(
        "Low relevance context."
    )

def test_rag_rewrites_query_before_retrieval():
    class FakeRewriter:
        def rewrite(self, query: str) -> str:
            return "rewritten query"

    class CapturingRetriever:
        def __init__(self) -> None:
            self.last_query = None

        def retrieve(
            self,
            query: str,
            top_k: int,
            score_threshold=None,
            metadata_filter=None,
        ):
            self.last_query = query
            return [
                {
                    "id": "chunk-1",
                    "score": 1.0,
                    "metadata": {
                        "source": "test.txt",
                        "text": "context",
                    },
                }
            ]

    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return "answer"

    retriever = CapturingRetriever()

    rag = RAG(
        retriever=retriever,
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=FakeLLM(),
        query_rewriter=FakeRewriter(),
    )

    rag.ask("original query")

    assert retriever.last_query == "rewritten query"

def test_rag_uses_multi_query_retriever_when_provided():
    class FakeMultiQueryRetriever:
        def __init__(self) -> None:
            self.last_query = None

        def retrieve(self, query: str, top_k: int):
            self.last_query = query

            return [
                {
                    "id": "chunk-1",
                    "score": 1.0,
                    "metadata": {
                        "source": "test.txt",
                        "text": "multi query context",
                    },
                }
            ]

    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return "answer"

    multi_retriever = FakeMultiQueryRetriever()

    rag = RAG(
        retriever=multi_retriever,
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=FakeLLM(),
    )

    result = rag.ask(
        query="original question",
        top_k=3,
    )

    assert result == "answer"
    assert multi_retriever.last_query == "original question"

def test_rag_uses_reranker_before_building_context():
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int):
            return [
                {
                    "id": "a",
                    "score": 0.95,
                    "metadata": {
                        "source": "a.txt",
                        "text": "candidate-a",
                    },
                },
                {
                    "id": "b",
                    "score": 0.70,
                    "metadata": {
                        "source": "b.txt",
                        "text": "candidate-b",
                    },
                }
            ]

    class FakeReranker:
        def __init__(self):
            self.calls = []

        def rerank(self, query, results, top_k):
            self.calls.append((query, results, top_k))
            return [results[1]]

    class FakeLLM:
        def __init__(self):
            self.prompt = None

        def generate(self, prompt: str) -> str:
            self.prompt = prompt
            return "answer"

    retriever = FakeRetriever()
    reranker = FakeReranker()
    llm = FakeLLM()

    rag = RAG(
        retriever=retriever,
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=llm,
        reranker=reranker,
    )

    result = rag.ask(
        query="question",
        top_k=1,
        retrieval_k=2,
    )

    assert result == "answer"
    assert len(reranker.calls) == 1
    assert reranker.calls[0][0] == "question"
    assert len(reranker.calls[0][1]) == 2
    assert reranker.calls[0][2] == 2
    assert "candidate-b" in llm.prompt
    assert "candidate-a" not in llm.prompt

def test_rag_uses_reranker_and_context_compressor():
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int):
            return [
                {
                    "id": "a",
                    "score": 0.95,
                    "metadata": {
                        "source": "a.txt",
                        "text": "candidate-a",
                    },
                },
                {
                    "id": "b",
                    "score": 0.70,
                    "metadata": {
                        "source": "b.txt",
                        "text": "candidate-b",
                    },
                },
            ]

    class FakeReranker:
        def __init__(self):
            self.calls = []

        def rerank(self, query, results, top_k):
            self.calls.append((query, results, top_k))
            return results

    class FakeCompressor:
        def __init__(self):
            self.calls = []

        def compress(self, query, results, top_k):
            self.calls.append((query, results, top_k))
            return [results[1]]

    class FakeLLM:
        def __init__(self):
            self.prompt = None

        def generate(self, prompt: str) -> str:
            self.prompt = prompt
            return "answer"

    reranker = FakeReranker()
    compressor = FakeCompressor()
    llm = FakeLLM()

    rag = RAG(
        retriever=FakeRetriever(),
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=llm,
        reranker=reranker,
        context_compressor=compressor,
    )

    result = rag.ask(
        query="question",
        top_k=1,
        retrieval_k=2,
    )

    assert result == "answer"

    assert len(reranker.calls) == 1
    assert reranker.calls[0][2] == 2

    assert len(compressor.calls) == 1
    assert compressor.calls[0][2] == 1

    assert "candidate-b" in llm.prompt
    assert "candidate-a" not in llm.prompt

def test_rag_uses_multi_query_retriever():
    class FakeMultiQueryRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query: str, top_k: int):
            self.calls.append((query, top_k))

            return [
                {
                    "id": "a",
                    "score": 0.8,
                    "metadata": {
                        "source": "a.txt",
                        "text": "candidate-a",
                    },
                },
                {
                    "id": "b",
                    "score": 0.7,
                    "metadata": {
                        "source": "b.txt",
                        "text": "candidate-b",
                    },
                },
            ]

    class FakeReranker:
        def rerank(self, query, results, top_k):
            return results[:top_k]

    class FakeCompressor:
        def compress(self, query, results, top_k):
            return results[:top_k]

    class FakeLLM:
        def __init__(self):
            self.prompt = None

        def generate(self, prompt: str) -> str:
            self.prompt = prompt
            return "answer"

    multi_retriever = FakeMultiQueryRetriever()
    llm = FakeLLM()

    rag = RAG(
        retriever=multi_retriever,
        multi_query_retriever=multi_retriever,
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=llm,
        reranker=FakeReranker(),
        context_compressor=FakeCompressor(),
    )

    result = rag.ask(
        query="original question",
        top_k=1,
        retrieval_k=2,
    )

    assert result == "answer"
    assert multi_retriever.calls == [
        ("original question", 2),
    ]
    assert "candidate-a" in llm.prompt

def test_ask_with_sources_returns_answer_and_sources():
    class SourcesFakeRetriever:
        def retrieve(
            self,
            query: str,
            top_k: int,
        ) -> list[dict]:
            return [
                {
                    "id": "doc-1-chunk-0",
                    "score": 0.95,
                    "metadata": {
                        "source": "python.txt",
                        "document_id": "doc-1",
                        "text": "Python is a programming language.",
                    },
                },
                {
                    "id": "doc-2-chunk-0",
                    "score": 0.80,
                    "metadata": {
                        "source": "java.txt",
                        "document_id": "doc-2",
                        "text": "Java is a programming language.",
                    },
                },
            ]

    class SourcesFakeLLM:
        def generate(self, prompt: str) -> str:
            return "Python is a programming language."

    rag = RAG(
        retriever=SourcesFakeRetriever(),
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=SourcesFakeLLM(),
    )

    response = rag.ask_with_sources(
        query="What is Python?",
        top_k=2,
    )

    assert response.answer == (
        "Python is a programming language."
    )

    assert len(response.sources) == 2

    assert response.sources[0].id == "doc-1-chunk-0"
    assert response.sources[0].document_id == "doc-1"
    assert response.sources[0].source == "python.txt"
    assert response.sources[0].score == 0.95

    assert response.sources[1].id == "doc-2-chunk-0"
    assert response.sources[1].document_id == "doc-2"
    assert response.sources[1].source == "java.txt"
    assert response.sources[1].score == 0.80

def test_rag_limits_context_before_generation():
    from app.rag.context_limiter import ContextLimiter

    class FakeRetriever:
        def retrieve(self, query: str, top_k: int):
            return [
                {
                    "id": "first",
                    "score": 0.9,
                    "metadata": {
                        "source": "first.txt",
                        "text": "first context",
                    },
                },
                {
                    "id": "second",
                    "score": 0.8,
                    "metadata": {
                        "source": "second.txt",
                        "text": "second context",
                    },
                }
            ]

    class FakeLLM:
        def __init__(self):
            self.prompt = ""

        def generate(self, prompt: str) -> str:
            self.prompt = prompt
            return "answer"

    llm = FakeLLM()

    rag = RAG(
        retriever=FakeRetriever(),
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=llm,
        context_limiter=ContextLimiter(
            max_characters=len("first context"),
        ),
    )

    result = rag.ask(
        query="question",
        top_k=2,
    )

    assert result == "answer"
    assert "first context" in llm.prompt
    assert "second context" not in llm.prompt

def test_rag_returns_controlled_answer_when_no_results_are_found():
    class EmptyRetriever:
        def retrieve(self, query: str, top_k: int):
            return []

    class FakeLLM:
        def __init__(self):
            self.called = False

        def generate(self, prompt: str) -> str:
            self.called = True
            return "hallucinated answer"

    llm = FakeLLM()

    rag = RAG(
        retriever=EmptyRetriever(),
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=llm,
    )

    result = rag.ask("What is Python?")

    assert result == (
        "I don't have enough context to answer this question."
    )

    assert llm.called is False

def test_rag_with_sources_returns_empty_sources_when_no_results_are_found():
    class EmptyRetriever:
        def retrieve(self, query: str, top_k: int):
            return []

    class FakeLLM:
        def generate(self, prompt: str) -> str:
            raise AssertionError(
                "LLM should not be called without context"
            )

    rag = RAG(
        retriever=EmptyRetriever(),
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=FakeLLM(),
    )

    response = rag.ask_with_sources(
        query="What is Python?",
    )

    assert response.answer == (
        "I don't have enough context to answer this question."
    )
    assert response.sources == []