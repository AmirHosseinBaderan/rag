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