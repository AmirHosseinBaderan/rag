from app.generation.llm import LLM
from app.rag.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever
from app.rag.rag import RAG


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

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return "FastAPI is a Python web framework."


def test_rag_generates_answer_from_retrieved_context() -> None:
    retriever = FakeRetriever()
    context_builder = ContextBuilder()
    llm = FakeLLM()

    rag = RAG(
        retriever=retriever,
        context_builder=context_builder,
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