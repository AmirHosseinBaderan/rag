from app.rag.context_ranker import ContextRanker
from app.rag.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.rag.rag import RAG


class FakeRetriever:
    def retrieve(self, query: str, top_k: int):
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


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "answer"


def test_rag_uses_ranked_context():
    retriever = FakeRetriever()
    llm = FakeLLM()

    ranker = ContextRanker()

    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()

    rag = RAG(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm,
        context_ranker=ranker
    )

    results = retriever.retrieve(
        query="test",
        top_k=2,
    )

    ranked_results = ranker.rank(results, top_n=2)

    context = context_builder.build(ranked_results)
    prompt = prompt_builder.build(
        query="test",
        context=context,
    )

    llm.generate(prompt)

    assert prompt.index("High relevance context.") < prompt.index(
        "Low relevance context."
    )