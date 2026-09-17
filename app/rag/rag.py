from app.generation.llm import LLM
from app.generation.prompt_builder import PromptBuilder
from app.rag.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever


class RAG:
    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm: LLM,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm = llm

    def ask(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        results = self._retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        context = self._context_builder.build(
            results
        )

        prompt = self._prompt_builder.build(
            query=query,
            context=context,
        )

        return self._llm.generate(prompt)