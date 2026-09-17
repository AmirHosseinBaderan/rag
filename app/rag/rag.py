from app.generation.llm import LLM
from app.rag.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever

class RAG:
    def __init__(
            self,
            retriever:Retriever,
            context_builder:ContextBuilder,
            llm:LLM
    ):
        self._retriever = retriever
        self._context_builder = context_builder
        self._llm = llm

    def ask(
            self,
            query:str,
            top_k:int = 3
    )-> str:
        results = self._retriever.retrieve(
            query=query,
            top_k=top_k
        )

        context = self._context_builder.build(
            results=results
        )
        prompt = self._build_prompt(
            query=query,
            context=context
        )

        return self._llm.generate(prompt=prompt)

    def _build_prompt(
            self,
            query:str,
            context:str
    )-> str:
        return (
            "Answer the question using only "
            "the provided context \n\n"
            f"Contex:\n {context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )