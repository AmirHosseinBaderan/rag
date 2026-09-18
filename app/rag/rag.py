from app.generation.llm import LLM
from app.generation.prompt_builder import PromptBuilder
from app.rag.context_builder import ContextBuilder
from app.rag.context_compressor import ContextCompressor
from app.rag.context_ranker import ContextRanker
from app.retrieval.retriever import Retriever
from app.rag.rag_response import RAGResponse, RAGSource
from app.rag.context_limiter import ContextLimiter

class RAG:
    def __init__(
        self,
        retriever: Retriever,
        context_ranker: ContextRanker,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm: LLM,
        query_rewriter=None,
        reranker=None,
        context_compressor: ContextCompressor | None = None,
        multi_query_retriever=None,
        context_limiter: ContextLimiter | None = None,
    ) -> None:
        self._retriever = retriever
        self._context_ranker = context_ranker
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._query_rewriter = query_rewriter
        self._reranker = reranker
        self._context_compressor = context_compressor
        self._multi_query_retriever = multi_query_retriever
        self._context_limiter = context_limiter

    def ask(
        self,
        query: str,
        top_k: int = 3,
        retrieval_k: int | None = None,
    ) -> str:
        search_query = query

        if self._query_rewriter is not None:
            search_query = self._query_rewriter.rewrite(query)

        candidate_k = retrieval_k or top_k

        if self._multi_query_retriever is not None:
            results = self._multi_query_retriever.retrieve(
                query=search_query,
                top_k=candidate_k,
            )
        else:
            results = self._retriever.retrieve(
                query=search_query,
                top_k=candidate_k,
            )

        if self._reranker is not None:
            results = self._reranker.rerank(
                query=query,
                results=results,
                top_k=candidate_k,
            )

        if self._context_compressor is not None:
            results = self._context_compressor.compress(
                query=query,
                results=results,
                top_k=top_k,
            )

        ranked_results = self._context_ranker.rank(
            results,
            top_n=top_k,
        )

        if self._context_limiter is not None:
            ranked_results = self._context_limiter.limit(
                ranked_results,
            )

        context = self._context_builder.build(ranked_results)

        prompt = self._prompt_builder.build(
            query=query,
            context=context,
        )

        return self._llm.generate(prompt)


    def ask_with_sources(
        self,
        query: str,
        top_k: int = 3,
        retrieval_k: int | None = None,
    ) -> RAGResponse:
        search_query = query
    
        if self._query_rewriter is not None:
            search_query = self._query_rewriter.rewrite(query)
    
        candidate_k = retrieval_k or top_k
    
        if self._multi_query_retriever is not None:
            results = self._multi_query_retriever.retrieve(
                query=search_query,
                top_k=candidate_k,
            )
        else:
            results = self._retriever.retrieve(
                query=search_query,
                top_k=candidate_k,
            )
    
        if self._reranker is not None:
            results = self._reranker.rerank(
                query=query,
                results=results,
                top_k=candidate_k,
            )
    
        if self._context_compressor is not None:
            results = self._context_compressor.compress(
                query=query,
                results=results,
                top_k=top_k,
            )
    
        ranked_results = self._context_ranker.rank(
            results,
            top_n=top_k,
        )

        if self._context_limiter is not None:
            ranked_results = self._context_limiter.limit(
                ranked_results,
            )
    
        context = self._context_builder.build(ranked_results)
    
        prompt = self._prompt_builder.build(
            query=query,
            context=context,
        )
    
        answer = self._llm.generate(prompt)
    
        sources = [
            RAGSource(
                id=result["id"],
                document_id=result["metadata"].get(
                    "document_id",
                    "",
                ),
                source=result["metadata"].get(
                    "source",
                    "unknown",
                ),
                score=result["score"],
            )
            for result in ranked_results
        ]
    
        return RAGResponse(
            answer=answer,
            sources=sources,
        )