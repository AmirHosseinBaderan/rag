from app.generation.llm import LLM
from app.generation.prompt_builder import PromptBuilder
from app.rag.context_builder import ContextBuilder
from app.rag.context_compressor import ContextCompressor
from app.rag.context_ranker import ContextRanker
from app.retrieval.retriever import Retriever
from app.rag.rag_response import RAGResponse, RAGSource
from app.rag.context_limiter import ContextLimiter
import time

from app.rag.rag_trace import RAGTrace
from app.rag.rag_trace_response import RAGTraceResponse

class RAG:
    _NO_CONTEXT_ANSWER = (
        "I don't have enough context to answer this question."
    )

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

        if not results:
            return self._NO_CONTEXT_ANSWER

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

        if not results:
            return RAGResponse(
                answer=self._NO_CONTEXT_ANSWER,
                sources=[],
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

    def ask_with_trace(
        self,
        query: str,
        top_k: int,
        retrieval_k: int,
    ) -> RAGTraceResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")
    
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
    
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than zero")
    
        search_query = query
    
        if self._query_rewriter is not None:
            search_query = self._query_rewriter.rewrite(query)
    
        retrieval_started_at = time.perf_counter()
    
        if self._multi_query_retriever is not None:
            results = self._multi_query_retriever.retrieve(
                query=search_query,
                top_k=retrieval_k,
            )
        else:
            results = self._retriever.retrieve(
                query=search_query,
                top_k=retrieval_k,
            )
    
        retrieval_duration_ms = (
            time.perf_counter() - retrieval_started_at
        ) * 1000
    
        retrieved_count = len(results)
    
        if not results:
            response = RAGResponse(
                answer=self._NO_CONTEXT_ANSWER,
                sources=[],
            )
    
            trace = RAGTrace(
                query=query,
                retrieved_count=0,
                final_context_count=0,
                retrieval_duration_ms=retrieval_duration_ms,
                generation_duration_ms=0.0,
            )
    
            return RAGTraceResponse(
                response=response,
                trace=trace,
            )
    
        if self._reranker is not None:
            results = self._reranker.rerank(
                query=search_query,
                results=results,
                top_k=retrieval_k,
            )
    
        if self._context_compressor is not None:
            results = self._context_compressor.compress(
                query=search_query,
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
    
        final_context_count = len(ranked_results)
    
        context = self._context_builder.build(ranked_results)
    
        prompt = self._prompt_builder.build(
            query=query,
            context=context,
        )
    
        generation_started_at = time.perf_counter()
    
        answer = self._llm.generate(prompt)
    
        generation_duration_ms = (
            time.perf_counter() - generation_started_at
        ) * 1000
    
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
    
        response = RAGResponse(
            answer=answer,
            sources=sources,
        )
    
        trace = RAGTrace(
            query=query,
            retrieved_count=retrieved_count,
            final_context_count=final_context_count,
            retrieval_duration_ms=retrieval_duration_ms,
            generation_duration_ms=generation_duration_ms,
        )
    
        return RAGTraceResponse(
            response=response,
            trace=trace,
        )