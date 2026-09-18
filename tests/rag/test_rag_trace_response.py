from app.rag.rag_response import RAGResponse, RAGSource
from app.rag.rag_trace import RAGTrace
from app.rag.rag_trace_response import RAGTraceResponse


def test_rag_trace_response_contains_response_and_trace():
    response = RAGResponse(
        answer="Python is a programming language.",
        sources=[
            RAGSource(
                id="chunk-1",
                document_id="doc-1",
                source="python.txt",
                score=0.95,
            )
        ],
    )

    trace = RAGTrace(
        query="What is Python?",
        retrieved_count=5,
        final_context_count=1,
        retrieval_duration_ms=12.5,
        generation_duration_ms=45.2,
    )

    result = RAGTraceResponse(
        response=response,
        trace=trace,
    )

    assert result.response == response
    assert result.trace == trace