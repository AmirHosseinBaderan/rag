import pytest
from app.rag.rag_trace import RAGTrace


def test_rag_trace_contains_execution_metrics():
    trace = RAGTrace(
        query="What is Python?",
        retrieved_count=5,
        final_context_count=3,
        retrieval_duration_ms=12.5,
        generation_duration_ms=45.2,
    )

    assert trace.query == "What is Python?"
    assert trace.retrieved_count == 5
    assert trace.final_context_count == 3
    assert trace.retrieval_duration_ms == 12.5
    assert trace.generation_duration_ms == 45.2




def test_rag_trace_rejects_negative_counts():
    with pytest.raises(
        ValueError,
        match="Counts cannot be negative",
    ):
        RAGTrace(
            query="question",
            retrieved_count=-1,
            final_context_count=1,
            retrieval_duration_ms=1.0,
            generation_duration_ms=1.0,
        )