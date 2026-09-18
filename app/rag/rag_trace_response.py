from dataclasses import dataclass

from app.rag.rag_response import RAGResponse
from app.rag.rag_trace import RAGTrace


@dataclass(frozen=True)
class RAGTraceResponse:
    response: RAGResponse
    trace: RAGTrace