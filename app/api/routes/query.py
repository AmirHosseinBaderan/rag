from fastapi import APIRouter

from app.api.dependencies import get_rag
from app.api.schemas.query import QueryRequest, QueryResponse

query_router = APIRouter()


@query_router.post("", response_model=QueryResponse)
def query(request: QueryRequest):
    rag = get_rag()
    answer = rag.ask(query=request.q)
    return QueryResponse(answer=answer)
