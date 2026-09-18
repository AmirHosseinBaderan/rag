from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answer: str
