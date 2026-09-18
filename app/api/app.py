from fastapi import FastAPI

from app.api.routes import health, documents, query

app = FastAPI(title="RAG API")

app.include_router(health.health_router)
app.include_router(documents.documents_router, prefix="/documents", tags=["documents"])
app.include_router(query.query_router, prefix="/query", tags=["query"])
