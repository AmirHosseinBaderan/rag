from fastapi import FastAPI

from app.api.routes import health, documents, query

app = FastAPI(
    title="RAG API",
    description=(
        "Retrieval-Augmented Generation API. "
        "Upload documents (Markdown, Word, PDF), "
        "index them with Ollama embeddings into Qdrant, "
        "and query them using a RAG pipeline."
    ),
    version="1.0.0",
    contact={
        "name": "RAG Team",
    },
    license_info={
        "name": "MIT",
    },
)

app.include_router(
    health.health_router,
    tags=["Health"],
)
app.include_router(
    documents.documents_router,
    prefix="/documents",
    tags=["Documents"],
)
app.include_router(
    query.query_router,
    prefix="/query",
    tags=["Query"],
)
