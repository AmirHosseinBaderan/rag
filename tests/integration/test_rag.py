from uuid import uuid4

from app.domain.documents.document import Document
from app.generation.ollama_llm import OllamaLLM
from app.ingestion.chunker import Chunker
from app.rag.context_builder import ContextBuilder
from app.rag.document_indexer import DocumentIndexer
from app.rag.rag import RAG
from app.retrieval.ollama_embedder import OllamaEmbedder
from app.retrieval.qdrant_vector_store import QdrantVectorStore
from app.retrieval.retriever import Retriever
from app.generation.prompt_builder import PromptBuilder
from app.rag.context_ranker import ContextRanker


def test_rag_end_to_end() -> None:
    collection_name = f"rag-e2e-{uuid4()}"

    embedder = OllamaEmbedder(
        model="nomic-embed-text:latest",
        base_url="http://192.168.0.247:11434",
    )

    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=768,
        host="192.168.0.247",
        port=6333,
    )

    chunker = Chunker(
        chunk_size=200,
        overlap=50,
    )

    indexer = DocumentIndexer(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    document = Document(
        id="fastapi",
        content=(
            "FastAPI is a modern Python web framework "
            "for building APIs. "
            "It is based on standard Python type hints "
            "and provides automatic API documentation. "
            "FastAPI uses Pydantic for data validation. "
            "FastAPI applications can be served using Uvicorn."
        ),
        metadata={
            "source": "fastapi.md",
        },
    )

    indexer.index(document)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder()

    llm = OllamaLLM(
        model="gemma3:1b",
        base_url="http://192.168.0.247:11434",
    )

    rag = RAG(
       retriever=retriever,
       context_ranker=ContextRanker(),
       context_builder=ContextBuilder(),
       prompt_builder=PromptBuilder(),
       llm=llm,
    )

    answer = rag.ask(
        "What does FastAPI use for data validation?",
        top_k=2,
    )

    assert isinstance(answer, str)
    assert answer.strip()