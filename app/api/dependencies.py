from app.generation.ollama_llm import OllamaLLM
from app.generation.prompt_builder import PromptBuilder
from app.ingestion.chunker import Chunker
from app.rag.context_builder import ContextBuilder
from app.rag.context_ranker import ContextRanker
from app.rag.document_indexer import DocumentIndexer
from app.rag.rag import RAG
from app.retrieval.embedder import Embedder
from app.retrieval.ollama_embedder import OllamaEmbedder
from app.retrieval.qdrant_vector_store import QdrantVectorStore
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore

OLLAMA_BASE_URL = "http://192.168.0.247:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:1b"
QDRANT_HOST = "192.168.0.247"
QDRANT_PORT = 6333
VECTOR_SIZE = 768
COLLECTION_NAME = "rag-documents"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


def get_vector_store() -> VectorStore:
    return QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        vector_size=VECTOR_SIZE,
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


def get_embedder() -> Embedder:
    return OllamaEmbedder(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_indexer() -> DocumentIndexer:
    return DocumentIndexer(
        chunker=Chunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )


def get_rag() -> RAG:
    vector_store = get_vector_store()
    embedder = get_embedder()
    return RAG(
        retriever=Retriever(embedder=embedder, vector_store=vector_store),
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL),
    )
