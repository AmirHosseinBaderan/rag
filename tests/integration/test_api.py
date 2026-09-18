import os
import tempfile
from uuid import uuid4
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.app import app
from app.api.dependencies import get_indexer, get_rag
from app.ingestion.chunker import Chunker
from app.ingestion.document_loader import DocumentLoader
from app.rag.context_builder import ContextBuilder
from app.rag.document_indexer import DocumentIndexer
from app.rag.rag import RAG
from app.retrieval.embedder import Embedder
from app.retrieval.ollama_embedder import OllamaEmbedder
from app.retrieval.qdrant_vector_store import QdrantVectorStore
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.domain.documents.document import Document
from app.generation.ollama_llm import OllamaLLM
from app.generation.prompt_builder import PromptBuilder
from app.rag.context_ranker import ContextRanker


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def tmp_documents_dir(tmp_path):
    original = Path("app/api/routes/documents.py")
    import app.api.routes.documents as documents_module

    original_dir = documents_module.DOCUMENTS_DIR
    documents_module.DOCUMENTS_DIR = tmp_path
    yield tmp_path
    documents_module.DOCUMENTS_DIR = original_dir


class FakeEmbedder(Embedder):
    def embed(self, text: str):
        return [0.5] * 768


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.points = []

    def upsert(self, vector_id, vector, metadata):
        self.points.append({
            "vector_id": vector_id,
            "vector": vector,
            "metadata": metadata,
        })

    def search(self, vector, top_k, score_threshold=None, metadata_filter=None):
        return [
            {
                "id": p["vector_id"],
                "score": 0.9,
                "metadata": {
                    **p["metadata"],
                    "text": p["metadata"].get("text", ""),
                },
            }
            for p in self.points[:top_k]
        ]


class FakeLLM:
    def __init__(self):
        self.generated_responses = []

    def generate(self, prompt: str) -> str:
        self.generated_responses.append(prompt)
        return "Test answer"


def fake_get_indexer():
    return DocumentIndexer(
        chunker=Chunker(chunk_size=200, overlap=50),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )


def fake_get_rag():
    vector_store = FakeVectorStore()
    vector_store.points = [
        {
            "vector_id": "doc-1-chunk-0",
            "vector": [0.5] * 768,
            "metadata": {
                "source": "test.md",
                "document_id": "doc-1",
                "text": "FastAPI is a Python framework for building APIs.",
            },
        }
    ]
    return RAG(
        retriever=Retriever(
            embedder=FakeEmbedder(),
            vector_store=vector_store,
        ),
        context_ranker=ContextRanker(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=FakeLLM(),
    )


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestDocumentsEndpoint:
    def test_list_documents_empty(self, client, tmp_documents_dir):
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json() == {"documents": []}

    def test_upload_markdown_document(self, client, tmp_documents_dir):
        content = b"# Test Document\n\nThis is a test markdown document."
        response = client.post(
            "/documents",
            files={"file": ("test.md", content, "text/markdown")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test.md"
        assert data["document_id"] == "test"
        assert "Document indexed successfully" in data["message"]

        assert (tmp_documents_dir / "test.md").exists()

    def test_upload_pdf_document(self, client, tmp_documents_dir):
        from PyPDF2 import PdfWriter

        pdf_path = tmp_documents_dir / "test.pdf"
        with open(pdf_path, "wb") as f:
            writer = PdfWriter(f)
            writer.add_blank_page(width=200, height=200)
            writer.write(f)

        with open(pdf_path, "rb") as f:
            response = client.post(
                "/documents",
                files={"file": ("test.pdf", f.read(), "application/pdf")},
            )
        assert response.status_code == 201

    def test_upload_bad_file_type(self, client, tmp_documents_dir):
        response = client.post(
            "/documents",
            files={"file": ("bad.exe", b"binary", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "File type not allowed" in response.json()["detail"]

    def test_upload_duplicate_file(self, client, tmp_documents_dir):
        content = b"# Duplicate\n\nSame content."
        client.post(
            "/documents",
            files={"file": ("dup.md", content, "text/markdown")},
        )
        response = client.post(
            "/documents",
            files={"file": ("dup.md", content, "text/markdown")},
        )
        assert response.status_code == 201

    def test_list_documents_after_upload(self, client, tmp_documents_dir):
        content = b"# Doc 1\n\nContent."
        client.post(
            "/documents",
            files={"file": ("doc1.md", content, "text/markdown")},
        )
        content2 = b"# Doc 2\n\nContent."
        client.post(
            "/documents",
            files={"file": ("doc2.md", content2, "text/markdown")},
        )

        response = client.get("/documents")
        assert response.status_code == 200
        docs = response.json()["documents"]
        assert len(docs) == 2
        names = {d["name"] for d in docs}
        assert "doc1.md" in names
        assert "doc2.md" in names

        for doc in docs:
            assert "size" in doc
            assert "modified" in doc
            assert doc["size"] > 0


class TestQueryEndpoint:
    def test_query_returns_answer(self, client, tmp_documents_dir):
        response = client.post(
            "/query",
            json={"q": "What is FastAPI?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_query_empty_returns_422(self, client):
        response = client.post("/query", json={"q": ""})
        assert response.status_code == 422

    def test_query_uses_rag_ask(self, client, tmp_documents_dir):
        from app.api.routes import query as query_module

        rag_mock = fake_get_rag()
        original = query_module.get_rag
        query_module.get_rag = lambda: rag_mock
        try:
            response = client.post(
                "/query",
                json={"q": "test query"},
            )
            assert response.status_code == 200
            assert response.json()["answer"] == "Test answer"
        finally:
            query_module.get_rag = original


class TestDocumentIndexingIntegration:
    def test_upload_indexes_document_in_qdrant(self, client, tmp_documents_dir):
        from app.api.routes.documents import _extract_text

        content = b"# Integration Test\n\nFastAPI integration test document."
        md_path = tmp_documents_dir / "integration.md"
        md_path.write_bytes(content)

        extracted = _extract_text(md_path)
        assert "Integration Test" in extracted
        assert "FastAPI" in extracted

    def test_document_flow_with_real_services(self):
        collection_name = f"api-test-{uuid4()}"

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

        indexer = DocumentIndexer(
            chunker=Chunker(chunk_size=200, overlap=50),
            embedder=embedder,
            vector_store=vector_store,
        )

        document = Document(
            id="api-test-doc",
            content=(
                "FastAPI is a modern Python web framework "
                "for building APIs. "
                "It uses Pydantic for data validation."
            ),
            metadata={"source": "api-test.md"},
        )

        indexer.index(document)

        retriever = Retriever(embedder=embedder, vector_store=vector_store)
        rag = RAG(
            retriever=retriever,
            context_ranker=ContextRanker(),
            context_builder=ContextBuilder(),
            prompt_builder=PromptBuilder(),
            llm=OllamaLLM(
                model="gemma3:1b",
                base_url="http://192.168.0.247:11434",
            ),
        )

        answer = rag.ask("What does FastAPI use?", top_k=2)
        assert isinstance(answer, str)
        assert answer.strip()
