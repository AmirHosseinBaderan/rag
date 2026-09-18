# RAG API — Retrieval-Augmented Generation

A complete Retrieval-Augmented Generation (RAG) system with a REST API. Upload documents (Markdown, Word, PDF), index them using embeddings, store them in a vector database, and query them using a powerful RAG pipeline that combines retrieval, reranking, context building, and LLM-based answer generation.

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [How This Project Works](#how-this-project-works)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Layer-by-Layer Explanation](#layer-by-layer-explanation)
6. [File-by-File Description](#file-by-file-description)
7. [API Endpoints](#api-endpoints)
8. [Configuration](#configuration)
9. [Installation](#installation)
10. [Running](#running)
11. [Testing](#testing)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines two powerful AI capabilities:

1. **Retrieval**: Finding relevant information from a collection of documents
2. **Generation**: Using a Large Language Model (LLM) to generate an answer based on that retrieved information

Instead of relying only on the LLM's training data (which may be outdated or incomplete), RAG **looks up** relevant documents first, then asks the LLM to answer **using only those documents** as context.

Think of it like this:
- **Without RAG**: You ask an AI a question and it answers from memory (training data).
- **With RAG**: You ask an AI a question. First, it searches through your files for relevant information. Then it answers your question **using only what it found in your files**.

This gives you:
- **More accurate answers** — grounded in your actual documents
- **Up-to-date information** — no need to retrain the model
- **Traceable sources** — you can see which documents were used

---

## How This Project Works

The system has two main phases: **Ingestion** (adding documents) and **Querying** (asking questions).

### Phase 1: Ingestion (Adding Documents)

```
Client uploads file (md, docx, pdf)
        │
        ▼
   DocumentLoader
   (reads file content)
        │
        ▼
      Chunker
   (splits text into pieces)
        │
        ▼
   Ollama Embedder
   (converts text to vectors)
        │
        ▼
    Qdrant
   (stores vectors for search)
```

**Step-by-step:**

1. You upload a file (e.g., `fastapi.md`)
2. `DocumentLoader` reads the file and creates a `Document` object with the text content
3. `Chunker` splits the long text into smaller chunks (like cutting a book into pages)
4. `OllamaEmbedder` converts each chunk into a numerical vector (a list of 768 numbers) — this is the "meaning" of the text
5. `DocumentIndexer` stores each vector in Qdrant along with metadata (file name, chunk ID, text content)

### Phase 2: Querying (Asking Questions)

```
Client asks a question
        │
        ▼
   Query Rewriter (optional)
   (improves the question)
        │
        ▼
  Embed Query
   (convert question to vector)
        │
        ▼
 Search Qdrant
   (find similar chunks)
        │
        ▼
    Reranker (optional)
   (re-score by semantic similarity)
        │
        ▼
  Rank Results
   (sort by relevance score)
        │
        ▼
Build Context
   (combine top chunks into text)
        │
        ▼
   Build Prompt
   (attach question + context)
        │
        ▼
     LLM (Ollama)
   (generate answer)
        │
        ▼
   Return Answer
```

**Step-by-step:**

1. You send a question via the API (e.g., "What does FastAPI use for data validation?")
2. Optionally, a `QueryRewriter` improves the question for better search results
3. The question is converted to a vector using `OllamaEmbedder`
4. The vector is searched against Qdrant to find the most similar document chunks
5. Optionally, a `SemanticReranker` re-scores results using cosine similarity
6. `ContextRanker` sorts results by their score (highest first)
7. `ContextBuilder` combines the top chunks into a single context string
8. Optionally, `ContextLimiter` ensures the context doesn't exceed a character limit
9. `PromptBuilder` creates a prompt: "Answer this question using only this context..."
10. `OllamaLLM` generates the final answer
11. The answer is returned to you

---

## Architecture

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
     ▼
┌──────────┐       ┌──────────────┐
│  FastAPI │       │    RAG       │
│   API    │──────▶│   .ask()     │
└────┬─────┘       └──────┬───────┘
     │                    │
     │  POST /documents   │  POST /query
     ▼                    ▼
┌──────────────┐    ┌──────────────┐
│ DocumentLoader│    │  Retriever   │
│   Chunker    │    │  Embedder    │
│  Indexer     │    │ VectorStore  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Ollama Embed │    │  Ollama LLM  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│   Qdrant     │    │   Ollama     │
│  (Vector DB) │    │  (LLM)       │
└──────────────┘    └──────────────┘
```

---

## Project Structure

```
rag/
├── main.py                          # Entry point (starts uvicorn server)
├── requirements.txt                 # Python dependencies
├── documents/                       # Where uploaded files are stored
│   └── fastapi.md
├── app/
│   ├── api/                         # FastAPI web layer
│   │   ├── app.py                   # FastAPI app instance + router setup
│   │   ├── dependencies.py          # Shared components (indexer, RAG, etc.)
│   │   ├── schemas/                 # Request/response data models
│   │   │   ├── document.py          # Document schemas
│   │   │   └── query.py             # Query schemas
│   │   └── routes/                  # HTTP route handlers
│   │       ├── health.py            # Health check endpoint
│   │       ├── documents.py         # Document upload & list endpoints
│   │       └── query.py             # Query endpoint
│   ├── domain/                      # Core domain objects
│   │   ├── documents/
│   │   │   └── document.py          # Document data class
│   │   └── chunks/
│   │       └── chunk.py             # Chunk data class
│   ├── ingestion/                   # Document loading & chunking
│   │   ├── document_loader.py       # Reads files into Document objects
│   │   └── chunker.py               # Splits documents into chunks
│   ├── rag/                         # RAG pipeline components
│   │   ├── rag.py                   # Main RAG orchestrator
│   │   ├── document_indexer.py      # Indexes documents into vector store
│   │   ├── context_ranker.py        # Ranks retrieved results
│   │   ├── context_builder.py       # Builds context text from results
│   │   ├── context_compressor.py    # Deduplicates similar contexts
│   │   ├── context_limiter.py       # Limits context by character count
│   │   ├── query_rewriter.py        # Rewrites queries for better search
│   │   ├── query_generator.py       # Generates multiple search queries
│   │   ├── semantic_reranker.py     # Semantic reranking of results
│   │   ├── multi_query_retriever.py # Retrieves using multiple queries
│   │   ├── retrieval_strategy.py    # Abstract base for retrieval
│   │   ├── rag_response.py          # Response data classes
│   │   ├── rag_trace.py             # Trace/observability data
│   │   └── rag_trace_response.py    # Trace response data class
│   ├── retrieval/                   # Information retrieval components
│   │   ├── embedder.py              # Abstract Embedder base class
│   │   ├── ollama_embedder.py       # Ollama embedding implementation
│   │   ├── local_embedder.py        # Local sentence-transformers embedder
│   │   ├── vector_store.py          # Abstract VectorStore base class
│   │   ├── qdrant_vector_store.py   # Qdrant implementation
│   │   ├── retriever.py             # Full retrieval (embed + search)
│   │   ├── hybrid_search.py         # Combines semantic + keyword search
│   │   └── keyword_search.py        # Keyword-based search
│   ├── generation/                  # Text generation components
│   │   ├── llm.py                   # Abstract LLM base class
│   │   ├── ollama_llm.py            # Ollama LLM implementation
│   │   └── prompt_builder.py        # Builds prompts from query + context
│   └── evaluation/                  # Evaluation & metrics
│       ├── metrics/
│       │   ├── cosine_similarity.py # Cosine similarity calculation
│       │   ├── precision_at_k.py    # Precision@K metric
│       │   ├── recall_at_k.py       # Recall@K metric
│       │   └── mrr.py               # Mean Reciprocal Rank metric
│       └── ...                      # Other evaluation components
└── tests/
    ├── rag/                         # Unit tests for RAG components
    ├── retrieval/                   # Tests for retrieval components
    ├── generation/                  # Tests for generation components
    └── integration/                 # Integration tests
        ├── test_api.py              # API endpoint integration tests
        └── test_rag.py              # RAG end-to-end integration test
```

---

## Layer-by-Layer Explanation

### Layer 1: Domain (Core Objects)

**Location**: `app/domain/`

This is the **foundation** of the project. It defines the core data objects that every other layer uses.

Think of it like the "blueprints" — everything else is built on top of these objects.

#### `app/domain/documents/document.py`

```python
@dataclass(frozen=True)
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

**What it is**: A `Document` represents a piece of text with an ID and extra information (metadata).

**Analogy**: Imagine a library book. The `id` is the book's catalog number, `content` is the text inside, and `metadata` is the card in the pocket (author, title, genre, etc.).

**Why frozen?**: Once created, a Document can't be changed. This prevents accidental modifications and makes the system more predictable.

#### `app/domain/chunks/chunk.py`

```python
@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

**What it is**: A `Chunk` is a small piece of a Document.

**Analogy**: If a Document is a whole pizza, a Chunk is a slice. Each slice belongs to a pizza (has a `document_id`) and has its own content (the toppings and crust on that slice).

**Why chunks?**: LLMs can only process a limited amount of text at once. So we split long documents into smaller pieces (chunks) that fit within the LLM's context window.

---

### Layer 2: Ingestion (Loading & Chunking)

**Location**: `app/ingestion/`

This layer is responsible for **taking raw files and preparing them for indexing**. It's like a factory assembly line: raw materials come in, processed components come out.

#### `app/ingestion/document_loader.py`

```python
class DocumentLoader:
    def load(self, path: Path) -> Document:
        content = path.read_text(encoding="utf-8")
        return Document(
            id=path.stem,
            content=content,
            metadata={"source": path.name},
        )
```

**What it does**: Reads a file from disk and creates a `Document` object.

**How it works**:
1. Takes a file path (e.g., `/path/to/fastapi.md`)
2. Reads the file content as text
3. Creates a `Document` with:
   - `id` = filename without extension (`fastapi`)
   - `content` = file text
   - `metadata` = `{"source": "fastapi.md"}`

**Example**:
```
File: fastapi.md → Document(id="fastapi", content="# FastAPI...", metadata={"source": "fastapi.md"})
```

#### `app/ingestion/chunker.py`

```python
class Chunker:
    def __init__(self, chunk_size: int, overlap: int = 0):
        ...

    def chunk(self, document: Document) -> list[Chunk]:
        ...
```

**What it does**: Splits a long document into smaller chunks.

**How it works**: Imagine you have a 1000-character document and `chunk_size=200, overlap=50`:
1. Characters 0-200 → Chunk 0 (id="doc-chunk-0")
2. Characters 150-350 → Chunk 1 (overlap of 50 chars)
3. Characters 300-500 → Chunk 2
4. ... and so on

**Why overlap?**: Overlapping chunks prevent important content from being lost at chunk boundaries. If a sentence starts at character 195 and ends at 210, it appears in both Chunk 0 and Chunk 1.

**Key parameters**:
- `chunk_size`: How many characters per chunk (default: 200)
- `overlap`: How many characters to overlap between chunks (default: 50)

---

### Layer 3: API (Web Interface)

**Location**: `app/api/`

This is the **front door** of the application. It handles HTTP requests and responses using FastAPI.

#### `app/api/app.py`

```python
app = FastAPI(title="RAG API", ...)
app.include_router(health.health_router, tags=["Health"])
app.include_router(documents.documents_router, prefix="/documents", tags=["Documents"])
app.include_router(query.query_router, prefix="/query", tags=["Query"])
```

**What it does**: Creates the FastAPI application and connects all routes.

**Think of it as**: The main office building. The routers are the departments (Health, Documents, Query), and `include_router` is like putting up signs to direct visitors to the right department.

#### `app/api/dependencies.py`

```python
def get_vector_store() -> VectorStore: ...
def get_embedder() -> Embedder: ...
def get_indexer() -> DocumentIndexer: ...
def get_rag() -> RAG: ...
```

**What it does**: Centralized factory functions that create and return shared components.

**Why this matters**: Instead of each route creating its own components (which would be wasteful), the routes call these functions to get shared instances. It's like a utility closet — everyone borrows from the same supply.

**Components it provides**:
- `get_vector_store()`: Qdrant connection for storing/retrieving vectors
- `get_embedder()`: Ollama embedder for converting text to vectors
- `get_indexer()`: DocumentIndexer (combines chunker + embedder + vector store)
- `get_rag()`: Full RAG pipeline (retriever + ranker + builder + LLM)

#### `app/api/schemas/document.py`

**What it does**: Defines data structures for document-related requests and responses.

**Classes**:
- `DocumentResponse`: A document's summary (id, source, size)
- `DocumentCreate`: Data needed to create a document (content, metadata)
- `DocumentListResponse`: A list of documents

#### `app/api/schemas/query.py`

**What it does**: Defines data structures for query requests and responses.

**Classes**:
- `QueryRequest`: The question being asked (`q: str`, min length 1)
- `QueryResponse`: The answer (`answer: str`)

#### `app/api/routes/health.py`

```python
@health_router.get("/health")
def health_check():
    return {"status": "ok"}
```

**What it does**: Simple health check endpoint. Used to verify the server is running.

**When to use**: Load balancers and monitoring tools call this to check if the API is alive.

#### `app/api/routes/documents.py`

**`POST /documents`** — Upload a document:
1. Validates file extension (must be `.md`, `.docx`, or `.pdf`)
2. Saves the file to the `documents/` folder
3. Extracts text from the file (handles all 3 formats)
4. Creates a `Document` object
5. Indexes it through the `DocumentIndexer` (chunk → embed → store in Qdrant)
6. Returns success response with filename and document ID

**`GET /documents`** — List documents:
1. Lists all files in the `documents/` folder
2. Returns their names, sizes, and modification times

#### `app/api/routes/query.py`

**`POST /query`** — Ask a question:
1. Receives a `QueryRequest` with the question text
2. Gets the RAG pipeline via `get_rag()`
3. Calls `rag.ask(query)` which runs the full RAG pipeline
4. Returns the generated answer as `QueryResponse`

---

### Layer 4: Retrieval (Finding Information)

**Location**: `app/retrieval/`

This layer is responsible for **finding relevant information** in the vector store. Think of it as a search engine.

#### `app/retrieval/embedder.py`

```python
class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...
```

**What it is**: Abstract base class for text embeddings.

**What is an embedding?**: A list of numbers (e.g., 768 numbers) that represents the "meaning" of a piece of text. Similar texts have similar embeddings.

**Analogy**: Think of it as a fingerprint. Each word/concept has a unique fingerprint, and similar concepts have similar fingerprints.

#### `app/retrieval/ollama_embedder.py`

```python
class OllamaEmbedder(Embedder):
    def embed(self, text: str) -> list[float]:
        response = httpx.post(f"{self._base_url}/api/embed", json={...})
        return data["embeddings"][0]
```

**What it does**: Calls Ollama's embedding API to convert text to vectors.

**Model used**: `nomic-embed-text:latest` — a model specifically trained for creating text embeddings.

#### `app/retrieval/local_embedder.py`

```python
class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)
```

**What it does**: Uses a local sentence-transformers model for embeddings (no Ollama needed).

#### `app/retrieval/vector_store.py`

```python
class VectorStore(ABC):
    @abstractmethod
    def upsert(self, vector_id, vector, metadata): ...
    @abstractmethod
    def search(self, vector, top_k, ...): ...
```

**What it is**: Abstract base class for vector databases.

**`upsert`**: Store a vector (insert or update)
**`search`**: Find vectors similar to a query vector

#### `app/retrieval/qdrant_vector_store.py`

```python
class QdrantVectorStore(VectorStore):
    def __init__(self, collection_name, vector_size, host, port):
        self._client = QdrantClient(host=host, port=port)
        self._ensure_collection(vector_size)
```

**What it does**: Concrete implementation using Qdrant as the vector database.

**How it works**:
- **`upsert`**: Creates a point with an ID, vector, and metadata, then stores it in Qdrant
- **`search`**: Sends a query vector to Qdrant and returns the most similar points with their scores

**Why Qdrant?**: It's a purpose-built vector database optimized for fast similarity search across millions of vectors.

#### `app/retrieval/retriever.py`

```python
class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        ...

    def retrieve(self, query: str, top_k: int, ...) -> list[dict]:
        vector = self._embedder.embed(query)
        return self._vector_store.search(vector=vector, top_k=top_k, ...)
```

**What it does**: The complete retrieval pipeline — embeds a query and searches the vector store.

**How it works**:
1. Convert the question to a vector
2. Search the vector store for similar chunks
3. Return the results (id, score, metadata)

#### `app/retrieval/keyword_search.py`

**What it does**: Traditional keyword-based search using token matching.

**How it works**:
1. Tokenize both query and document text into words
2. Count matching words
3. Score = matching_words / total_query_words
4. Sort by score, return top K

**When to use**: Good for exact word matches, poor for semantic similarity.

#### `app/retrieval/hybrid_search.py`

**What it does**: Combines semantic and keyword search results using weighted scoring.

**How it works**:
1. Takes semantic results and keyword results
2. For each document, calculates: `final_score = semantic_score * 0.6 + keyword_score * 0.4`
3. Sorts by final score, returns top K

**Why hybrid?**: Semantic search understands meaning but might miss exact terms. Keyword search catches exact terms but misses paraphrases. Together, they cover both.

---

### Layer 5: RAG Pipeline (Intelligence Layer)

**Location**: `app/rag/`

This is the **brain** of the system. It orchestrates all the retrieval and generation steps.

#### `app/rag/rag.py`

```python
class RAG:
    def __init__(self, retriever, context_ranker, context_builder, prompt_builder, llm, ...):
        ...

    def ask(self, query: str, top_k: int = 3, retrieval_k: int | None = None) -> str:
        ...
```

**What it is**: The main RAG orchestrator. This is where the magic happens.

**How `ask()` works** (step by step):

1. **Query Rewriting** (optional): If a `query_rewriter` is configured, it rewrites the query for better search
   ```
   "What does FastAPI use?" → "What data validation tool does FastAPI utilize?"
   ```

2. **Retrieval**: Embed the query and search the vector store for top-k results
   ```
   query_vector → Qdrant → [chunk1, chunk2, chunk3]
   ```

3. **Reranking** (optional): If a `reranker` is configured, re-score results by semantic similarity

4. **Context Compression** (optional): If a `context_compressor` is configured, remove duplicate or irrelevant chunks

5. **Ranking**: `ContextRanker` sorts results by score (highest first)
   ```
   chunk2 (score=0.95) → chunk1 (score=0.80) → chunk3 (score=0.60)
   ```

6. **Context Limiting** (optional): `ContextLimiter` ensures total context stays within character limits

7. **Context Building**: `ContextBuilder` combines top chunks into a single text
   ```
   "[Source: doc1.md]\nFastAPI uses Pydantic.\n\n[Source: doc2.md]\n..."
   ```

8. **Prompt Building**: `PromptBuilder` creates the LLM prompt
   ```
   "Answer the question using only the provided context.
   Context: ...
   Question: ...
   Answer:"
   ```

9. **Generation**: `LLM.generate(prompt)` returns the final answer

**Key parameters**:
- `top_k`: How many chunks to use for context (default: 3)
- `retrieval_k`: How many chunks to retrieve initially (more candidates = better chance of finding relevant ones)

#### `app/rag/context_ranker.py`

**What it does**: Sorts search results by their score (highest first).

**How it works**: Simple sorting — results are sorted by their `score` field in descending order. The highest-scoring result comes first.

#### `app/rag/context_builder.py`

**What it does**: Converts ranked results into a readable context string.

**How it works**: For each result, creates:
```
[Source: filename.md]
Text content of the chunk...
```
Then joins them with double newlines.

#### `app/rag/context_compressor.py`

**What it does**: Removes duplicate and irrelevant chunks from the context.

**How it works**:
1. Embed the query and each result
2. Skip results with cosine similarity to query < threshold (0.8 by default)
3. Skip results that are too similar to already-selected results (deduplication)
4. Keep going until top_k results are selected

#### `app/rag/context_limiter.py`

**What it does**: Limits the total characters in the context.

**How it works**: Adds results one by one, stopping when adding the next result would exceed `max_characters`.

#### `app/rag/query_rewriter.py`

**What it does**: Improves a user's question for better search results.

**How it works**: Sends the question to the LLM with a prompt like:
```
"Rewrite the following user question into a clear and concise search query.
Question: What does FastAPI use for data validation?
Search query:"
```

#### `app/rag/query_generator.py`

**What it does**: Generates multiple search queries from a single question.

**How it works**: Sends a prompt to the LLM asking for multiple queries on separate lines:
```
"Generate multiple search queries for the following question.
Question: What is FastAPI?
Search queries:"
→ ["What is FastAPI?", "FastAPI definition", "FastAPI web framework"]
```

#### `app/rag/semantic_reranker.py`

**What it does**: Re-scores results using cosine similarity between query and document vectors.

**How it works**:
1. Embed the query and each result text
2. Calculate cosine similarity for each pair
3. Add a `rerank_score` to each result
4. Sort by `rerank_score` (highest first)
5. Return top_k results

#### `app/rag/multi_query_retriever.py`

**What it does**: Generates multiple queries, searches for each, and merges results.

**How it works**:
1. Generate 3-5 queries from the original question
2. Search the vector store for each query
3. Deduplicate by result ID (keep highest score)
4. Sort by score, return top_k

**Why?**: A single question might be phrased in ways that miss important documents. Multiple queries cover different angles.

#### `app/rag/retrieval_strategy.py`

Abstract base class for retrieval strategies. Used for extensibility and testing.

#### `app/rag/document_indexer.py`

```python
class DocumentIndexer:
    def __init__(self, chunker, embedder, vector_store):
        ...

    def index(self, document: Document) -> None:
        chunks = self._chunker.chunk(document)
        for chunk in chunks:
            vector = self._embedder.embed(chunk.text)
            metadata = {**chunk.metadata, "document_id": chunk.document_id, "text": chunk.text}
            self._vector_store.upsert(vector_id=chunk.id, vector=vector, metadata=metadata)
```

**What it does**: The complete indexing pipeline — chunks a document, embeds each chunk, and stores in the vector store.

**How it works**:
1. Split document into chunks
2. For each chunk:
   a. Convert text to vector
   b. Store in Qdrant with metadata (source, document_id, text, chunk ID)

#### `app/rag/rag_response.py`

Data classes for the response when using `ask_with_sources()`:
- `RAGSource`: A single source (id, document_id, source filename, score)
- `RAGResponse`: The answer + list of sources

#### `app/rag/rag_trace.py`

Data class for observability/tracing:
- `query`: The original question
- `retrieved_count`: How many chunks were retrieved
- `final_context_count`: How many chunks were used for the answer
- `retrieval_duration_ms`: Time spent on retrieval
- `generation_duration_ms`: Time spent on LLM generation

#### `app/rag/rag_trace_response.py`

Combines `RAGResponse` (answer + sources) with `RAGTrace` (performance metrics).

---

### Layer 6: Generation (Answer Creation)

**Location**: `app/generation/`

This layer is responsible for **generating the final answer** using an LLM.

#### `app/generation/llm.py`

```python
class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...
```

**What it is**: Abstract base class for all LLM implementations. Any LLM that follows this interface can be plugged into the RAG pipeline.

#### `app/generation/ollama_llm.py`

```python
class OllamaLLM(LLM):
    def generate(self, prompt: str) -> str:
        response = httpx.post(f"{self._base_url}/api/generate", json={
            "model": self._model, "prompt": prompt, "stream": False
        })
        return data["response"]
```

**What it does**: Calls Ollama's generation API.

**Model used**: `gemma3:1b` — a lightweight, efficient language model.

**How it works**:
1. Send the prompt to Ollama
2. Ollama generates text based on the prompt and context
3. Return the generated text

#### `app/generation/prompt_builder.py`

```python
class PromptBuilder:
    def build(self, query: str, context: str) -> str:
        prompt = (
            "Answer the question using only the provided context.\n\n"
            f"Context:\n{context}"
        )
        prompt += (f"\n\nQuestion:\n{query}\n\nAnswer:")
        return prompt
```

**What it does**: Creates a well-structured prompt from the user's question and the retrieved context.

**Why it matters**: A good prompt guides the LLM to:
1. Only use the provided context (avoid hallucinations)
2. Answer the specific question asked
3. Produce a clear, concise answer

**Example output**:
```
Answer the question using only the provided context.

Context:
[Source: fastapi.md]
FastAPI is a modern Python web framework...

Question:
What does FastAPI use for data validation?

Answer:
```

---

### Layer 7: Evaluation (Quality Measurement)

**Location**: `app/evaluation/`

This layer provides metrics to **measure how well the RAG system performs**.

#### `app/evaluation/metrics/cosine_similarity.py`

**What it does**: Calculates the cosine similarity between two vectors.

**Formula**: `similarity = (A · B) / (|A| × |B|)`

**Range**: -1 (opposite) to 1 (identical). In practice for embeddings, usually 0.5 to 1.0.

**Why it matters**: This is the core metric used throughout the system for ranking, reranking, and compressing contexts.

#### `app/evaluation/metrics/precision_at_k.py`

**What it does**: Measures — out of the top K results retrieved, how many were actually relevant?

**Formula**: `Precision@K = (relevant results in top K) / K`

**Example**: If K=3 and 2 of the 3 retrieved results are relevant → Precision@3 = 2/3 = 0.67

#### `app/evaluation/metrics/recall_at_k.py`

**What it does**: Measures — out of ALL relevant results, how many did we find in the top K?

**Formula**: `Recall@K = (relevant results in top K) / (total relevant results)`

**Example**: If there are 5 relevant documents total and we find 3 in top K → Recall@K = 3/5 = 0.60

#### `app/evaluation/metrics/mrr.py`

**What it does**: Mean Reciprocal Rank — measures where the first relevant result appears.

**Formula**: `MRR = average(1/rank_of_first_relevant_result)`

**Example**: If the first relevant result is at rank 2 → score = 1/2 = 0.5. If at rank 1 → score = 1.0.

---

## API Endpoints

### `GET /health`

Check if the API is running.

**Request**:
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{"status": "ok"}
```

---

### `POST /documents`

Upload a document to index.

**Supported formats**: `.md` (Markdown), `.docx` (Word), `.pdf`

**Request**:
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@mydocument.md"
```

**Response** (success — 201):
```json
{
  "message": "Document indexed successfully",
  "filename": "mydocument.md",
  "document_id": "mydocument"
}
```

**Error** (unsupported format — 400):
```json
{
  "detail": "File type not allowed. Use .md, .docx, or .pdf"
}
```

---

### `GET /documents`

List all uploaded documents.

**Request**:
```bash
curl http://localhost:8000/documents
```

**Response**:
```json
{
  "documents": [
    {
      "name": "fastapi.md",
      "size": 250,
      "modified": 1789648456.34
    }
  ]
}
```

---

### `POST /query`

Ask a question and get an answer using RAG.

**Request**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "What does FastAPI use for data validation?"}'
```

**Response**:
```json
{
  "answer": "FastAPI uses Pydantic for data validation."
}
```

---

## Configuration

The following configuration values are set in `app/api/dependencies.py`:

| Variable | Value | Description |
|----------|-------|-------------|
| `OLLAMA_BASE_URL` | `http://192.168.0.247:11434` | Ollama server address |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Model for creating embeddings |
| `LLM_MODEL` | `gemma3:1b` | Model for answer generation |
| `QDRANT_HOST` | `192.168.0.247` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `VECTOR_SIZE` | `768` | Number of dimensions in vectors |
| `COLLECTION_NAME` | `rag-documents` | Qdrant collection name |
| `CHUNK_SIZE` | `200` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap characters between chunks |

---

## Installation

1. **Prerequisites**:
   - Python 3.11+
   - Ollama running with `nomic-embed-text:latest` and `gemma3:1b` models
   - Qdrant running at `192.168.0.247:6333`

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull Ollama models** (if not already done):
   ```bash
   ollama pull nomic-embed-text:latest
   ollama pull gemma3:1b
   ```

---

## Running

```bash
python main.py
```

Or:

```bash
uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## Testing

### Run all tests:

```bash
pytest tests/
```

### Run specific test categories:

```bash
# RAG unit tests
pytest tests/rag/

# Retrieval tests
pytest tests/retrieval/

# Generation tests
pytest tests/generation/

# Integration tests (API + RAG pipeline)
pytest tests/integration/

# Evaluation tests
pytest tests/evaluation/
```

### Test files:

| File | What it tests |
|------|---------------|
| `tests/rag/test_rag.py` | RAG pipeline (ask, ask_with_sources, ask_with_trace) |
| `tests/rag/test_document_indexer.py` | Document indexing (chunk → embed → store) |
| `tests/rag/test_context_ranker.py` | Context ranking |
| `tests/rag/test_context_builder.py` | Context building from results |
| `tests/retrieval/test_retriever.py` | Full retrieval pipeline |
| `tests/retrieval/test_ollama_embedder.py` | Ollama embedding API |
| `tests/retrieval/test_vector_search.py` | Vector store operations |
| `tests/integration/test_api.py` | API endpoints (health, documents, query) |
| `tests/integration/test_rag.py` | RAG end-to-end with real services |
