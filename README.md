# 📜 Production Arthur Schopenhauer RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system built over Arthur Schopenhauer's philosophical works (*The Wisdom of Life* and *Essays*, 1851). Engineered for high precision, hallucination prevention, low latency, token economics, and rigorous observability.

---

## 🏛️ System Architecture

```
                                  [ User Query ]
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │  Semantic Cache Check  │ ──(Hit)──► [ Instant 0ms Answer ]
                            └────────────────────────┘
                                         │ (Miss)
                                         ▼
                            ┌────────────────────────┐
                            │   Query Router Node    │
                            └────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
        ┌───────────────────────┐                 ┌───────────────────────┐
        │  Dense Vector Search  │                 │  Sparse Lexical (BM25)│
        │      (ChromaDB)       │                 │       Retriever       │
        └───────────────────────┘                 └───────────────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                            ┌────────────────────────┐
                            │ Reciprocal Rank Fusion │
                            │       (RRF Merge)      │
                            └────────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │  Cross-Encoder Rerank  │
                            │ (ms-marco-MiniLM-L-6)  │
                            └────────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ Document Grader (CRAG) │ ──(Low Signal)──► [ Query Rewriter ]
                            └────────────────────────┘                                 │
                                         │ (High Signal)                               │
                                         ▼                                             │
                            ┌────────────────────────┐                                 │
                            │ Token Budget Optimizer │ ◄───────────────────────────────┘
                            └────────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ LLM Answer Generation  │
                            │  with Exact Citations  │
                            └────────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ Self-RAG Grounding &   │
                            │ Faithfulness Verifier  │
                            └────────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ Update Semantic Cache  │
                            └────────────────────────┘
                                         │
                                         ▼
                            [ Scholarly Response + Citations ]
```

---

## ✨ Core Features & Technical Highlights

1. **Robust Ingestion & OCR Cleaning (`src/rag/ingestion/`)**:
   - Dynamic top header and bottom footer coordinate detection (filters out running titles and page numbers without line truncation).
   - Targeted OCR quote and symbol normalization (`<(`, `))`, `«`, `»`, `®` $\rightarrow$ `"`).
   - Cross-page and intra-page hyphenated word rejoining (e.g., `suf-\nfered` $\rightarrow$ `suffered`).
   - Paragraph indentation reconstruction and Small-to-Big (Parent-Document) chunking.

2. **Hybrid Search with Reciprocal Rank Fusion (`src/rag/retrievers/hybrid.py`)**:
   - Fuses Dense Vector Embeddings (`BAAI/bge-small-en-v1.5` in persistent `ChromaDB`) and Sparse Lexical search (`rank-bm25`).
   - Reciprocal Rank Fusion formula:
     $$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

3. **Cross-Encoder Deep Re-Ranking (`src/rag/retrievers/reranker.py`)**:
   - Evaluates `(query, document)` pairs simultaneously through cross-attention transformer layers (`cross-encoder/ms-marco-MiniLM-L-6-v2`), eliminating vector cosine blind spots.

4. **Agentic Self-Correcting RAG (`src/rag/graph/`)**:
   - Built with **LangGraph** StateGraph.
   - Includes **Self-RAG** grounding verification to catch hallucinations and **Corrective RAG (CRAG)** query rewriting loops if retrieved context is weak.

5. **Token Budgeting & Semantic Caching (`src/rag/guardrails/`, `src/rag/cache/`)**:
   - Dynamic prompt trimming ensures context fits within target token limits.
   - SQLite-backed semantic cache computes cosine similarity ($> 0.92$) on incoming queries to deliver instant responses at $\$0$ LLM cost.

6. **Production REST API & Streaming (`src/rag/api/`)**:
   - Asynchronous FastAPI server.
   - `POST /api/v1/query`: Standard JSON with citations, latency, and tokens.
   - `POST /api/v1/query/stream`: Server-Sent Events (SSE) token-by-token streaming.
   - `GET /api/v1/health`: System health and collection metrics.

7. **Interactive UI Playground (`src/rag/ui/app.py`)**:
   - Streamlit interface with chat history, side-by-side search comparisons (Dense vs BM25 vs Hybrid vs Reranked), citation cards, and latency metrics.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Install dependencies using uv
uv sync
```

Ensure your `.env` file contains your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Document Ingestion & Indexing
```bash
uv run python -c "from rag.ingestion import IngestionPipeline; IngestionPipeline().run()"
```

### 3. Start the FastAPI Production Server
```bash
uv run uvicorn rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs will be available at: `http://localhost:8000/docs`.

### 4. Launch the Streamlit Playground UI
```bash
uv run streamlit run src/rag/ui/app.py
```

### 5. Run Automated Tests
```bash
uv run pytest
```

### 6. Run Benchmark Evaluation
```bash
uv run python -c "from rag.evaluation import RAGEvaluator; print(RAGEvaluator().run_benchmark())"
```
