from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.query import router as query_router
from .routes.ingest import router as ingest_router

app = FastAPI(
    title="Production Arthur Schopenhauer RAG Service",
    description="Enterprise RAG system with Hybrid Search (BM25 + ChromaDB), Reciprocal Rank Fusion, Cross-Encoder Re-Ranking, LangGraph Self-RAG loops, and Semantic Caching.",
    version="1.0.0",
)

# Enable CORS for frontend applications (Streamlit, Next.js, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(ingest_router)
