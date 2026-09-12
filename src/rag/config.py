from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the production RAG system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCS_DIR: Path = BASE_DIR / "docs"
    DEFAULT_PDF_PATH: Path = DOCS_DIR / "wisdomoflife01scho.pdf"
    
    # Vector DB & Storage
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "schopenhauer_knowledge"
    BM25_INDEX_PATH: Path = DATA_DIR / "bm25_index.pkl"
    CACHE_DB_PATH: Path = DATA_DIR / "semantic_cache.db"
    
    # Ingestion & Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 60
    SEPARATORS: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])

    # Embeddings & Re-ranking
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"  # or 'cuda' if GPU is available
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval Tuning
    RETRIEVAL_TOP_K: int = 10  # Initial candidates fetched from Chroma + BM25
    RERANK_TOP_K: int = 4      # Final re-ranked chunks sent to context
    RRF_K: int = 60            # Reciprocal Rank Fusion constant
    DENSE_WEIGHT: float = 0.6  # Weight for dense vector retrieval
    SPARSE_WEIGHT: float = 0.4 # Weight for sparse BM25 retrieval

    # LLM Settings (OpenRouter / Fallback)
    OPENROUTER_API_KEY: str | None = None
    LLM_MODEL: str = "nvidia/nemotron-3.5-lightning:free"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1500
    
    # Token Budget & Guardrails
    MAX_TOKENS_PER_REQUEST: int = 4000
    TOKEN_WARNING_THRESHOLD: int = 3200
    
    # Semantic Cache
    ENABLE_SEMANTIC_CACHE: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = 0.92

    # Self-RAG & Agentic Loop
    MAX_RETRY_LOOPS: int = 2
    MIN_DOC_RELEVANCE_SCORE: float = 0.4

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


# Global settings singleton instance
settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
