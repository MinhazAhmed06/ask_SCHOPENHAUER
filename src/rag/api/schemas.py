from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str
    section: str
    chapter: str
    book_page: str
    pdf_page: str
    snippet: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="User query or philosophical question", min_length=2)
    top_k: int = Field(default=4, ge=1, le=10, description="Number of source chunks to utilize")
    enable_cache: bool = Field(default=True, description="Whether to check/update the semantic cache")


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    is_grounded: bool = True
    grounding_explanation: str = ""
    tokens_used: int = 0
    cached: bool = False
    latency_ms: float = 0.0


class IngestResponse(BaseModel):
    status: str
    pages_count: int
    chapters_count: int
    chunks_count: int
    time_taken_sec: float


class HealthResponse(BaseModel):
    status: str
    chroma_chunks: int
    bm25_documents: int
    embedding_model: str
    llm_model: str
