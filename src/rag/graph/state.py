from typing import Annotated, Sequence, TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    """Execution state passed between nodes in the LangGraph workflow."""
    question: str
    transformed_query: str
    documents: list[Document]
    relevance_scores: list[float]
    generation: str
    citations: list[dict]
    retry_count: int
    needs_rewrite: bool
    is_grounded: bool
    grounding_explanation: str
    cached: bool
    tokens_used: int
