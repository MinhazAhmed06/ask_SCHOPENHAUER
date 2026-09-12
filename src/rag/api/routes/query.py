import json
import time
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

from ..schemas import QueryRequest, QueryResponse, Citation
from ...config import settings
from ...graph.workflow import RAGWorkflow
from ...retrievers.hybrid import HybridRetriever
from ...retrievers.reranker import CrossEncoderReranker
from ...guardrails.token_budget import TokenBudgetManager
from ...cache.semantic_cache import SemanticCache

router = APIRouter(prefix="/api/v1", tags=["Query"])

# Singleton workflow instances
_workflow: RAGWorkflow | None = None
_hybrid_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_budgeter: TokenBudgetManager | None = None
_cache: SemanticCache | None = None
_streaming_llm: ChatOpenRouter | None = None


def get_workflow() -> RAGWorkflow:
    global _workflow
    if _workflow is None:
        _workflow = RAGWorkflow()
    return _workflow


def get_streaming_components():
    global _hybrid_retriever, _reranker, _budgeter, _cache, _streaming_llm
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
        _reranker = CrossEncoderReranker()
        _budgeter = TokenBudgetManager()
        _cache = SemanticCache()
        _streaming_llm = ChatOpenRouter(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENROUTER_API_KEY,
            streaming=True,
        )
    return _hybrid_retriever, _reranker, _budgeter, _cache, _streaming_llm


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    start_time = time.perf_counter()
    try:
        wf = get_workflow()
        result = wf.invoke(request.question)
        latency_ms = (time.perf_counter() - start_time) * 1000

        citations = [Citation(**c) for c in result.get("citations", [])]

        return QueryResponse(
            question=request.question,
            answer=result.get("answer", ""),
            citations=citations,
            is_grounded=result.get("is_grounded", True),
            grounding_explanation=result.get("grounding_explanation", ""),
            tokens_used=result.get("tokens_used", 0),
            cached=result.get("cached", False),
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint."""
    retriever, reranker, budgeter, cache, llm = get_streaming_components()

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Semantic Cache check
        cached_result = cache.get(request.question)
        if cached_result and request.enable_cache:
            yield f"data: {json.dumps({'type': 'content', 'token': cached_result.get('answer', '')})}\n\n"
            yield f"data: {json.dumps({'type': 'citations', 'citations': cached_result.get('citations', []), 'cached': True})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 2. Hybrid Retrieval + Reranking
        candidates = retriever.retrieve(request.question, k=settings.RETRIEVAL_TOP_K)
        reranked = reranker.rerank(request.question, candidates, top_k=request.top_k)
        docs = budgeter.fit_documents_to_budget(reranked)

        context_parts = []
        citations = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            cite_id = f"[{i}]"
            section = meta.get("section", "General")
            chapter = meta.get("chapter", "Unknown")
            book_page = meta.get("book_pages", meta.get("book_page", "N/A"))
            pdf_page = meta.get("pdf_pages", meta.get("pdf_page", "N/A"))
            context_parts.append(
                f"--- SOURCE {cite_id} (Section: {section}, Chapter: {chapter}, Book Page: {book_page}) ---\n{doc.page_content}"
            )
            citations.append({
                "source_id": cite_id,
                "section": section,
                "chapter": chapter,
                "book_page": book_page,
                "pdf_page": pdf_page,
                "snippet": doc.page_content[:200] + "...",
            })

        formatted_context = "\n\n".join(context_parts)
        prompt = ChatPromptTemplate.from_template(
            """You are an authoritative scholar on Arthur Schopenhauer's philosophical works.
Answer the user's question based STRICTLY and ONLY on the provided context passages and their header metadata.
Embed citation markers like [1], [2] at the end of sentences referring to specific sources.

Context Passages:
{context}

Question:
{question}

Scholarly Answer:"""
        )
        chain = prompt | llm | StrOutputParser()

        full_answer = ""
        for chunk in chain.stream({"context": formatted_context, "question": request.question}):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'content', 'token': chunk})}\n\n"

        yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'cached': False})}\n\n"
        yield "data: [DONE]\n\n"

        # Cache update
        if request.enable_cache and full_answer:
            cache.set(request.question, {
                "question": request.question,
                "answer": full_answer,
                "citations": citations,
                "is_grounded": True,
                "grounding_explanation": "Streamed response.",
                "tokens_used": budgeter.count_tokens(formatted_context) + budgeter.count_tokens(full_answer),
                "retry_count": 0,
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")
