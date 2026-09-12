import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from ..schemas import IngestResponse, HealthResponse
from ...config import settings
from ...ingestion.pipeline import IngestionPipeline
from ...vectorstore.chroma_store import ChromaStore
from ...vectorstore.bm25_store import BM25Store

router = APIRouter(prefix="/api/v1", tags=["Ingest & Health"])


def _run_full_ingestion():
    pipeline = IngestionPipeline()
    result = pipeline.run()
    chunks = result["chunks"]
    
    chroma = ChromaStore()
    chroma.add_documents(chunks)
    
    bm25 = BM25Store()
    bm25.fit(chunks)


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(background_tasks: BackgroundTasks) -> IngestResponse:
    start_time = time.perf_counter()
    try:
        pipeline = IngestionPipeline()
        res = pipeline.run()
        
        chroma = ChromaStore()
        chroma.add_documents(res["chunks"])
        
        bm25 = BM25Store()
        bm25.fit(res["chunks"])
        
        duration = time.perf_counter() - start_time
        return IngestResponse(
            status="success",
            pages_count=len(res["pages"]),
            chapters_count=len(res["chapters"]),
            chunks_count=len(res["chunks"]),
            time_taken_sec=round(duration, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    try:
        chroma = ChromaStore()
        bm25 = BM25Store()
        return HealthResponse(
            status="healthy",
            chroma_chunks=chroma.count(),
            bm25_documents=bm25.count(),
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            llm_model=settings.LLM_MODEL,
        )
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            chroma_chunks=0,
            bm25_documents=0,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            llm_model=settings.LLM_MODEL,
        )
