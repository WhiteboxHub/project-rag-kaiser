# project-rag-kaiser/app/api/v1/router.py
import logging
from fastapi import APIRouter, HTTPException, Request
from app.api.v1.schemas import QueryRequest, QueryResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Rag"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Check API health and RAG pipeline status."""
    rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
    status = "healthy" if rag_pipeline else "degraded"
    return HealthResponse(
        status=status,
        message="RAG API is running" if rag_pipeline else "RAG Pipeline not initialized"
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest):
    """
    Query the RAG system with a question.

    The RAG pipeline instance is retrieved from app.state (set during startup).
    """
    rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
    if not rag_pipeline:
        logger.error("Query attempted but RAG Pipeline not initialized")
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    if not payload.question or len(payload.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag_pipeline.query(payload.question, top_k=payload.top_k if getattr(payload, "top_k", None) else None)
        # If pipeline.query returns keys matching the QueryResponse schema, this will validate and return it.
        return QueryResponse(**result)
    except Exception:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail="Error processing your query")
