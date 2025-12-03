"""API v1 routes."""
import logging
from fastapi import APIRouter, HTTPException
from app.api.v1.schemas import QueryRequest, QueryResponse, HealthResponse
from rag.query_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["rag"])

# Initialize RAG pipeline globally
try:
    rag_pipeline = RAGPipeline(top_k=5)
    logger.info("RAG Pipeline loaded successfully")
except Exception as e:
    logger.error("Failed to load RAG Pipeline: %s", e)
    rag_pipeline = None


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and RAG pipeline status."""
    status = "healthy" if rag_pipeline else "degraded"
    return HealthResponse(
        status=status,
        message="RAG API is running" if rag_pipeline else "RAG Pipeline not initialized"
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Returns relevant context chunks and a generated answer.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag_pipeline.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail="Error processing your query")
