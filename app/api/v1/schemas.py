"""API request/response schemas."""
from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request body for RAG queries."""
    question: str
    top_k: int = 5  # Number of context chunks to retrieve


class RetrievalResult(BaseModel):
    """Single retrieved chunk with relevance score."""
    text: str
    score: float


class QueryResponse(BaseModel):
    """Response from RAG query."""
    question: str
    answer: str
    context: List[str]
    scores: List[float] = []
    num_chunks: int
    error: Optional[bool] = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
