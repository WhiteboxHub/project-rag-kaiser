"""FastAPI application."""
import logging
from fastapi import FastAPI
from app.api.v1.router import router as v1_router
from app.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Kaiser RAG API",
    description="Retrieval-Augmented Generation API for Kaiser health insurance documents",
    version="1.0.0"
)

# Include v1 routes
app.include_router(v1_router)

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Kaiser RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Kaiser RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
