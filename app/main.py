# project-rag-kaiser/app/main.py
import os
import logging
import warnings
from fastapi import FastAPI
from dotenv import load_dotenv
from urllib3.exceptions import NotOpenSSLWarning
from rag.query_pipeline import RAGPipeline 
from app.api.v1.router import router as v1_router
from app.core.logging_config import setup_logging

# Load .env file
load_dotenv()

# Suppress LibreSSL warning (development only)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kaiser RAG API",
    description="Retrieval-Augmented Generation API for Kaiser health insurance documents",
    version="1.0.0",
)


app.include_router(v1_router)


@app.on_event("startup")
async def startup_event():
    """Initialize global resources (RAG Pipeline)."""
     
    DEFAULT_TOP_K = 5
    try:
        top_k = int(os.getenv("RAG_TOP_K", DEFAULT_TOP_K))
    except ValueError:
        top_k = DEFAULT_TOP_K

    logger.info("Initializing RAG pipeline (top_k=%s)...", top_k)
    try:
        app.state.rag_pipeline = RAGPipeline(top_k=top_k)
        logger.info("RAG Pipeline initialized successfully.")
    except Exception:
        app.state.rag_pipeline = None
        logger.exception("Failed to initialize RAG Pipeline.")


@app.on_event("shutdown")
async def shutdown_event():
    pipeline = getattr(app.state, "rag_pipeline", None)
    if pipeline:
        try:
            close_fn = getattr(pipeline, "close", None)
            if callable(close_fn):
                close_fn()
            logger.info("RAG Pipeline shut down cleanly.")
        except Exception:
            logger.exception("Error during RAG Pipeline shutdown.")


@app.get("/")
async def root():
    """Root API endpoint."""
    return {
        "name": "Kaiser RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/v1/health"
    }
