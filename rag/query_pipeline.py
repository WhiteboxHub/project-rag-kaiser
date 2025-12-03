"""RAG query pipeline - orchestrates retrieval and generation."""
import logging
from typing import List
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from rag.retriever import Retriever
from rag.generator import Generator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: Query -> Embed -> Retrieve -> Generate."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        self.retriever = Retriever()
        self.generator = Generator()
        logger.info("RAG Pipeline initialized (top_k=%d)", top_k)

    def query(self, question: str) -> dict:
        """
        Execute the complete RAG pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with question, context chunks, and answer
        """
        try:
            # Step 1: Embed the query
            logger.info("Embedding query: %s", question[:50])
            query_embedding = self.embeddings.embed_query(question)

            # Step 2: Retrieve relevant chunks
            logger.info("Retrieving top-%d chunks", self.top_k)
            retrieved = self.retriever.retrieve(query_embedding, top_k=self.top_k)

            if not retrieved:
                logger.warning("No documents retrieved for query")
                return {
                    "question": question,
                    "context": [],
                    "answer": "I couldn't find relevant information to answer your question.",
                    "num_chunks": 0
                }

            context_chunks = [chunk for chunk, _ in retrieved]
            scores = [score for _, score in retrieved]

            # Step 3: Generate answer
            logger.info("Generating answer based on %d retrieved chunks", len(context_chunks))
            answer = self.generator.generate(question, context_chunks)

            return {
                "question": question,
                "context": context_chunks,
                "scores": scores,
                "answer": answer,
                "num_chunks": len(context_chunks)
            }

        except Exception:
            logger.exception("Error in RAG pipeline")
            return {
                "question": question,
                "context": [],
                "answer": "An error occurred while processing your question.",
                "num_chunks": 0,
                "error": True
            }
