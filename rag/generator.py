"""LLM-based response generator."""
import logging
from typing import List
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class Generator:
    """Generate answers using LLM with retrieved context."""

    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2  # Lower temperature for more factual responses
        )
        logger.info("Generator initialized with model: %s", model)

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate an answer based on query and retrieved context.
        
        Args:
            query: User's question
            context_chunks: List of relevant document chunks
            
        Returns:
            Generated response string
        """
        if not context_chunks:
            return "I don't have enough information to answer your question."

        # Build context string
        context = "\n\n".join([f"[Context {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)])

        # Create the prompt
        prompt = f"""You are a helpful assistant answering questions about Kaiser health insurance policies and member guides.

Use the following context to answer the question. If the answer is not in the context, say so clearly.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            logger.info("Generated response for query: %s", query[:50])
            return answer
        except Exception:
            logger.exception("Error generating response")
            return "I encountered an error while generating a response. Please try again."
