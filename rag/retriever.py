"""Vector store retriever - queries Chroma for relevant documents."""
import logging
from typing import List, Tuple, Optional
import os

try:
    from chromadb import PersistentClient
except Exception:
    PersistentClient = None

logger = logging.getLogger(__name__)


class Retriever:
    """Query the Chroma vector store for relevant chunks."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "data/embeddings/chroma")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "project_rag")
        self.enabled = False
        self.collection = None

        if PersistentClient is None:
            logger.warning("chromadb not installed — Retriever disabled")
            return

        try:
            self.client = PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self.enabled = True
            logger.info("Retriever initialized (collection=%s)", self.collection_name)
        except Exception:
            logger.exception("Failed to initialize Retriever")
            self.enabled = False

    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for a given query embedding.
        
        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            
        Returns:
            List of (text, distance) tuples, sorted by relevance
        """
        if not self.enabled or self.collection is None:
            logger.warning("Retriever disabled — returning empty results")
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )

            if not results or not results.get("documents") or len(results["documents"]) == 0:
                logger.info("No results found for query")
                return []

            documents = results["documents"][0]
            distances = results["distances"][0]

            retrieved = []
            for doc, dist in zip(documents, distances):
                # Cosine similarity = 1 - cosine_distance
                similarity = 1 - dist
                retrieved.append((doc, similarity))

            logger.info("Retrieved %d documents for query (top similarity: %.3f)", len(retrieved), retrieved[0][1])
            return retrieved

        except Exception:
            logger.exception("Error retrieving documents from Chroma")
            return []
