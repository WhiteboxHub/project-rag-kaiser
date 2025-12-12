# project-rag-kaiser/rag/retriever.py
import logging
import re
from typing import List, Tuple, Optional
import os

try:
    from chromadb import PersistentClient
except Exception:
    PersistentClient = None

logger = logging.getLogger(__name__)


class Retriever:
    """Query the Chroma vector store for relevant chunks with metadata-enhanced retrieval."""

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

    def retrieve(self, query_embedding: List[float], query_text: str = "", top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Retrieve top-k most relevant chunks using hybrid search.
        
        Args:
            query_embedding: Vector embedding of the query
            query_text: Original query text for metadata extraction
            top_k: Number of results to return
            
        Returns:
            List of (text, score, metadata) tuples, sorted by relevance
        """
        if not self.enabled or self.collection is None:
            logger.warning("Retriever disabled — returning empty results")
            return []

        try:
            # Extract metadata filters from query
            metadata_filter = self._extract_metadata_filter(query_text)
            
            # Retrieve more candidates for reranking (2x top_k)
            n_results = min(top_k * 2, 50)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "distances", "metadatas"],
                where=metadata_filter if metadata_filter else None
            )

            if not results or not results.get("documents") or len(results["documents"]) == 0:
                logger.info("No results found for query")
                return []

            documents = results["documents"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            # Hybrid scoring: semantic + metadata bonus
            retrieved = []
            for doc, dist, meta in zip(documents, distances, metadatas):
                # Base semantic similarity (cosine)
                semantic_score = 1 - dist
                
                # Metadata bonus
                metadata_bonus = self._calculate_metadata_bonus(query_text, meta)
                
                # Hybrid score (70% semantic, 30% metadata)
                hybrid_score = 0.7 * semantic_score + 0.3 * metadata_bonus
                
                retrieved.append((doc, hybrid_score, meta))

            # Sort by hybrid score and return top-k
            retrieved.sort(key=lambda x: x[1], reverse=True)
            retrieved = retrieved[:top_k]

            if retrieved:
                logger.info("Retrieved %d documents (top score: %.3f)", len(retrieved), retrieved[0][1])
            return retrieved

        except Exception:
            logger.exception("Error retrieving documents from Chroma")
            return []
    
    def _extract_metadata_filter(self, query: str) -> Optional[dict]:
        """Extract metadata filters from query text."""
        # Look for chapter mentions
        match = re.search(r'[Cc]hapter\s+(\d+)', query)
        if match:
            return {"chapter": match.group(1)}
        
        # Look for page mentions
        match = re.search(r'[Pp]age\s+(\d+)', query)
        if match:
            return {"page": int(match.group(1))}
        
        return None
    
    def _calculate_metadata_bonus(self, query: str, metadata: dict) -> float:
        """Calculate metadata match bonus (0.0 to 1.0)."""
        bonus = 0.0
        query_lower = query.lower()
        
        # Chapter match bonus
        if metadata.get("chapter"):
            chapter_num = metadata["chapter"]
            if f"chapter {chapter_num}" in query_lower or f"chapter{chapter_num}" in query_lower:
                bonus += 0.5
        
        # Source file match bonus
        if metadata.get("source_file"):
            source = metadata["source_file"].lower()
            # Check if any part of source filename is in query
            source_parts = source.replace("-", " ").replace("_", " ").split()
            for part in source_parts:
                if len(part) > 3 and part in query_lower:
                    bonus += 0.3
                    break
        
        # Page match bonus
        if metadata.get("page"):
            page_num = metadata["page"]
            if f"page {page_num}" in query_lower or f"page{page_num}" in query_lower:
                bonus += 0.2
        
        return min(bonus, 1.0)  # Cap at 1.0
