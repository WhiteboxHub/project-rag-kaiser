from __future__ import annotations

import os
import logging
import uuid
from typing import List

try:
    from chromadb import PersistentClient
except Exception: 
    PersistentClient = None 

logger = logging.getLogger(__name__)


class ChromaClient:
    def __init__(self, collection_name: str | None = None, persist_dir: str | None = None):
        self.enabled = False
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "data/embeddings/chroma")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "project_rag")

        if PersistentClient is None:
            logger.warning("chromadb not installed — Chroma client disabled")
            return

        try:
       
            self.client = PersistentClient(path=self.persist_dir)
    
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self.enabled = True
            logger.info("Chroma client initialized (persist_dir=%s, collection=%s)", self.persist_dir, self.collection_name)
        except Exception:
            logger.exception("Failed to initialize Chroma client; Chroma disabled")
            self.enabled = False

    def insert(self, embeddings: List[List[float]], chunks: List[str], source: str):

        if not self.enabled:
            logger.info("Chroma client disabled — skipping insert of %d chunks", len(chunks))
            return

        if not chunks:
            logger.warning("No chunks to insert into Chroma")
            return

        try:
            metadatas = [{"source": source} for _ in chunks]
            
            ids = [str(uuid.uuid4()) for _ in chunks]
            
            self.collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
          
            try:
                self.client.persist()
            except Exception:
                pass
            logger.info("Inserted %d chunks into Chroma collection '%s'", len(chunks), self.collection_name)
        except Exception:
            logger.exception("Failed to insert into Chroma collection %s", self.collection_name)
