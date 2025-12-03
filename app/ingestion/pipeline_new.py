from app.ingestion.doc_loader import load_document_from_url
from app.ingestion.docling_processor import DoclingProcessor
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import Embedder
from app.ingestion.chroma_client import ChromaClient
from app.schemas.ingestion import IngestionDocument


class IngestionPipeline:
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        # Use a Chroma-backed local vector store by default
        self.store = ChromaClient()

    def ingest(self, doc: IngestionDocument):
        # Step 1: Load
        text, _metadata = load_document_from_url(doc.file_path)

        # Step 2: Clean using Docling
        cleaned_text = DoclingProcessor.clean_text(text)

        # Step 3: Chunk
        chunks = self.chunker.chunk(cleaned_text)

        # Step 4: Embed
        embeddings = self.embedder.embed(chunks)

        # Step 5: Store in vector store (Chroma)
        self.store.insert(embeddings, chunks, doc.source)

        return {
            "chunks": len(chunks),
            "status": "success",
            "source": doc.source,
        }
