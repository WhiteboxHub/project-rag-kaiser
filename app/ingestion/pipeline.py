from pathlib import Path
from app.ingestion.doc_loader import load_document_from_url
from app.ingestion.docling_processor import DoclingProcessor
from app.ingestion.metadata_chunker import MetadataChunker
from app.ingestion.embedder import Embedder
from app.ingestion.chroma_client import ChromaClient
from app.schemas.ingestion import IngestionDocument


class IngestionPipeline:
    def __init__(self):
        self.chunker = MetadataChunker()
        self.embedder = Embedder()
        self.store = ChromaClient()

    def ingest(self, doc: IngestionDocument):
        page_texts, metadata = load_document_from_url(doc.file_path)
        source_file = Path(doc.file_path).name if doc.file_path else "unknown"
    
        cleaned_pages = []
        for page_num, text in page_texts:
            cleaned_text = DoclingProcessor.clean_text(text)
            cleaned_pages.append((page_num, cleaned_text))
        
        # Chunk with metadata
        chunks_with_metadata = self.chunker.chunk_with_metadata(cleaned_pages, source_file)
        
        # Separate chunks and metadata
        chunks = [chunk for chunk, _ in chunks_with_metadata]
        metadatas = [meta for _, meta in chunks_with_metadata]
        embeddings = self.embedder.embed(chunks)
        self.store.insert(embeddings, chunks, metadatas)

        return {
            "chunks": len(chunks),
            "status": "success",
            "source": doc.source,
        }
