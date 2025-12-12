import re
from typing import List, Tuple, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings


class MetadataChunker:
    """Chunks text while preserving page numbers and extracting metadata."""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "]
        )
        
    def chunk_with_metadata(
        self, 
        page_texts: List[Tuple[int, str]], 
        source_file: str
    ) -> List[Tuple[str, Dict]]:
        chunks_with_metadata = []
        
        for page_num, page_text in page_texts:
            if not page_text.strip():
                continue
                
            # Split page into chunks
            page_chunks = self.splitter.split_text(page_text)
            
            for chunk in page_chunks:
                # Extract chapter/section info
                chapter = self._extract_chapter(chunk)
                section = self._extract_section(chunk)
                
                metadata = {
                    "source_file": source_file,
                    "page": page_num,
                    "chapter": chapter,
                    "section": section,
                }
                
                chunks_with_metadata.append((chunk, metadata))
        
        return chunks_with_metadata
    
    def _extract_chapter(self, text: str) -> str:
        """Extract chapter number from text using regex patterns."""
        # Pattern 1: "Chapter 12" or "chapter 12"
        match = re.search(r'[Cc]hapter\s+(\d+)', text[:500])
        if match:
            return match.group(1)
        
        # Pattern 2: Section numbers like "12.1" at start of text
        match = re.search(r'^(\d+)\.\d+', text.strip())
        if match:
            return match.group(1)
            
        return ""
    
    def _extract_section(self, text: str) -> str:
        """Extract section title from text."""
        # Look for section headers in first 200 chars
        lines = text[:200].split('\n')
        for line in lines:
            line = line.strip()
            # Section headers are often short and may be title-cased
            if 3 < len(line) < 100 and (line.istitle() or line.isupper()):
                return line
        return ""
