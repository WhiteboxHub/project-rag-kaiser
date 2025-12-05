from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict


class DocumentMetadata(BaseModel):
    source_url: Optional[HttpUrl] = None
    document_type: str = "policy"
    title: Optional[str] = None
    region: Optional[str] = "CA"
    version: Optional[str] = None
    extra: Optional[dict] = None


class Chunk(BaseModel):
    text: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None


class IngestionDocument(BaseModel):
    source: str
    file_path: str
    # metadata: dict | None = None
    metadata: Optional[Dict] = None
