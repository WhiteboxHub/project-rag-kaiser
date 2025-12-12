# app/ingestion/doc_loader.py
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Tuple, Optional
from pypdf import PdfReader

logger = logging.getLogger(__name__)

try:
    # docling may be installed but with a different API; guard imports.
    from docling import DocumentLoader as DoclingLoader  
except Exception:
    DoclingLoader = None  

from app.schemas.ingestion import DocumentMetadata


def _extract_text_from_pdf_path(path: Path) -> list[tuple[int, str]]:
    """Extract text from PDF with page numbers."""
    try:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
        return pages
    except Exception:
        logger.exception("Failed to extract text from PDF using pypdf for %s", path)
        return []


def _extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        logger.exception("Failed to extract text from PDF bytes")
        return ""


def load_document_from_url(url: str) -> Tuple[list[tuple[int, str]], DocumentMetadata]:
    """
    Load document and return page-indexed text.
    Returns: ([(page_num, page_text), ...], metadata)
    """
    page_texts: list[tuple[int, str]] = []
    metadata: Optional[DocumentMetadata] = None

    # Try using docling first if available
    if DoclingLoader is not None:
        try:
            dl = DoclingLoader()
            # Try common loader method names
            if hasattr(dl, "load_from_url"):
                doc = dl.load_from_url(url)
            elif hasattr(dl, "load"):
                doc = dl.load(url)
            else:
                doc = dl  # last resort

            # Extract text from returned object
            if hasattr(doc, "get_text") and callable(getattr(doc, "get_text")):
                text = doc.get_text()
            elif hasattr(doc, "text"):
                text = getattr(doc, "text")
            elif hasattr(doc, "content"):
                text = getattr(doc, "content")
            elif hasattr(doc, "export_to_text"):
                text = doc.export_to_text()
            else:
                text = str(doc)

            # Docling doesn't give us page numbers, so treat as single page
            page_texts = [(1, text)] if text else []

            metadata = DocumentMetadata(
                source_url=None if not str(url).startswith("http") else url,
                document_type=getattr(doc, "doc_type", "policy"),
                title=getattr(doc, "title", None),
            )
            logger.info("Loaded document via docling from %s; extracted %d pages", url, len(page_texts))
            return page_texts, metadata
        except Exception:
            logger.exception("docling loader failed, falling back to lightweight loader for %s", url)

    # Fallback: treat url as local path if possible
    p = Path(url)
    if p.exists():
        try:
            if p.suffix.lower() == ".pdf":
                page_texts = _extract_text_from_pdf_path(p)
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
                page_texts = [(1, text)]
        except Exception:
            logger.exception("Error reading local file %s", p)
            page_texts = []

        metadata = DocumentMetadata(
            source_url=None,
            document_type="policy",
            title=p.name,
        )
        total_chars = sum(len(text) for _, text in page_texts)
        logger.info("Loaded local document %s; extracted %d pages, %d chars", p, len(page_texts), total_chars)
        return page_texts, metadata

    # Fallback for remote HTTP(S) when docling isn't available
    if str(url).startswith("http"):
        try:
            import requests

            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                # Can't easily get page numbers from bytes, so treat as single page
                text = _extract_text_from_pdf_bytes(resp.content)
                page_texts = [(1, text)]
            else:
                page_texts = [(1, resp.text)]
        except Exception:
            logger.exception("Failed to fetch or parse remote URL %s", url)
            page_texts = []

        metadata = DocumentMetadata(
            source_url=url,
            document_type="policy",
            title=None,
        )
        total_chars = sum(len(text) for _, text in page_texts)
        logger.info("Loaded remote document %s; extracted %d pages, %d chars", url, len(page_texts), total_chars)
        return page_texts, metadata

    # Nothing worked
    logger.warning("Could not load document from %s", url)
    metadata = DocumentMetadata(source_url=None, document_type="policy", title=None)
    return [], metadata
