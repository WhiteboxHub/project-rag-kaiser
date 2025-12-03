# app/ingestion/doc_loader.py
"""Document loader wrapper with resilient fallbacks.

This module first tries to delegate to the `docling` package when available and
compatible. If `docling` is absent or doesn't expose the expected API, the
module falls back to a lightweight local implementation that can read local
text files and extract text from PDFs using `pypdf`.

The public API is `load_document_from_url(url: str) -> (text, metadata)` to
keep compatibility with the rest of the codebase.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    # docling may be installed but with a different API; guard imports.
    from docling import DocumentLoader as DoclingLoader  # type: ignore
except Exception:
    DoclingLoader = None  # type: ignore

from app.schemas.ingestion import DocumentMetadata


def _extract_text_from_pdf_path(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        logger.exception("Failed to extract text from PDF using pypdf for %s", path)
        return ""


def _extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(b))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        logger.exception("Failed to extract text from PDF bytes")
        return ""


def load_document_from_url(url: str) -> Tuple[str, DocumentMetadata]:
    """Load a document from a local path or URL and return (text, metadata).

    - If `docling` is available and exposes a compatible loader, delegate to it.
    - For local files, use a simple extractor (text files or PDF via pypdf).
    - For remote URLs, if docling is unavailable, attempt HTTP GET and inspect
      content-type; for PDFs use pypdf on the response bytes.
    """
    text: str = ""
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

            metadata = DocumentMetadata(
                source_url=None if not str(url).startswith("http") else url,
                document_type=getattr(doc, "doc_type", "policy"),
                title=getattr(doc, "title", None),
            )
            logger.info("Loaded document via docling from %s; extracted %d chars", url, len(text or ""))
            return (text or ""), metadata
        except Exception:
            logger.exception("docling loader failed, falling back to lightweight loader for %s", url)

    # Fallback: treat url as local path if possible
    p = Path(url)
    if p.exists():
        try:
            if p.suffix.lower() == ".pdf":
                text = _extract_text_from_pdf_path(p)
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logger.exception("Error reading local file %s", p)
            text = ""

        metadata = DocumentMetadata(
            source_url=None,
            document_type="policy",
            title=p.name,
        )
        logger.info("Loaded local document %s; extracted %d chars", p, len(text or ""))
        return (text or ""), metadata

    # Fallback for remote HTTP(S) when docling isn't available
    if str(url).startswith("http"):
        try:
            import requests

            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                text = _extract_text_from_pdf_bytes(resp.content)
            else:
                text = resp.text
        except Exception:
            logger.exception("Failed to fetch or parse remote URL %s", url)
            text = ""

        metadata = DocumentMetadata(
            source_url=url,
            document_type="policy",
            title=None,
        )
        logger.info("Loaded remote document %s; extracted %d chars", url, len(text or ""))
        return (text or ""), metadata

    # Nothing worked
    logger.warning("Could not load document from %s", url)
    metadata = DocumentMetadata(source_url=None, document_type="policy", title=None)
    return "", metadata
