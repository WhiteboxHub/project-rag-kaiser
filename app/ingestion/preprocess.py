# ingestion/preprocess.py
import re
from typing import Tuple

def normalize_text(text: str) -> str:
    if not text:
        return ""
    txt = text
    # Normalize line endings
    txt = re.sub(r'\r\n', '\n', txt)
    # Collapse long runs of newlines to double newline
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    # Replace many spaces/tabs with a single space
    txt = re.sub(r'[ \t]{2,}', ' ', txt)
    # Remove common "Page X" footers (heuristic)
    txt = re.sub(r'^\s*page\s*\d+\s*$', '', txt, flags=re.I | re.M)
    txt = txt.strip()
    return txt
