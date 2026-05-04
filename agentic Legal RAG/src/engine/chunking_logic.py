from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(chunk_size - overlap, 1)
    for offset in range(0, len(words), step):
        chunk = " ".join(words[offset : offset + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_chunks_from_pdf(pdf_path: str, source_name: Optional[str] = None) -> List[Dict[str, Optional[str]]]:
    source = source_name or Path(pdf_path).name
    page_texts = extract_text_from_pdf(pdf_path)
    chunks: List[Dict[str, Optional[str]]] = []

    for page_number, page_text in enumerate(page_texts, start=1):
        for chunk_index, chunk in enumerate(chunk_text(page_text), start=1):
            chunks.append(
                {
                    "content": chunk,
                    "source": source,
                    "page": page_number,
                    "article_ref": f"{source}-page-{page_number}-chunk-{chunk_index}",
                }
            )

    return chunks
