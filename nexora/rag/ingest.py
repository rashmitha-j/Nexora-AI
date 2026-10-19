"""Turn uploaded files into page-aware text chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".rst"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


class UnsupportedFileError(ValueError):
    pass


@dataclass
class Section:
    text: str
    page: int | None = None


@dataclass
class Chunk:
    index: int
    text: str
    page: int | None = None


# Control characters other than tab, newline and carriage return. PDF text extraction often
# yields NUL (\x00), which PostgreSQL refuses to store in text columns.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    return _CONTROL.sub("", text)


def extract_sections(path: Path, filename: str) -> list[Section]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(
            f"{filename}: unsupported type. Upload one of: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    if suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            sections = [
                Section(text=clean_text(page.extract_text() or ""), page=number)
                for number, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:  # pypdf raises many types for damaged or encrypted files
            raise UnsupportedFileError(f"{filename}: couldn't read this PDF (it may be damaged or encrypted).") from exc
    else:
        sections = [Section(text=clean_text(path.read_text(encoding="utf-8", errors="replace")))]
    sections = [s for s in sections if s.text.strip()]
    if not sections:
        raise UnsupportedFileError(f"{filename}: no extractable text (scanned PDFs need OCR first).")
    return sections


def chunk_sections(sections: list[Section], size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Split text into ~`size`-character chunks, preferring paragraph and sentence
    boundaries, with `overlap` characters carried between neighbouring chunks."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    for section in sections:
        text = section.text
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                window = text[start:end]
                for sep in ("\n\n", "\n", ". ", " "):
                    cut = window.rfind(sep)
                    if cut > size // 2:
                        end = start + cut + len(sep)
                        break
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(index=len(chunks), text=piece, page=section.page))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
