"""Three ways to cut filings into retrievable pieces.

    fixed      800-character windows with 100 characters of overlap, ignoring page boundaries (the common default)
    page       pages kept intact, split only when a page is longer than the limit (a chunk never spans two pages)
    structure  pages split at blank lines into blocks (paragraphs, tables), blocks merged up to the limit, and each
               chunk prefixed with a header naming the company, filing and period ("contextual" chunks: a table of
               numbers on its own does not say which company or year it is about)

Every chunk records the page(s) it came from, so retrieval can be scored against FinanceBench's evidence pages.
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import Page

LIMIT = 1200        # characters (~300 tokens): well inside the 512-token window of the embedding model


@dataclass
class Chunk:
    id: int
    doc: str
    pages: tuple[int, ...]
    text: str            # what is embedded and searched
    display: str = ""    # what is shown to the answer model (same text without the header)


def _header(doc: str, meta: dict) -> str:
    m = meta.get(doc, {})
    return f"[{m.get('company', doc)} | {m.get('doc_type', '').upper()} | {m.get('period', '')}] "


def fixed(pages: list[Page], meta: dict, size: int = 800, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    by_doc: dict[str, list[Page]] = {}
    for p in pages:
        by_doc.setdefault(p.doc, []).append(p)
    for doc, ps in by_doc.items():
        text, owner = "", []          # owner[i] = page of character i
        for p in ps:
            t = p.text + "\n"
            text += t
            owner += [p.page] * len(t)
        step = size - overlap
        for start in range(0, max(len(text) - overlap, 1), step):
            piece = text[start:start + size]
            if piece.strip():
                pg = tuple(sorted(set(owner[start:start + len(piece)])))
                chunks.append(Chunk(len(chunks), doc, pg, piece, piece))
    return chunks


def page(pages: list[Page], meta: dict, limit: int = LIMIT) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in pages:
        for i in range(0, max(len(p.text), 1), limit):
            piece = p.text[i:i + limit]
            if piece.strip():
                chunks.append(Chunk(len(chunks), p.doc, (p.page,), piece, piece))
    return chunks


def structure(pages: list[Page], meta: dict, limit: int = LIMIT) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in pages:
        head = _header(p.doc, meta)
        blocks = [b.strip() for b in p.text.split("\n\n") if b.strip()]
        current = ""
        for b in blocks:
            while len(b) > limit:                       # a single huge block (long table): cut it
                if current:
                    chunks.append(Chunk(len(chunks), p.doc, (p.page,), head + current, current))
                    current = ""
                chunks.append(Chunk(len(chunks), p.doc, (p.page,), head + b[:limit], b[:limit]))
                b = b[limit:]
            if len(current) + len(b) + 2 > limit and current:
                chunks.append(Chunk(len(chunks), p.doc, (p.page,), head + current, current))
                current = ""
            current = f"{current}\n\n{b}" if current else b
        if current:
            chunks.append(Chunk(len(chunks), p.doc, (p.page,), head + current, current))
    return chunks


STRATEGIES = {"fixed": fixed, "page": page, "structure": structure}
