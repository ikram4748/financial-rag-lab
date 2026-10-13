"""FinanceBench questions and filings, parsed into pages.

Each question has one to three pieces of evidence, each a page of one filing. Page numbers in FinanceBench are
0-based PDF page indices (checked in tests/test_corpus.py by finding the evidence text on the page).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


@dataclass
class Question:
    id: str
    company: str
    doc: str
    question: str
    answer: str
    justification: str
    qtype: str
    reasoning: str
    evidence: list[tuple[str, int]]          # (doc_name, page index)
    evidence_text: list[str] = field(default_factory=list)
    period: str = ""
    doc_type: str = ""


@dataclass
class Page:
    doc: str
    page: int
    text: str


def load_questions() -> list[Question]:
    out = []
    for line in (RAW / "financebench.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ev = r["evidence"]
        out.append(Question(
            id=r["financebench_id"], company=r["company"], doc=r["doc_name"], question=r["question"], answer=r["answer"],
            justification=r.get("justification") or "", qtype=r["question_type"], reasoning=r.get("question_reasoning") or "",
            evidence=[(e["doc_name"], int(e["evidence_page_num"])) for e in ev],
            evidence_text=[e.get("evidence_text", "") for e in ev], period=str(r.get("doc_period", "")),
            doc_type=r.get("doc_type", "")))
    return out


def split(questions: list[Question], name: str) -> list[Question]:
    """dev = every third question (50), used to choose the configuration; test = the other 100, reported."""
    ordered = sorted(questions, key=lambda q: q.id)
    dev = ordered[::3]
    if name == "dev":
        return dev
    ids = {q.id for q in dev}
    return [q for q in ordered if q.id not in ids] if name == "test" else ordered


def _clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


@lru_cache(maxsize=1)
def load_pages() -> list[Page]:
    """Text of every page of every filing (cached as JSONL after the first parse)."""
    cache = PROCESSED / "pages.jsonl"
    if cache.exists():
        return [Page(**json.loads(line)) for line in cache.read_text(encoding="utf-8").splitlines()]
    import pymupdf
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pages = []
    for pdf in sorted((RAW / "pdfs").glob("*.pdf")):
        with pymupdf.open(pdf) as d:
            for i, p in enumerate(d):
                pages.append(Page(pdf.stem, i, _clean(p.get_text("text"))))
    cache.write_text("\n".join(json.dumps(p.__dict__) for p in pages), encoding="utf-8")
    return pages


def doc_meta() -> dict[str, dict]:
    """doc name -> company, period, type, from the questions file (e.g. 3M_2018_10K -> 3M, 2018, 10k)."""
    meta = {}
    for q in load_questions():
        meta[q.doc] = {"company": q.company, "period": q.period, "doc_type": q.doc_type}
    return meta
