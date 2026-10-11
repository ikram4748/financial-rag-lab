"""Offline tests (no API key). Tests that need the downloaded filings are skipped if they are missing."""
from __future__ import annotations

import re

import pytest

from lab.run_generation import corrupt
from ragl import chunking
from ragl.corpus import RAW, Page, load_questions, split
from ragl.retrieval import rrf, scope, tokenize

META = {"3M_2018_10K": {"company": "3M", "period": "2018", "doc_type": "10k"},
        "3M_2022_10K": {"company": "3M", "period": "2022", "doc_type": "10k"},
        "AMD_2022_10K": {"company": "AMD", "period": "2022", "doc_type": "10k"}}


def pages():
    return [Page("3M_2018_10K", 0, "Intro text.\n\nCash flow table\nCapex 1,577\nDividends 3,193"),
            Page("3M_2018_10K", 1, "Risk factors. " * 120),
            Page("AMD_2022_10K", 0, "Revenue was $23.6 billion.")]


def test_splits_are_disjoint_and_cover_everything():
    qs = load_questions()
    dev, test = split(qs, "dev"), split(qs, "test")
    assert len(dev) == 50 and len(test) == 100 and not {q.id for q in dev} & {q.id for q in test}


@pytest.mark.parametrize("name", list(chunking.STRATEGIES))
def test_chunks_keep_page_provenance(name):
    chunks = chunking.STRATEGIES[name](pages(), META)
    assert all(c.text.strip() and c.pages for c in chunks)
    assert any(c.doc == "AMD_2022_10K" and c.pages == (0,) for c in chunks)
    if name != "fixed":
        assert all(len(c.pages) == 1 for c in chunks)                 # never spans two pages
        assert all(len(c.display) <= chunking.LIMIT for c in chunks)


def test_structure_chunks_carry_a_header():
    c = chunking.structure(pages(), META)[0]
    assert c.text.startswith("[3M | 10K | 2018]") and not c.display.startswith("[")


def test_fixed_chunks_can_span_pages():
    chunks = chunking.fixed(pages(), META, size=100, overlap=10)
    assert any(len(c.pages) == 2 for c in chunks)


def test_scope_by_company_and_year():
    assert scope("What was 3M's capex in FY2018?", META) == {"3M_2018_10K"}
    assert scope("What was 3M's capex?", META, "company") == {"3M_2018_10K", "3M_2022_10K"}
    assert scope("What was 3M's capex in FY2019?", META) == {"3M_2018_10K", "3M_2022_10K"}   # no 2019 filing: fall back
    assert scope("Advanced Micro Devices revenue?", META) == {"AMD_2022_10K"}
    assert scope("What drove the margin decline?", META) is None
    assert scope("3M's capex", META, "all") is None


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [3, 1, 4]])[:2] == [1, 3]


def test_tokenizer_keeps_numbers():
    assert tokenize("FY2018 capex of $1,577.5 million") == ["fy2018", "capex", "of", "1", "577.5", "million"]


def test_corrupt_changes_the_number_only():
    assert corrupt("$1577.00") == "$1813.55"
    assert corrupt("Revenue rose to 1,000 million") == "Revenue rose to 1,150 million"
    assert corrupt("No, it did not") is None


@pytest.mark.skipif(not (RAW / "pdfs").exists(), reason="filings not downloaded")
def test_evidence_pages_are_zero_based():
    import pymupdf

    def norm(t):
        return re.sub(r"\W+", "", t.lower())

    checked = 0
    for q in load_questions()[:40]:
        for (doc, pg), txt in zip(q.evidence, q.evidence_text):
            pdf = RAW / "pdfs" / f"{doc}.pdf"
            if pdf.exists() and txt.strip():
                with pymupdf.open(pdf) as d:
                    assert norm(txt)[:60] in norm(d[pg].get_text())
                checked += 1
    assert checked > 10
