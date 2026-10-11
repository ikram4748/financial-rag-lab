"""Answering from retrieved excerpts, HyDE, and grading answers against FinanceBench's gold answers."""
from __future__ import annotations

import dataclasses

from .chunking import Chunk
from .config import get_settings
from .llm import Gemini, parse_json

ANSWER_SYSTEM = """You are a financial analyst answering questions about company filings (10-K, 10-Q, 8-K, earnings).
Use only the excerpts provided. Each excerpt is labelled with its filing and page.
- Give the final answer first, with units (e.g. "$1,577 million", "24.5%"). Then one or two sentences of support,
  including any calculation you did.
- Follow the rounding and units the question asks for.
- If the excerpts do not contain what is needed, answer exactly "NOT FOUND" and say what is missing.
Reply with JSON only: {"answer": "<final answer and short support>"}"""

CLOSED_BOOK_SYSTEM = """You are a financial analyst. Answer from your own knowledge. Give the final answer first, with
units, then one sentence of support. If you do not know, answer "NOT FOUND".
Reply with JSON only: {"answer": "<final answer and short support>"}"""

HYDE_SYSTEM = """Write the passage of a company filing (10-K, 10-Q or earnings release) that would answer the question:
for a number, the line item or table row in the wording filings use, with a plausible value. Three sentences at most.
Do not mention that it is hypothetical. Reply with JSON only: {"passage": "<text>"}"""

JUDGE_SYSTEM = """You grade answers to financial questions against a reference answer.
Correct if the candidate's final answer matches the reference in substance: numbers equal after rounding to the
precision the question asks for (or within 1% when no precision is given), same units and sign; for explanations,
the same conclusion and key facts. Extra correct detail is fine. A refusal, "NOT FOUND", a different number, or a
hedge between several answers is incorrect.
Reply with JSON only: {"correct": true | false, "reason": "<one sentence>"}"""


def context_block(chunks: list[Chunk], meta: dict) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = meta.get(c.doc, {})
        pages = ", ".join(str(p + 1) for p in c.pages)
        parts.append(f"[{i}] {m.get('company', c.doc)} {m.get('doc_type', '').upper()} {m.get('period', '')}, "
                     f"page {pages}:\n{c.display or c.text}")
    return "\n\n".join(parts)


class Generator:
    def __init__(self, llm: Gemini | None = None, judge: Gemini | None = None):
        s = get_settings()
        self.llm = llm or Gemini(s)
        self.judge_llm = judge or Gemini(dataclasses.replace(s, models=s.judge_models))

    def answer(self, question: str, chunks: list[Chunk] | None, meta: dict) -> str:
        if chunks is None:
            r = self.llm.generate(f"Question: {question}", CLOSED_BOOK_SYSTEM, json_mode=True)
        else:
            r = self.llm.generate(f"Excerpts:\n\n{context_block(chunks, meta)}\n\nQuestion: {question}", ANSWER_SYSTEM,
                                  json_mode=True)
        try:
            return str(parse_json(r.text).get("answer", r.text))
        except ValueError:
            return r.text

    def hyde(self, question: str) -> str:
        r = self.llm.generate(f"Question: {question}", HYDE_SYSTEM, json_mode=True, thinking="minimal")
        try:
            return str(parse_json(r.text).get("passage", ""))
        except ValueError:
            return ""

    def grade(self, question: str, reference: str, justification: str, candidate: str) -> tuple[bool, str]:
        prompt = (f"Question: {question}\n\nReference answer: {reference}\n"
                  + (f"Reference justification: {justification}\n" if justification else "")
                  + f"\nCandidate answer: {candidate}")
        r = self.judge_llm.generate(prompt, JUDGE_SYSTEM, json_mode=True, thinking="minimal")
        try:
            d = parse_json(r.text)
            return bool(d.get("correct")), str(d.get("reason", ""))
        except ValueError:
            return False, f"unparseable judge reply: {r.text[:100]}"
