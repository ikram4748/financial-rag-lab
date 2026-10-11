"""Retrievers, all behind one interface: `search(query, allowed_docs) -> chunk ids, best first`.

    BM25      keyword matching (rank_bm25). Strong on exact terms: "capital expenditure", "FY2018", ticker names
    Dense     bge-small embeddings (fastembed, ONNX, CPU). Strong on paraphrase: "money spent on equipment"
    Hybrid    reciprocal rank fusion of the two (score = sum of 1 / (60 + rank)); no score calibration needed
    Rerank    a cross-encoder reads (query, chunk) pairs together and re-sorts the hybrid top 50: slower, more precise

Scope: retrieval over all 84 filings, or only over the filings of the company and year named in the question
(found by string matching, no LLM). Real document-QA systems filter by metadata like this before searching.
"""
from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .corpus import ROOT

CACHE = ROOT / ".cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _providers() -> list[str] | None:
    """ONNX_PROVIDERS=CUDAExecutionProvider (with `pip install fastembed-gpu`) runs embeddings and rerankers on a
    GPU; unset = CPU. The models and results are the same either way."""
    p = os.getenv("ONNX_PROVIDERS", "").strip()
    return [x.strip() for x in p.split(",") if x.strip()] or None


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL, cache_dir=str(CACHE / "models"), providers=_providers())


@lru_cache(maxsize=4)
def _cross_encoder(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(name, cache_dir=str(CACHE / "models"), providers=_providers())


class Index:
    """BM25 + dense vectors over one chunking of the corpus."""

    def __init__(self, name: str, chunks: list[Chunk]):
        self.name, self.chunks = name, chunks
        self.docs = np.array([c.doc for c in chunks])
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.vectors = self._vectors()

    def _vectors(self) -> np.ndarray:
        digest = hashlib.sha256("".join(c.text for c in self.chunks[:: max(len(self.chunks) // 500, 1)]).encode()).hexdigest()[:12]
        path = CACHE / f"vectors_{self.name}_{len(self.chunks)}_{digest}.npy"
        if path.exists():
            return np.load(path)
        CACHE.mkdir(exist_ok=True)
        v = np.array(list(_embedder().passage_embed([c.text for c in self.chunks], batch_size=64)), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        np.save(path, v)
        return v

    def _mask(self, allowed: set[str] | None) -> np.ndarray:
        return np.ones(len(self.chunks), bool) if not allowed else np.isin(self.docs, list(allowed))

    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query))

    def dense_scores(self, query: str) -> np.ndarray:
        q = np.array(next(iter(_embedder().query_embed([query]))), dtype=np.float32)
        return self.vectors @ (q / np.linalg.norm(q))

    def search(self, query: str, method: str, allowed: set[str] | None = None, k: int = 20,
               reranker: str | None = None, dense_query: str | None = None) -> list[int]:
        """`dense_query` lets HyDE embed a hypothetical answer while BM25 still matches the question."""
        mask = self._mask(allowed)

        def top(scores, n):
            s = np.where(mask, scores, -np.inf)
            idx = np.argpartition(-s, min(n, len(s) - 1))[:n]
            idx = idx[np.isfinite(s[idx])]
            return idx[np.argsort(-s[idx])].tolist()

        if method == "bm25":
            ranked = top(self.bm25_scores(query), max(k, 50))
        elif method == "dense":
            ranked = top(self.dense_scores(dense_query or query), max(k, 50))
        elif method == "hybrid":
            ranked = rrf([top(self.bm25_scores(query), 100), top(self.dense_scores(dense_query or query), 100)])
        else:
            raise ValueError(method)
        if reranker:
            cand = ranked[:50]
            scores = list(_cross_encoder(reranker).rerank(query, [self.chunks[i].text for i in cand], batch_size=16))
            ranked = [cand[i] for i in np.argsort(-np.array(scores))]
        return ranked[:k]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion (Cormack et al., 2009): robust to the two retrievers' scores being on different scales."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


# ---------------------------------------------------------------------------------------------------------------
# Metadata scope: which filings does the question talk about?
# ---------------------------------------------------------------------------------------------------------------

# How each FinanceBench company is written in questions (lower case). Default: the company name itself.
ALIASES = {"amd": ["amd", "advanced micro devices"], "3m": ["3m"], "american express": ["american express", "amex"],
           "jpmorgan": ["jpmorgan", "jp morgan", "jpm"], "johnson & johnson": ["johnson & johnson", "j&j", "jnj"],
           "mgm resorts": ["mgm"], "coca-cola": ["coca-cola", "coca cola", "coke"], "pepsico": ["pepsico", "pepsi"],
           "best buy": ["best buy", "bestbuy"], "cvs health": ["cvs"], "kraft heinz": ["kraft heinz", "kraft"],
           "ulta beauty": ["ulta"], "block": ["block", "square"], "activision blizzard": ["activision"],
           "aes corporation": ["aes"], "american water works": ["american water"], "lockheed martin": ["lockheed"]}


SCOPES = ["all", "company", "company+year"]


def scope(question: str, doc_meta: dict[str, dict], mode: str = "company+year") -> set[str] | None:
    """Filings of the company named in the question (mode "company"), narrowed to the year(s) named if any of them
    match (mode "company+year"). None = search everything (mode "all", or no company found)."""
    if mode == "all":
        return None
    q = question.lower()
    found = set()
    for doc, m in doc_meta.items():
        company = m["company"].lower()
        names = ALIASES.get(company, [company, company.replace(" ", "")])
        if any(re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", q) for n in names):
            found.add(doc)
    if not found:
        return None
    if mode == "company":
        return found
    years = set(re.findall(r"(20\d\d)", q))
    dated = {d for d in found if doc_meta[d]["period"] in years}
    return dated or found
