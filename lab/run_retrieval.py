"""Retrieval grid: chunking x scope x retriever, scored on FinanceBench evidence pages. No LLM calls.

    python -m lab.run_retrieval            # all configurations (reranking makes this take ~30-60 min on CPU)
    python -m lab.run_retrieval --fast     # skip the cross-encoders

A retrieved chunk is a hit when it comes from an evidence page of the question. For each question and configuration
the rank of the first hit is stored (results/retrieval_ranks.csv), so recall@k and MRR can be computed for any k
and any split afterwards.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragl import chunking  # noqa: E402
from ragl.corpus import doc_meta, load_pages, load_questions  # noqa: E402
from ragl.retrieval import SCOPES, Index, scope  # noqa: E402

RESULTS = ROOT / "results"
RERANKERS = {"rerank-minilm": "Xenova/ms-marco-MiniLM-L-12-v2", "rerank-bge": "BAAI/bge-reranker-base"}
K = 20


def first_hit(ranked: list[int], index: Index, evidence: list[tuple[str, int]]) -> int | None:
    ev = set(evidence)
    for r, i in enumerate(ranked, 1):
        c = index.chunks[i]
        if any((c.doc, p) in ev for p in c.pages):
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--strategies", nargs="+", default=list(chunking.STRATEGIES))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "retrieval_ranks.csv"
    done = set(pd.read_csv(out)["config"]) if out.exists() else set()
    questions, pages, meta = load_questions(), load_pages(), doc_meta()
    methods = [("bm25", None), ("dense", None), ("hybrid", None)]
    if not args.fast:
        methods += [("hybrid", r) for r in RERANKERS]
    for name in args.strategies:
        t0 = time.time()
        chunks = chunking.STRATEGIES[name](pages, meta)
        index = Index(name, chunks)
        print(f"{name}: {len(chunks):,} chunks, indexed in {time.time() - t0:.0f}s")
        for sc in SCOPES:
            allowed = {q.id: scope(q.question, meta, sc) for q in questions}
            for method, rr in methods:
                config = f"{name}|{sc}|{method}{'+' + rr if rr else ''}"
                if config in done:
                    continue
                t1 = time.time()
                rows = []
                for q in questions:
                    ranked = index.search(q.question, method, allowed[q.id], K, RERANKERS.get(rr))
                    in_scope = allowed[q.id] is None or any(d in allowed[q.id] for d, _ in q.evidence)
                    rows.append({"config": config, "chunking": name, "scope": sc, "retriever": method + (f"+{rr}" if rr else ""),
                                 "qid": q.id, "first_hit": first_hit(ranked, index, q.evidence), "evidence_in_scope": in_scope,
                                 "seconds": None})
                secs = (time.time() - t1) / len(questions)
                for r in rows:
                    r["seconds"] = round(secs, 4)
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                hits5 = sum(1 for r in rows if r["first_hit"] and r["first_hit"] <= 5) / len(rows)
                print(f"  {config:45s} recall@5 {hits5:.0%}  ({secs * 1000:.0f} ms/query)")


if __name__ == "__main__":
    main()
