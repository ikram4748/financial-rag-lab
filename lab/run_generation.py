"""End-to-end answers on the test split, graded against FinanceBench's gold answers.

    python -m lab.run_generation hyde          # HyDE passages for every question (retrieval metric, dev + test)
    python -m lab.run_generation answer        # answers for each variant on the test split
    python -m lab.run_generation judge-check   # how reliable is the judge? (gold answers vs corrupted ones)

Variants (all use the same answer model and the same 5-excerpt budget):
    closed_book  no retrieval: what the model "knows"
    naive        fixed chunks, dense retrieval over all filings (the usual tutorial setup)
    best         the configuration with the highest recall@5 on the dev split
    best_hyde    the same, with HyDE: the dense search embeds a hypothetical answer passage instead of the question
    oracle       the evidence pages themselves (upper bound: perfect retrieval)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragl import chunking  # noqa: E402
from ragl.chunking import Chunk  # noqa: E402
from ragl.corpus import doc_meta, load_pages, load_questions, split  # noqa: E402
from ragl.generation import Generator  # noqa: E402
from ragl.llm import LLMUnavailable  # noqa: E402
from ragl.retrieval import Index, scope  # noqa: E402

RESULTS = ROOT / "results"
GEN = RESULTS / "generation"
TOP_K = 5
RERANKERS = {"rerank-minilm": "Xenova/ms-marco-MiniLM-L-12-v2", "rerank-bge": "BAAI/bge-reranker-base"}


def best_config() -> dict:
    """Highest recall@5 on the dev split (ties: higher MRR, then faster)."""
    ranks = pd.read_csv(RESULTS / "retrieval_ranks.csv")
    dev = {q.id for q in split(load_questions(), "dev")}
    d = ranks[ranks.qid.isin(dev)].copy()
    d["hit5"] = d.first_hit.fillna(99) <= 5
    d["rr"] = 1 / d.first_hit.fillna(1e9)
    agg = d.groupby("config").agg(recall5=("hit5", "mean"), mrr=("rr", "mean"), seconds=("seconds", "mean"))
    best = agg.sort_values(["recall5", "mrr", "seconds"], ascending=[False, False, True]).index[0]
    chunk, sc, retr = best.split("|")
    method, _, rr = retr.partition("+")
    return {"config": best, "chunking": chunk, "scope": sc, "method": method, "reranker": RERANKERS.get(rr), **agg.loc[best].to_dict()}


def _jsonl(path: Path) -> dict[str, dict]:
    return {r["qid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else {}


def _append(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_hyde(gen: Generator, workers: int) -> None:
    path = GEN / "hyde_passages.jsonl"
    done, lock = _jsonl(path), threading.Lock()
    todo = [q for q in load_questions() if q.id not in done]
    print(f"HyDE: {len(done)} done, {len(todo)} to go")

    def one(q):
        _append(path, {"qid": q.id, "passage": gen.hyde(q.question)}, lock)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    # retrieval metric with HyDE on every split, for the best configuration
    cfg, meta, pages = best_config(), doc_meta(), load_pages()
    index = Index(cfg["chunking"], chunking.STRATEGIES[cfg["chunking"]](pages, meta))
    from lab.run_retrieval import first_hit
    passages = _jsonl(path)
    rows = []
    for q in load_questions():
        dq = f"{q.question}\n{passages[q.id]['passage']}" if q.id in passages else None
        ranked = index.search(q.question, cfg["method"], scope(q.question, meta, cfg["scope"]), 20, cfg["reranker"], dense_query=dq)
        rows.append({"config": cfg["config"] + "+hyde", "chunking": cfg["chunking"], "scope": cfg["scope"],
                     "retriever": cfg["config"].split("|")[2] + "+hyde", "qid": q.id, "first_hit": first_hit(ranked, index, q.evidence),
                     "evidence_in_scope": True, "seconds": None})
    out = RESULTS / "retrieval_ranks.csv"
    ranks = pd.read_csv(out)
    ranks = ranks[ranks.config != cfg["config"] + "+hyde"]
    pd.concat([ranks, pd.DataFrame(rows)]).to_csv(out, index=False)
    print("HyDE retrieval scored for", cfg["config"])


def run_answers(gen: Generator, workers: int, variants: list[str]) -> None:
    meta, pages = doc_meta(), load_pages()
    test = split(load_questions(), "test")
    cfg = best_config()
    print("best configuration on dev:", cfg["config"], f"recall@5 {cfg['recall5']:.0%}")
    page_text = {(p.doc, p.page): p.text for p in pages}
    indexes: dict[str, Index] = {}

    def index(name):
        if name not in indexes:
            indexes[name] = Index(name, chunking.STRATEGIES[name](pages, meta))
        return indexes[name]

    passages = _jsonl(GEN / "hyde_passages.jsonl")

    def contexts(variant, q) -> list[Chunk] | None:
        if variant == "closed_book":
            return None
        if variant == "oracle":
            return [Chunk(i, d, (p,), page_text.get((d, p), "")) for i, (d, p) in enumerate(q.evidence)]
        if variant == "naive":
            ix = index("fixed")
            return [ix.chunks[i] for i in ix.search(q.question, "dense", None, TOP_K)]
        ix = index(cfg["chunking"])
        dq = None
        if variant == "best_hyde":
            dq = f"{q.question}\n{passages.get(q.id, {}).get('passage', '')}"
        ids = ix.search(q.question, cfg["method"], scope(q.question, meta, cfg["scope"]), TOP_K, cfg["reranker"], dense_query=dq)
        return [ix.chunks[i] for i in ids]

    for variant in variants:
        path = GEN / f"{variant}.jsonl"
        done, lock = _jsonl(path), threading.Lock()
        todo = [q for q in test if q.id not in done]
        print(f"{variant}: {len(done)} done, {len(todo)} to go")
        ctx = {q.id: contexts(variant, q) for q in todo}      # retrieval first (CPU), then the API calls in parallel

        def one(q):
            c = ctx[q.id]
            ans = gen.answer(q.question, c, meta)
            ok, why = gen.grade(q.question, q.answer, q.justification, ans)
            hit = None if c is None else any((ch.doc, p) in set(q.evidence) for ch in c for p in ch.pages)
            _append(path, {"qid": q.id, "variant": variant, "qtype": q.qtype, "answer": ans, "gold": q.answer,
                           "correct": ok, "judge_reason": why, "evidence_retrieved": hit,
                           "context": None if c is None else [[ch.doc, list(ch.pages)] for ch in c]}, lock)

        with ThreadPoolExecutor(workers) as pool:
            for f in [pool.submit(one, q) for q in todo]:
                try:
                    f.result()
                except LLMUnavailable as e:
                    print("stopping:", e)
                    pool.shutdown(cancel_futures=True)
                    return


def corrupt(answer: str) -> str | None:
    """Change the first number by 15% (keeps formatting), or None if the answer has no number."""
    m = re.search(r"-?\d[\d,]*\.?\d*", answer)
    if not m:
        return None
    raw = m.group(0)
    try:
        v = float(raw.replace(",", ""))
    except ValueError:
        return None
    if v == 0:
        return None
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    new = f"{v * 1.15:,.{decimals}f}" if "," in raw else f"{v * 1.15:.{decimals}f}"
    return answer[: m.start()] + new + answer[m.end():]


def judge_check(gen: Generator, workers: int) -> None:
    """The judge should accept every gold answer and reject the same answer with a wrong number."""
    path = GEN / "judge_check.jsonl"
    done, lock = _jsonl(path), threading.Lock()
    todo = [q for q in split(load_questions(), "test") if q.id not in done]

    def one(q):
        gold_ok, _ = gen.grade(q.question, q.answer, q.justification, q.answer)
        bad = corrupt(q.answer)
        bad_ok = gen.grade(q.question, q.answer, q.justification, bad)[0] if bad else None
        _append(path, {"qid": q.id, "accepts_gold": gold_ok, "corrupted": bad, "accepts_corrupted": bad_ok}, lock)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["hyde", "answer", "judge-check"])
    ap.add_argument("--variants", nargs="+", default=["closed_book", "naive", "best", "best_hyde", "oracle"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    gen = Generator()
    try:
        {"hyde": lambda: run_hyde(gen, args.workers), "answer": lambda: run_answers(gen, args.workers, args.variants),
         "judge-check": lambda: judge_check(gen, args.workers)}[args.step]()
    except LLMUnavailable as e:
        print("stopping:", e)
    print(f"real API calls: answers {gen.llm.calls}, judge {gen.judge_llm.calls}")


if __name__ == "__main__":
    main()
